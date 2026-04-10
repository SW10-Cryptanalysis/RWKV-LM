import random, torch, os, datetime, time, glob, re
from datetime import timedelta
from tqdm import tqdm # Added TQDM
import wandb
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.cpp_extension import load
from datasets import load_from_disk
from torch.utils.data import DataLoader
# Assuming these are your local files
from pad_collator import PadCollator
from config import cfg
from model import get_model

torch.set_float32_matmul_precision('high')

# --- GPU CONFIG ---
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- RWKV-7 KERNEL LOADING ---
load(name="wind_backstepping", 
     sources=[f'cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], 
     is_python_module=False, verbose=False, extra_cuda_cflags=cfg.cuda_flags)

# --- DATA LOADING ---
class CipherDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path):
        self.hf_dataset = load_from_disk(str(directory_path))
    def __len__(self):
        return len(self.hf_dataset)
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return {"input_ids": item["input_ids"], "labels": item["labels"]}
    
base_collator = PadCollator(pad_token_id=cfg.pad_token_id, max_context=cfg.sequence_length)

def rwkv_collate_fn(batch):
    padded = base_collator(batch)
    t = padded["input_ids"].shape[1]
    if t % cfg.chunk_len != 0:
        pad_len = cfg.chunk_len - (t % cfg.chunk_len)
        padded["input_ids"] = F.pad(padded["input_ids"], (0, pad_len), value=cfg.pad_token_id)
        padded["labels"] = F.pad(padded["labels"], (0, pad_len), value=-100)
        padded["attention_mask"] = F.pad(padded["attention_mask"], (0, pad_len), value=0)
    return padded

# --- METRICS ---
def compute_ser(logits, labels, pad_id):
    preds = logits.argmax(dim=-1)[:, :-1]
    labels = labels[:, 1:]
    mask = (labels != pad_id)
    correct = (preds == labels) & mask
    total_correct = correct.sum().item()
    total_symbols = mask.sum().item()
    return 1.0 - (total_correct / total_symbols) if total_symbols > 0 else 0.0

if __name__ == "__main__":
    # Initialize Model
    model = get_model().to('cuda')
    model = torch.compile(model)
    
    # Optimizer Setup
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if ('.weight' in n or 'emb' in n) and ('ln' not in n):
            decay.append(p)
        else:
            no_decay.append(p)

    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.learning_rate_init, fused=True)
    
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.learning_rate_final)

    # Checkpoint Loading
    ckpt_files = glob.glob("rwkv7_step_*.pth")
    start_step = 1
    if ckpt_files:
        steps_found = [int(re.search(r'step_(\d+)', f).group(1)) for f in ckpt_files if re.search(r'step_(\d+)', f)]
        if steps_found:
            latest_step = max(steps_found)
            print(f"Loading checkpoint from step {latest_step}...")
            model.load_state_dict(torch.load(f"rwkv7_step_{latest_step}.pth", map_location='cuda'))
            start_step = latest_step + 1
            for _ in range(latest_step): sch.step()

    # DataLoaders
    dataset = CipherDataset(cfg.tokenized_training_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=rwkv_collate_fn)
    data_iter = iter(dataloader)
    
    val_dataset = CipherDataset(cfg.tokenized_val_dir)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, collate_fn=rwkv_collate_fn)

    # WandB
    wandb.init(project="RWKV-7-Cipher", name=f"Goose-L{cfg.n_layer}-C{cfg.n_embd}", resume="allow")

    # --- MAIN TRAINING LOOP ---
    print(f"Starting training at step {start_step}...")
    
    # Wrap the range in tqdm for a beautiful progress bar
    progress_bar = tqdm(range(start_step, cfg.steps + 1), desc="Training", dynamic_ncols=True)
    
    for step in progress_bar:
        model.train()
        
        # Get Batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        x, y = batch["input_ids"].to('cuda'), batch["labels"].to('cuda')

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1), ignore_index=-100)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sch.step()

        # Update Progress Bar Postfix
        if step % 10 == 0:
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{sch.get_last_lr()[0]:.2e}"
            })

        # Logging
        if step % cfg.logging_steps == 0:
            wandb.log({"loss": loss.item(), "lr": sch.get_last_lr()[0]}, step=step)

        # Validation
        if step % 1000 == 0:
            model.eval()
            total_val_loss, total_val_ser, count = 0, 0, 0
            
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for i, v_batch in enumerate(val_loader):
                    if i >= 10: break
                    vx, vy = v_batch["input_ids"].to('cuda'), v_batch["labels"].to('cuda')
                    v_logits = model(vx)
                    v_loss = F.cross_entropy(v_logits.view(-1, cfg.vocab_size), vy.view(-1), ignore_index=-100)
                    
                    total_val_loss += v_loss.item()
                    total_val_ser += compute_ser(v_logits, vy, -100)
                    count += 1

            avg_val_loss = total_val_loss / count
            avg_val_ser = total_val_ser / count
            # Use progress_bar.write to avoid messing up the bar rendering
            progress_bar.write(f"Step {step} | Val Loss: {avg_val_loss:.4f} | SER: {avg_val_ser:.4f}")
            wandb.log({"val_loss": avg_val_loss, "val_ser": avg_val_ser}, step=step)

        # Checkpointing
        if step % 5000 == 0:
            torch.save(model.state_dict(), f"rwkv7_step_{step}.pth")

    torch.save(model.state_dict(), "rwkv7_cipher_final.pth")
    wandb.finish()