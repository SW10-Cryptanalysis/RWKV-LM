import random, torch, os, datetime
import time
import glob
import re
from datetime import timedelta
import wandb
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.cpp_extension import load
from datasets import load_from_disk
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from config import cfg
from model import get_model

# --- RWKV-7 KERNEL LOADING ---
load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], is_python_module=False, verbose=False, extra_cuda_cflags=cfg.cuda_flags)


# --- DATA LOADING ---
class CipherDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path):
        self.hf_dataset = load_from_disk(str(directory_path))
        self.target_len = cfg.sequence_length + 1
    def __len__(self):
        return len(self.hf_dataset)
    def __getitem__(self, idx):
        ids = self.hf_dataset[idx]["input_ids"]
        if len(ids) > self.target_len:
            ids = ids[:self.target_len]
        else:
            ids = ids + [cfg.pad_token_id] * (self.target_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

dataset = CipherDataset(cfg.tokenized_training_dir)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
data_iter = iter(dataloader)

def compute_ser(logits, labels, pad_id):
    # Logits: [B, T, V] -> [B, T]
    preds = logits.argmax(dim=-1)
    
    # Mistral logic: shift to align (standard for causal LM)
    # The model predicts token t+1 at position t
    preds = preds[:, :-1]
    labels = labels[:, 1:]
    
    mask = (labels != pad_id)
    correct = (preds == labels) & mask
    
    total_correct = correct.sum().item()
    total_symbols = mask.sum().item()
    
    return 1.0 - (total_correct / total_symbols) if total_symbols > 0 else 0.0

def get_batch():
    global data_iter
    try:
        return next(data_iter).to('cuda')
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter).to('cuda')

if __name__ == "__main__":
    model = get_model()
    model = torch.compile(model)
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Size: {model_size/1e6:.1f}M parameters")

    # Setup Optimizer & Scheduler (Keep your existing logic)
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

    # Validation DataLoader
    val_dataset = CipherDataset(cfg.tokenized_val_dir)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    wandb.init(project="RWKV-7-Cipher", name=f"Goose-C{cfg.n_embd}-L{cfg.n_layer}-{datetime.datetime.now().strftime('%H%M')}")

    print(f"Starting training for {cfg.steps} steps...")
    start_time = time.time()

    # --- CHECKPOINT LOADING ---
    # Look for files matching the pattern rwkv7_step_*.pth
    ckpt_files = glob.glob("rwkv7_step_*.pth")
    start_step = 1

    if ckpt_files:
        # We use a more specific regex to capture the digits AFTER 'step_'
        steps_found = []
        for f in ckpt_files:
            match = re.search(r'step_(\d+)', f)
            if match:
                steps_found.append(int(match.group(1)))

        if steps_found:
            latest_step = max(steps_found)
            latest_ckpt = f"rwkv7_step_{latest_step}.pth"

            print(f"Found checkpoint! Loading: {latest_ckpt}")
            # Use weights_only=True for extra safety if your torch version supports it
            model.load_state_dict(torch.load(latest_ckpt, map_location='cuda'))
            
            start_step = latest_step + 1
            
            print(f"Fast-forwarding scheduler to step {start_step}...")
            for _ in range(latest_step):
                sch.step()
                
            print(f"Resuming from step {start_step} at LR: {sch.get_last_lr()[0]:.2e}")
    else:
        print("No checkpoints found. Starting from scratch.")

    # --- UPDATED WANDB INIT ---
    wandb.init(
        project="RWKV-7-Cipher", 
        name=f"Goose-C{cfg.n_embd}-L{cfg.n_layer}",
        resume="allow"  # This joins the logs together if you restart the same run
    )

    # --- UPDATED TRAINING LOOP ---
    # Change the range to start from start_step
    for step in range(start_step, cfg.steps + 1):
        model.train()
        batch = get_batch()
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        # Use Autocast for BF16 speedup on L4
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(x)
            # Ensure the loss calculation handles the logit scale correctly
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(), 
                                   y.reshape(-1), 
                                   ignore_index=cfg.pad_token_id)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sch.step()

        # 1. Periodic Logging
        if step % cfg.logging_steps == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            tokens_per_sec = (step * cfg.batch_size * cfg.sequence_length) / elapsed
            eta = timedelta(seconds=int((cfg.steps - step) / steps_per_sec))

            print(f"Step {step}/{cfg.steps} | Loss: {loss.item():.4f} | TPS: {tokens_per_sec:.0f} | ETA: {eta}", flush=True)
            wandb.log({"loss": loss.item(), "lr": sch.get_last_lr()[0], "tps": tokens_per_sec}, step=step)

        # 2. Validation Check (Every 1000 steps)
        if step % 1000 == 0:
            model.eval()
            val_loss = 0
            val_ser = 0
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16): # H100 Speed
                for i, v_batch in enumerate(val_loader):
                    if i > 10: break
                    v_batch = v_batch.to('cuda')
                    vx, vy = v_batch[:, :-1], v_batch[:, 1:]
                    
                    v_logits = model(vx)
                    v_loss = F.cross_entropy(v_logits.reshape(-1, cfg.vocab_size), vy.reshape(-1), ignore_index=cfg.pad_token_id)
                    
                    val_loss += v_loss.item()
                    val_ser += compute_ser(v_logits, vy, cfg.pad_token_id)

            avg_val_ser = val_ser / 11
            wandb.log({"val_loss": val_loss / 11, "val_ser": avg_val_ser}, step=step)

        # 3. Checkpointing (Every 5000 steps)
        if step % 5000 == 0:
            ckpt_name = f"rwkv7_step_{step}.pth"
            torch.save(model.state_dict(), ckpt_name)
            print(f"Saved checkpoint: {ckpt_name}", flush=True)

    # Final Save
    torch.save(model.state_dict(), "rwkv7_cipher_final.pth")
    wandb.finish()