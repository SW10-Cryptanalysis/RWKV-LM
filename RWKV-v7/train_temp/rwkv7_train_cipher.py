import random, torch, os, datetime, time
import wandb
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.cpp_extension import load
from datasets import load_from_disk
from torch.utils.data import DataLoader

from config import cfg
from model import get_model

# --- RWKV-7 KERNEL LOADING ---
load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], is_python_module=False, verbose=False, extra_cuda_cflags=cfg.cuda_flags)


# --- DATA LOADING ---
class CipherDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path):
        self.hf_dataset = load_from_disk(str(directory_path))
    def __len__(self):
        return len(self.hf_dataset)
    def __getitem__(self, idx):
        ids = self.hf_dataset[idx]["input_ids"]
        if len(ids) > cfg.sequence_length:
            ids = ids[:cfg.sequence_length]
        else:
            ids = ids + [cfg.pad_token_id] * (cfg.sequence_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

dataset = CipherDataset(cfg.tokenized_training_dir)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
data_iter = iter(dataloader)

def get_batch():
    global data_iter
    try:
        # Move to GPU but don't cast to bf16 yet (labels need to be long)
        return next(data_iter).to('cuda')
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter).to('cuda')
    
if __name__ == "__main__":
    # --- MODEL INITIALIZATION ---
    model = get_model()

    # Setup Decay Groups
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
    ], lr=cfg.learning_rate_init)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.learning_rate_final)

    wandb.init(project="RWKV-7-Cipher", name=f"Goose-C{cfg.n_embd}-L{cfg.n_layer}-{datetime.datetime.now().strftime('%H%M')}")

    # --- TIMING ---
    start_time = time.time()
    step_times = []

    # --- TRAINING LOOP ---
    for step in range(cfg.steps):
        step_start = time.time()
        # New: Loop for accumulation
        loss_accum = 0
        for _ in range(cfg.accumulate_grad_batches):
            batch = get_batch()
            x, y = batch[:, :-1], batch[:, 1:]
            
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=cfg.pad_token_id)
            
            # Scale loss by accumulation steps
            scaled_loss = loss / cfg.accumulate_grad_batches
            scaled_loss.backward()
            loss_accum += loss.item()

        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sch.step()
        opt.zero_grad(set_to_none=True)

        # --- TIMING & LOGGING ---
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Keep rolling average of last 10 steps
        avg_step_time = sum(step_times[-10:]) / len(step_times[-10:])
        steps_remaining = cfg.steps - step - 1
        eta_seconds = steps_remaining * avg_step_time
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

        if step % cfg.logging_steps == 0:
            elapsed = time.time() - start_time
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            print(f"Step {step:5d}/{cfg.steps} | Loss: {loss_accum/cfg.accumulate_grad_batches:.4f} | " 
                  f"Time: {elapsed_str} | ETA: {eta_str} | Avg Speed: {avg_step_time:.2f}s/step")
            wandb.log({
                "loss": loss_accum/cfg.accumulate_grad_batches, 
                "lr": sch.get_last_lr()[0],
                "step_time": avg_step_time,
            }, step=step)

        # Checkpointing
        if step > 0 and step % 1000 == 0:
            ckpt_path = f"rwkv7_step_{step}.pth"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': sch.state_dict(),
            }, ckpt_path)
            print(f"--- Saved Checkpoint: {ckpt_path} ---")

    torch.save(model.state_dict(), "rwkv7_cipher_final.pth")