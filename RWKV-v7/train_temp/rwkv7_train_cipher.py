import random, torch, os, datetime
import time
from datetime import timedelta
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

def get_batch():
    global data_iter
    try:
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

    print(f"Starting training for {cfg.steps} steps...")
    start_time = time.time()

    # --- TRAINING LOOP ---

    for step in range(cfg.steps):
        step_start = time.time() # Track individual step for more granular logging
        
        batch = get_batch()
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=cfg.pad_token_id)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sch.step()

        if step % cfg.logging_steps == 0 and step > 0:
            # Time Calculations
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            remaining_steps = cfg.steps - step
            eta_seconds = remaining_steps / steps_per_sec
            
            # Throughput (Tokens per second)
            # tokens = batch_size * sequence_length
            tokens_per_sec = (step * cfg.batch_size * cfg.sequence_length) / elapsed

            fps = steps_per_sec * cfg.batch_size # Frames (samples) per second

            print(f"Step {step}/{cfg.steps} | Loss: {loss.item():.4f} | "
                f"TPS: {tokens_per_sec:.0f} | ETA: {timedelta(seconds=int(eta_seconds))}")
            
            wandb.log({
                "loss": loss.item(), 
                "lr": sch.get_last_lr()[0],
                "tokens_per_sec": tokens_per_sec,
                "fps": fps
            }, step=step)

    total_time = time.time() - start_time
    print(f"Training Complete! Total Time: {timedelta(seconds=int(total_time))}")
    torch.save(model.state_dict(), "rwkv7_cipher_final.pth")