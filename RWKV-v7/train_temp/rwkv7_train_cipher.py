import random, torch, os, datetime
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

    # --- TRAINING LOOP ---
    for step in range(cfg.steps):
        batch = get_batch()
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        logits = model(x)
        # ignore_index=0 ensures we don't calculate loss on the padding
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=cfg.pad_token_id)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sch.step()

        if step % cfg.logging_steps == 0:
            print(f"Step {step}/{cfg.steps} | Loss: {loss.item():.4f} | LR: {sch.get_last_lr()[0]:.2e}")
            wandb.log({"loss": loss.item(), "lr": sch.get_last_lr()[0]}, step=step)

    torch.save(model.state_dict(), "rwkv7_cipher_final.pth")