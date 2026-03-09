import random, torch, os, math, time, datetime
import numpy as np
import wandb
from types import SimpleNamespace
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.cpp_extension import load
from datasets import load_from_disk
from torch.utils.data import DataLoader

# --- CONFIG ALIGNMENT ---
V = 2560      # Matches your Mistral vocab_size
C = 512       # Increased to match your Mistral hidden_size
B = 8         # Batch size (Adjust for L4 VRAM)
T = 512      # Sequence length (RWKV-7 handles long sequences easily)
STEPS = 10000
LR_MAX = 3e-4 # Matches your Mistral learning_rate

# Path to your tokenized data
TOKENIZED_TRAINING_DIR = "/ceph/project/SW10-CausalLM/Ciphers/tokenized_normal/Training"

# --- RWKV-7 KERNEL LOADING ---
HEAD_SIZE = 64 # Optimized for L4
CHUNK_LEN = 16
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3"]
load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], is_python_module=False, verbose=False, extra_cuda_cflags=flags)

# ... [Keep WindBackstepping class and RUN_CUDA_RWKV7g from your original snippet] ...

# --- DATA LOADING ---
class CipherDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path):
        self.hf_dataset = load_from_disk(str(directory_path))
    def __len__(self):
        return len(self.hf_dataset)
    def __getitem__(self, idx):
        # We need input_ids of length T
        ids = self.hf_dataset[idx]["input_ids"]
        if len(ids) > T:
            ids = ids[:T]
        else:
            ids = ids + [0] * (T - len(ids)) # Pad with pad_token_id (0)
        return torch.tensor(ids, dtype=torch.long)

# Setup DataLoader
dataset = CipherDataset(TOKENIZED_TRAINING_DIR)
dataloader = DataLoader(dataset, batch_size=B, shuffle=True, num_samples=None)
data_iter = iter(dataloader)

def get_batch():
    global data_iter
    try:
        return next(data_iter).to('cuda')
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter).to('cuda')

# ... [Keep RWKV_Tmix_x070 and FFN classes from your original snippet] ...

class MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        args = SimpleNamespace(n_embd=C, n_head=C//HEAD_SIZE, head_size=HEAD_SIZE, dim_att=C, n_layer=12) # Scaled up to 12 layers
        self.e = nn.Embedding(V, C)
        self.blocks = nn.ModuleList()
        for i in range(args.n_layer):
            self.blocks.append(RWKV_Tmix_x070(args, i))
            self.blocks.append(FFN(C))
        self.lno = nn.LayerNorm(C)
        self.o = nn.Linear(C, V)

    def forward(self, x):
        x = self.e(x)
        v_first = torch.empty_like(x)
        for i, block in enumerate(self.blocks):
            if isinstance(block, RWKV_Tmix_x070):
                x, v_first = block(x, v_first)
            else:
                x = x + block(x)
        return self.o(self.lno(x))

# --- TRAINING LOOP ---
model = MODEL().to('cuda')
# ... [Weight Decay / Optimizer setup same as your snippet] ...

wandb.init(project="RWKV-7-Cipher", name=f"Goose-C{C}-L12")

for step in range(STEPS):
    batch = get_batch()
    x = batch[:, :-1]
    y = batch[:, 1:]
    
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=0) # Ignore pad tokens

    # Backprop
    opt.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sch.step()

    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")
        wandb.log({"loss": loss.item()}, step=step)