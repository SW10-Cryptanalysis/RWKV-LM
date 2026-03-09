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

# --- MISSING DEFINITIONS FROM RWKV CORE ---
MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

# --- CONFIG ALIGNMENT ---
V = 2560      # Matches your Mistral vocab_size
C = 512       # Matches your Mistral hidden_size
B = 8         # Adjust based on L4 VRAM (8-16 should fit easily)
T = 513       # Must be multiple of CHUNK_LEN (16)
steps = 10000
lr0 = 3e-4    # Initial LR
lr1 = 1e-5    # Final LR (Cosine decay)

TOKENIZED_TRAINING_DIR = "/ceph/project/SW10-CausalLM/Ciphers/tokenized_normal/Training"

# --- RWKV-7 KERNEL LOADING ---
HEAD_SIZE = 64 
CHUNK_LEN = 16
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3"]
load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], is_python_module=False, verbose=False, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape
        assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
        assert all(i.dtype==torch.float32 for i in [w,q,k,v,z,b]) # !!! this simplified demo is in fp32 !!!
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.float32 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def RUN_CUDA_RWKV7g(q, w, k, v, a, b, HEAD_SIZE: int): # Added HEAD_SIZE as argument
    B, T, HC = q.shape
    H = HC // HEAD_SIZE
    # Explicitly view each to avoid TorchScript list-comprehension quirks
    q = q.view(B, T, H, HEAD_SIZE)
    w = w.view(B, T, H, HEAD_SIZE)
    k = k.view(B, T, H, HEAD_SIZE)
    v = v.view(B, T, H, HEAD_SIZE)
    a = a.view(B, T, H, HEAD_SIZE)
    b = b.view(B, T, H, HEAD_SIZE)
    return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)

class FFN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.key = nn.Linear(C, C * 4, bias=False)
        self.value = nn.Linear(C * 4, C, bias=False)
        with torch.no_grad():
            self.value.weight.data.zero_()
            nn.init.orthogonal_(self.key.weight.data, gain=(4**0.5))
    def forward(self, x):
        xx = self.time_shift(x) - x
        x = x + xx * self.x_k
        x = torch.relu(self.key(x)) ** 2
        return self.value(x)

# --- DATA LOADING ---
class CipherDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path):
        self.hf_dataset = load_from_disk(str(directory_path))
    def __len__(self):
        return len(self.hf_dataset)
    def __getitem__(self, idx):
        ids = self.hf_dataset[idx]["input_ids"]
        if len(ids) > T:
            ids = ids[:T]
        else:
            ids = ids + [0] * (T - len(ids))
        return torch.tensor(ids, dtype=torch.long)

dataset = CipherDataset(TOKENIZED_TRAINING_DIR)
dataloader = DataLoader(dataset, batch_size=B, shuffle=True, num_workers=4, pin_memory=True)
data_iter = iter(dataloader)

def get_batch():
    global data_iter
    try:
        return next(data_iter).to('cuda')
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter).to('cuda')

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x
            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = 8 # !!! use max(32, int(round(  (2.5*(C**0.5))  /32)*32)) for LM !!!
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5)

            D_AAA_LORA = 8 # !!! use max(32, int(round(  (2.5*(C**0.5))  /32)*32)) for LM !!!
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            D_MV_LORA = 8 # !!! use max(32, int(round(  (1.7*(C**0.5))  /32)*32)) for LM !!!
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            D_GATE_LORA = 8 # !!! use max(32, int(round(  (5*(C**0.5))  /32)*32)) for LM !!!
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a, self.head_size)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first


class MODEL(nn.Module):
    def __init__(s):
        super().__init__()
        args = SimpleNamespace()
        args.n_head = C//HEAD_SIZE
        args.head_size = HEAD_SIZE
        args.n_embd = C
        args.dim_att = C
        args.n_layer = 2

        s.e=nn.Embedding(V,C)
        
        s.ln1a=nn.LayerNorm(C)
        s.ln1b=nn.LayerNorm(C)
        s.ln1c=nn.LayerNorm(C)
        s.rwkv1=RWKV_Tmix_x070(args,0)
        s.ffn1=FFN(C)

        s.ln2a=nn.LayerNorm(C)
        s.ln2b=nn.LayerNorm(C)
        s.ln2c=nn.LayerNorm(C)
        s.rwkv2=RWKV_Tmix_x070(args,1)
        s.ffn2=FFN(C)

        s.lno=nn.LayerNorm(C)
        s.o=nn.Linear(C,V)

    def forward(s,x):
        x = s.e(x)
       
        xx, v_first = s.rwkv1(s.ln1a(x), torch.empty_like(x))
        x = x + xx
        x = x + s.ffn1(s.ln1b(x))
        xx, v_first = s.rwkv2(s.ln2a(x), v_first)
        x = x + xx
        x = x + s.ffn2(s.ln2b(x))

        x = s.o(s.lno(x))
        return x    


if __name__ == "__main__":
    # --- INITIALIZATION ---
    model = MODEL().to('cuda').float() # Force FP32 for the simplified kernel

    # Setup Decay Groups
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if ('.weight' in n or 'emb' in n) and ('ln' not in n):
            decay.append(p)
        else:
            no_decay.append(p)

    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr1)

    wandb.init(project="RWKV-7-Cipher", name=f"Goose-C{C}-L12-{datetime.datetime.now().strftime('%H%M')}")

    # --- TRAINING LOOP ---
    for step in range(steps):
        batch = get_batch()
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        logits = model(x)
        # ignore_index=0 ensures we don't calculate loss on the padding
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if step % 100 == 0:
            print(f"Step {step}/{steps} | Loss: {loss.item():.4f} | LR: {sch.get_last_lr()[0]:.2e}")
            wandb.log({"loss": loss.item(), "lr": sch.get_last_lr()[0]}, step=step)

    torch.save(model.state_dict(), "rwkv7_cipher_final.pth")