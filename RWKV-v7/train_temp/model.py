import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from config import cfg

# --- TORCH JIT DEFINITIONS ---
MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method


class WindBackstepping(torch.autograd.Function):
    """RWKV-7 attention mechanism with wind backstepping algorithm."""
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % cfg.chunk_len == 0, f"sequence_length ({T}) must be divisible by chunk_len ({cfg.chunk_len})"
        assert all(i.dtype == torch.float32 for i in [w, q, k, v, z, b]), "All inputs must be float32"
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b]), "All inputs must be contiguous"
        
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // cfg.chunk_len, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype == torch.float32 for i in [dy]), "Gradient must be float32"
        assert all(i.is_contiguous() for i in [dy]), "Gradient must be contiguous"
        
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db


def RUN_CUDA_RWKV7g(q, w, k, v, a, b, head_size: int):
    """Execute RWKV-7 attention mechanism via CUDA kernel."""
    B, T, HC = q.shape
    H = HC // head_size
    
    # Reshape to (B, T, H, head_size) for kernel processing
    q = q.view(B, T, H, head_size)
    w = w.view(B, T, H, head_size)
    k = k.view(B, T, H, head_size)
    v = v.view(B, T, H, head_size)
    a = a.view(B, T, H, head_size)
    b = b.view(B, T, H, head_size)
    
    return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)


class FFN(nn.Module):
    """Position-wise Feed-Forward Network with time shift."""
    def __init__(self, C):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.key = nn.Linear(C, C * 4, bias=False)
        self.value = nn.Linear(C * 4, C, bias=False)
        
        with torch.no_grad():
            self.value.weight.data.zero_()
            nn.init.orthogonal_(self.key.weight.data, gain=(4 ** 0.5))

    def forward(self, x):
        xx = self.time_shift(x) - x
        x = x + xx * self.x_k
        x = torch.relu(self.key(x)) ** 2
        return self.value(x)


class RWKV_Tmix_x070(MyModule):
    """RWKV-7 Time-Mixing attention layer with variance-preserving initialization."""
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.head_size = cfg.head_size
        self.n_head = cfg.dim_att // self.head_size
        assert cfg.dim_att % self.n_head == 0, "dim_att must be divisible by num_heads"
        
        H = self.n_head
        N = self.head_size
        C = cfg.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (cfg.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / cfg.n_layer)  # 1 to ~0
            
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
                """Orthogonal initialization with proper scaling."""
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
                        raise ValueError(f"Unexpected shape for ortho_init: {shape}")
                    return x

            # Decay LoRA initialization
            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = 8
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

            # Attention scaling LoRA initialization
            D_AAA_LORA = 8
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

            # Value mixing LoRA initialization
            D_MV_LORA = 8
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

            # Gate LoRA initialization
            D_GATE_LORA = 8
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

            # Initialize projection weights
            self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
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
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a, self.head_size)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


class RWKV7Model(nn.Module):
    """
    RWKV-7 language model for sequence-to-sequence cipher decryption.
    Architecture: 2-layer transformer with time-mixing attention and FFN blocks.
    """
    def __init__(self):
        super().__init__()
        C = cfg.n_embd

        self.embedding = nn.Embedding(cfg.vocab_size, C)
        
        # Layer 1
        self.ln1a = nn.LayerNorm(C)
        self.ln1b = nn.LayerNorm(C)
        self.rwkv1 = RWKV_Tmix_x070(0)
        self.ffn1 = FFN(C)

        # Layer 2
        self.ln2a = nn.LayerNorm(C)
        self.ln2b = nn.LayerNorm(C)
        self.rwkv2 = RWKV_Tmix_x070(1)
        self.ffn2 = FFN(C)

        # Output projection
        self.ln_out = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, cfg.vocab_size)

    def forward(self, x):
        """
        Args:
            x: token ids of shape (B, T)
        Returns:
            logits of shape (B, T, vocab_size)
        """
        x = self.embedding(x)
        v_first = None
        
        # Layer 1
        xx, v_first = self.rwkv1(self.ln1a(x), torch.empty_like(x))
        x = x + xx
        x = x + self.ffn1(self.ln1b(x))
        
        # Layer 2
        xx, v_first = self.rwkv2(self.ln2a(x), v_first)
        x = x + xx
        x = x + self.ffn2(self.ln2b(x))

        # Output projection
        x = self.lm_head(self.ln_out(x))
        return x


def apply_custom_initialization(model, config):
    """
    Applies variance-preserving weight initialization to RWKV-7 model.
    Enforces stable standard deviation and depth-dependent residual scaling.
    
    Based on initialization principles:
    - Embeddings and projection layers: std=0.02
    - Residual stream outputs: depth-scaled std
    - LoRA matrices: orthogonal initialization
    """
    std = 0.02
    # Residual scaling factor: 1 / sqrt(2 * Layers)
    scaled_std = std / math.sqrt(2 * config.n_layer)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Base initialization for embeddings and input projections
        if any(nd in name for nd in ["embedding.weight", "receptance.weight", "key.weight", "value.weight"]):
            nn.init.normal_(param, mean=0.0, std=std)
        
        # Depth-dependent scaling for residual stream outputs
        elif any(nd in name for nd in ["output.weight", "lm_head.weight"]):
            nn.init.normal_(param, mean=0.0, std=scaled_std)
        
        # LoRA matrix outputs (already handled in RWKV_Tmix_x070)
        # Just ensure they're contiguous
        elif any(nd in name for nd in ["w2", "a2", "v2", "g2"]):
            pass  # Already orthogonally initialized


def get_model():
    """
    Instantiates a RWKV-7 model configured for cipher decryption.
    
    Returns:
        Initialized RWKV7Model on CUDA in float32 (required for kernel)
    """
    model = RWKV7Model().to('cuda').float()
    
    # Apply variance-preserving initialization
    apply_custom_initialization(model, cfg)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RWKV-7 Model Initialized | Params: {num_params / 1e6:.1f}M")
    print(f"Config: {cfg.n_layer} layers, {cfg.n_embd} hidden size, {cfg.head_size} head size")
    print("Applied variance-preserving initialization (std=0.02, with depth scaling).")
    
    return model
