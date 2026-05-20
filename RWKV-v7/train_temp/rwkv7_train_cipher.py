import os
import time
import torch
import logging
import numpy as np
import contextlib
import wandb
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.cpp_extension import load
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from config import cfg
from model import get_model  # Your RWKV model loader
from easy_logging import EasyFormatter

# --- INITIAL SETUP ---
handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# L4/Ada Lovelace Optimizations
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def log_environment_details():
    logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    logger.info("========================================")

# --- DATASET LOGIC ---
class PretokenizedCipherDataset(Dataset):
    def __init__(self, directory_path: Path) -> None:
        self.hf_dataset = load_from_disk(str(directory_path))
        
    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.hf_dataset[idx]
        input_ids = item["input_ids"][:cfg.max_context]
        labels = item["labels"][:cfg.max_context]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def safe_pad_collate(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=cfg.pad_token_id
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    # Force alignment to cfg.chunk_len (32)
    current_len = input_ids_padded.shape[1]
    remainder = current_len % cfg.chunk_len
    if remainder != 0:
        pad_len = cfg.chunk_len - remainder
        input_ids_padded = torch.nn.functional.pad(input_ids_padded, (0, pad_len), value=cfg.pad_token_id)
        labels_padded = torch.nn.functional.pad(labels_padded, (0, pad_len), value=-100)
        
    return {"input_ids": input_ids_padded, "labels": labels_padded}

# --- METRICS (SER Calculation) ---
def compute_ser(logits, labels):
    """Calculates Symbol Error Rate, ignoring padding."""
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return 1.0 - (correct / total) if total > 0 else 0.0

# --- SIMPLIFIED TRAINING LOOP ---
def train():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl") 
    global_rank = dist.get_rank() 

    if global_rank == 0:
        log_environment_details()
        if not wandb.run:
            wandb.init(project="RWKV7-Cipher-4GPU", config=cfg.__dict__)

    dist.barrier() 

    # Rank-0 compilation guard
    if global_rank == 0:
        load(
            name="wind_backstepping", 
            sources=['cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], 
            is_python_module=False, 
            extra_cuda_cflags=cfg.cuda_flags
        )
    
    dist.barrier() 
    
    if global_rank != 0:
        load(
            name="wind_backstepping", 
            sources=['cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], 
            is_python_module=False, 
            extra_cuda_cflags=cfg.cuda_flags
        )
    
    # Model & DDP Setup
    model = get_model().to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Optimizer Setup
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    
    # Data Engine
    train_ds = PretokenizedCipherDataset(cfg.tokenized_training_dir)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True, seed=cfg.seed)
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        sampler=train_sampler,
        collate_fn=safe_pad_collate,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=4
    )

    global_step = 0

    # Clean Training Loop
    for epoch in range(cfg.epochs): 
        train_sampler.set_epoch(epoch) 
        
        # Use standard tqdm or native data iterator wrapper
        pbar = tqdm(train_loader, total=len(train_loader), disable=(global_rank != 0))
        
        for step, batch in enumerate(pbar):
            model.train()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            is_accumulating = (step + 1) % cfg.grad_accum != 0
            context = model.no_sync() if is_accumulating else contextlib.nullcontext()
            
            with context:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(input_ids)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, cfg.vocab_size), 
                        labels.view(-1), 
                        ignore_index=-100
                    )
                    loss = loss / cfg.grad_accum

                scaler.scale(loss).backward()

            # Optimization Step
            if not is_accumulating:
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                # Rank-0 Logging on optimization steps
                if global_rank == 0 and global_step % cfg.logging_steps == 0:
                    ser = compute_ser(logits, labels)
                    vram = torch.cuda.max_memory_allocated() / (1024**3)
                    wandb.log({
                        "train/loss": loss.item() * cfg.grad_accum,
                        "train/ser": ser,
                        "train/vram_gb": vram,
                        "train/global_step": global_step,
                        "train/epoch": epoch
                    })

    # Final Sync and Save
    dist.barrier()
    if global_rank == 0:
        logger.info("Training complete. Saving final model weights...")
        torch.save(model.module.state_dict(), cfg.output_dir / "rwkv7_cipher_final.pth")
    dist.destroy_process_group()

if __name__ == "__main__":
    train()