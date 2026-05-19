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

# --- DATASET LOGIC
class PretokenizedCipherDataset(Dataset):
    def __init__(self, directory_path: Path) -> None:
        self.hf_dataset = load_from_disk(str(directory_path))
        
    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.hf_dataset[idx]
        # Ensure we stay within RWKV's fixed context length
        input_ids = item["input_ids"][:cfg.max_context]
        labels = item["labels"][:cfg.max_context]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def safe_pad_collate(batch):
    # item["input_ids"] is already a tensor from __getitem__
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=cfg.pad_token_id
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    # FIX: Force alignment to cfg.chunk_len (32) instead of 16
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
    
    # Align shapes (RWKV usually predicts next token)
    # Shift labels and preds if necessary depending on your loss function
    mask = labels != -100
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    
    return 1.0 - (correct / total) if total > 0 else 0.0

# --- TRAINING LOOP ---
# --- UPDATED TRAINING LOOP ---
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

    # Wrap the load call in a rank-0 guard
    if global_rank == 0:
        load(
            name="wind_backstepping", 
            sources=['cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], 
            is_python_module=False, 
            extra_cuda_cflags=cfg.cuda_flags
        )
    
    # Force Ranks 1, 2, and 3 to wait until Rank 0 is done compiling
    dist.barrier() 
    
    # Now Ranks 1-3 load the already-compiled kernel
    if global_rank != 0:
        load(
            name="wind_backstepping", 
            sources=['cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], 
            is_python_module=False, 
            extra_cuda_cflags=cfg.cuda_flags
        )
    
    # 2. Model Setup
    # --- 2. Model Setup (Resume Logic Fix) ---
    model = get_model().to(device)

    print(cfg.output_dir)
    
    checkpoints = sorted(list(cfg.output_dir.glob("rwkv_step_*.pth")), 
                         key=lambda x: int(x.stem.split('_')[-1]))
    
    start_step = 0
    start_epoch = 0
    # Updated Checkpoint Logic for Cross-Architecture loading
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        # Load to CPU first to be safe, then move to the current B200 rank
        checkpoint = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
        
        state_dict = checkpoint["model"]
        
        # Handle DDP vs non-DDP naming mismatch
        # If saved without DDP but loading into DDP:
        # Ensure we strip 'module.' from the checkpoint keys 
        # so they match the base, unwrapped model.
        new_state_dict = {}
        for k, v in state_dict.items():
            # If the checkpoint key starts with 'module.', remove those first 7 characters
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        start_step = checkpoint["step"]
        
        # We will load opt and scaler states AFTER they are initialized below
    
    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # 3. Optimizer Setup
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda')

    # 4. Load Optimizer/Scaler state if resuming
    if checkpoints:
        opt.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        del checkpoint # Free up memory
    
    data_path = cfg.tokenized_training_dir
    train_ds = PretokenizedCipherDataset(data_path)
    
    train_sampler = DistributedSampler(
    train_ds, 
    shuffle=True, 
    drop_last=True,
    seed=cfg.seed # Force deterministic shuffling across resumes
)
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        sampler=train_sampler, # Replaces shuffle=True
        collate_fn=safe_pad_collate,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=4
    )

    start_epoch = start_step // len(train_loader)
    global_step = start_step

    for epoch in range(cfg.epochs): 
        train_sampler.set_epoch(epoch) 
        # Calculate how many steps into the current epoch we are
        if epoch == start_epoch:
            steps_to_skip = start_step % len(train_loader)
        else:
            steps_to_skip = 0
        
        # Create the iterator
        data_iter = iter(train_loader)
        
        # Skip the batches we already processed
        if steps_to_skip > 0:
            if global_rank == 0:
                logger.info(f"Skipping {steps_to_skip} batches to resume data position...")
            for _ in range(steps_to_skip):
                next(data_iter)

        for step in tqdm(range(steps_to_skip, len(train_loader)), 
                         initial=steps_to_skip, 
                         total=len(train_loader),
                         disable=(global_rank != 0)):
            batch = next(data_iter)

            model.train()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Optimization: Disable gradient syncing during accumulation steps
            # Only sync on the step where we actually call the optimizer
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

            # Optimizer Step
            if not is_accumulating:
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                global_step += 1
                opt.zero_grad(set_to_none=True)

            # 4. Rank-0 Logging & Checkpointing
            if global_rank == 0 and global_step % cfg.logging_steps == 0:
                ser = compute_ser(logits, labels)
                vram = torch.cuda.max_memory_allocated() / (1024**3)
                wandb.log({
                    "train/loss": loss.item() * cfg.grad_accum,
                    "train/ser": ser,
                    "train/vram_gb": vram
                })

            if global_rank == 0 and global_step > 0 and global_step % cfg.save_steps == 0:
                checkpoint_path = cfg.output_dir / f"rwkv_step_{global_step}.pth"
                # Save a bundle
                state = {
                    "model": model.module.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": global_step
                }
                torch.save(state, checkpoint_path)
                
                # (Optional) Remove the checkpoint from 2 cycles ago to save disk space
                old_ckpt = cfg.output_dir / f"rwkv_step_{global_step - (cfg.save_steps * 2)}.pth"
                if old_ckpt.exists():
                    old_ckpt.unlink()
        
        start_step = 0

    # Final Sync and Save
    dist.barrier()
    if global_rank == 0:
        torch.save(model.module.state_dict(), cfg.output_dir / "rwkv7_cipher_final.pth")
    dist.destroy_process_group()

if __name__ == "__main__":
    train()