import os
import time
import torch
import logging
import numpy as np
import wandb
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.cpp_extension import load

from config import cfg
from model import get_model  # Your RWKV model loader
from easy_logging import EasyFormatter

# --- INITIAL SETUP ---
handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# L4/Ada Lovelace Optimizations
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def log_environment_details():
    """Environment telemetry ported from Mistral config."""
    logger.info("=== RWKV-7 Execution Environment ===")
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

    # Force 16-token alignment for CUDA kernels
    current_len = input_ids_padded.shape[1]
    remainder = current_len % 16
    if remainder != 0:
        pad_len = 16 - remainder
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
def train():
    log_environment_details()

    load(
        name="wind_backstepping", 
        sources=['cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], 
        is_python_module=False, 
        verbose=True, # Set to True once to verify compilation
        extra_cuda_cflags=cfg.cuda_flags
    )
    
    # 1. Model & Optimization
    model = get_model().to("cuda")
    # RWKV-7 specific: compile kernels
    #model = torch.compile(model) 
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler() # For BF16/FP16 stability
    
    # 2. Data Loading
    data_path = cfg.tokenized_spaced_train_dir if cfg.use_spaces else cfg.tokenized_training_dir
    train_ds = PretokenizedCipherDataset(data_path)
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=safe_pad_collate,
        num_workers=4
    )

    # 3. Telemetry Tracking
    wandb.init(project="RWKV7-Cipher", config=cfg.__dict__)
    start_time = time.time()
    
    logger.info(f"Starting Training. Dataset size: {len(train_ds)}")

    for step, batch in enumerate(tqdm(train_loader)):
        model.train()
        input_ids = batch["input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")

        # Mixed Precision Forward
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size), 
                labels.view(-1), 
                ignore_index=-100
            )
            loss = loss / cfg.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % cfg.grad_accum == 0:
            scaler.unscale_(opt)
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        if step % cfg.logging_steps == 0:
            ser = compute_ser(logits, labels)
            elapsed = time.time() - start_time
            # Tokens per second
            tps = (input_ids.numel() * cfg.grad_accum) / (elapsed / (step + 1))
            vram = torch.cuda.max_memory_allocated() / (1024**3)

            wandb.log({
                "train/loss": loss.item() * cfg.grad_accum,
                "train/ser": ser,
                "train/tps": tps,
                "train/vram_gb": vram
            })

        if step > 0 and step % cfg.save_steps == 0:
            checkpoint_path = cfg.output_dir / f"rwkv_step_{step}.pth"
            torch.save(model.state_dict(), checkpoint_path)

    # Final Save
    torch.save(model.state_dict(), cfg.output_dir / "rwkv7_cipher_final.pth")

if __name__ == "__main__":
    train()