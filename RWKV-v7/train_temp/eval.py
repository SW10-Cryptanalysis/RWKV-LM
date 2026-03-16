import torch
import torch.nn.functional as F
import Levenshtein
import logging
from pathlib import Path
from datasets import load_from_disk
from torch.utils.cpp_extension import load

from config import cfg
from model import get_model

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rwkv_eval")

# --- RWKV-7 KERNEL LOADING ---
load(name="wind_backstepping", sources=['cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], 
     is_python_module=False, verbose=False, extra_cuda_cflags=cfg.cuda_flags)

def decode_prediction(ids: list[int]) -> str:
    """Matches the logic of your Llama script for character decoding."""
    chars = []
    for idx in ids:
        if idx == cfg.pad_token_id: # Usually 0
            continue
        if idx == cfg.sep_token_id: # Stop if we accidentally see another SEP
            break
        if idx == cfg.eos_token_id:
            break
        # Logic: idx - offset + 'a'
        if idx >= cfg.char_offset:
            chars.append(chr(idx - cfg.char_offset + ord("a")))
        elif idx == 32: # Assuming 32 might be space in some configs
            chars.append(" ")
    return "".join(chars)

def evaluate():
    device = 'cuda'
    
    # 1. Load Model
    model = get_model()
    checkpoint = "rwkv7_cipher_final.pth"
    if Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        logger.info(f"Loaded weights from {checkpoint}")
    else:
        logger.error(f"Could not find {checkpoint}")
        return
    model.eval()

    # 2. Load Test Data (Arrow shards)
    test_path = Path(cfg.tokenized_training_dir).parent / "Test" # Adjust if different
    logger.info(f"Loading test data from {test_path}...")
    test_ds = load_from_disk(str(test_path))

    num_samples = min(50, len(test_ds))
    total_ser = 0.0

    logger.info(f"Starting evaluation on {num_samples} samples...")

    for i in range(num_samples):
        item = test_ds[i]
        all_ids = item["input_ids"]

        # 3. Find SEP token to split Cipher from Plain
        try:
            sep_idx = all_ids.index(cfg.sep_token_id)
            input_ids = all_ids[: sep_idx + 1] 
            true_ids = all_ids[sep_idx + 1 :]
            true_plain = decode_prediction(true_ids)
        except ValueError:
            logger.warning(f"Sample {i} missing SEP token. Skipping.")
            continue

        # 4. Generate Autoregressively
        curr_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        generated_ids = []
        
        with torch.no_grad():
            for _ in range(128): # max_new_tokens
                # RWKV-7 CUDA Alignment: Length must be multiple of 16
                T_curr = curr_tensor.size(1)
                pad_needed = (16 - (T_curr % 16)) % 16
                
                if pad_needed > 0:
                    model_input = F.pad(curr_tensor, (0, pad_needed), "constant", 0)
                else:
                    model_input = curr_tensor

                logits = model(model_input)
                
                # Get last non-padded logit
                next_token = torch.argmax(logits[0, T_curr - 1, :], dim=-1)
                
                token_id = next_token.item()
                if token_id == cfg.eos_token_id or token_id == cfg.pad_token_id:
                    break
                
                generated_ids.append(token_id)
                curr_tensor = torch.cat([curr_tensor, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        # 5. Calculate SER
        pred_plain = decode_prediction(generated_ids)
        
        # Levenshtein logic from your Llama script
        min_len = min(len(true_plain), len(pred_plain))
        if min_len > 0:
            dist = Levenshtein.distance(true_plain[:min_len], pred_plain[:min_len])
            ser = dist / min_len
            total_ser += ser
            
            if i % 5 == 0:
                logger.info(f"Sample {i} | SER: {ser:.4f}")
                logger.info(f"  True: {true_plain[:60]}")
                logger.info(f"  Pred: {pred_plain[:60]}")
        else:
            # Handle cases where model generates nothing
            total_ser += 1.0

    avg_ser = total_ser / num_samples
    logger.info("=" * 30)
    logger.info(f"FINAL AVERAGE SYMBOL ERROR RATE (SER): {avg_ser:.4f}")

if __name__ == "__main__":
    evaluate()