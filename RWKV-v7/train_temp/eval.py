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
    model = get_model()
    model.load_state_dict(torch.load("rwkv7_cipher_final.pth", map_location=device))
    model.eval()

    test_ds = load_from_disk(str(Path(cfg.tokenized_training_dir).parent / "Test"))
    subset = test_ds.select(range(5)) # Just check 5 samples

    print(f"\n--- DEBUGGING GENERATION ---")
    
    with torch.no_grad():
        for i, item in enumerate(subset):
            all_ids = item["input_ids"]
            sep_idx = all_ids.index(cfg.sep_token_id)
            input_ids = all_ids[: sep_idx + 1]
            if len(input_ids) > cfg.sequence_length:
                input_ids = input_ids[-(cfg.sequence_length):]
            
            curr_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            # --- STEP 1 LOGITS ---
            T_curr = curr_tensor.size(1)
            pad_needed = (16 - (T_curr % 16)) % 16
            model_input = F.pad(curr_tensor, (0, pad_needed), "constant", 0) if pad_needed > 0 else curr_tensor
            
            logits = model(model_input)
            last_logits = logits[0, T_curr - 1, :]
            
            # Get Top 5 predictions
            probs = F.softmax(last_logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, 5)

            print(f"\nSample {i} (Prompt length: {T_curr})")
            print(f"  Top 5 predicted IDs after SEP:")
            for p, idx in zip(top_probs, top_ids):
                print(f"    ID: {idx.item():<5} | Prob: {p.item():.4f}")

            # See what argmax chose
            token_id = torch.argmax(last_logits).item()
            print(f"  Model chose: {token_id} (EOS is {cfg.eos_token_id}, PAD is {cfg.pad_token_id})")

if __name__ == "__main__":
    evaluate()