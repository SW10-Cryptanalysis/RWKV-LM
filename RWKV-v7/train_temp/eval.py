import torch
import torch.nn.functional as F
from config import cfg
from model import get_model
from datasets import load_from_disk
from torch.utils.cpp_extension import load

# Load Kernel
load(name="wind_backstepping", sources=['cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], is_python_module=False, verbose=False, extra_cuda_cflags=cfg.cuda_flags)

def evaluate():
    device = 'cuda'
    model = get_model()
    model.load_state_dict(torch.load("rwkv7_cipher_final.pth", map_location=device))
    model.eval()

    # Setup character mapping (match your other script)
    chars = "abcdefghijklmnopqrstuvwxyz "
    id_to_char = {i + cfg.char_offset: char for i, char in enumerate(chars)}

    val_ds = load_from_disk(str(cfg.tokenized_val_dir))
    # Select a few samples to see variety
    subset = val_ds.select(range(5))

    print(f"\n--- RWKV-7 DECRYPTION EVALUATION ---")
    
    with torch.no_grad():
        for i, item in enumerate(subset):
            # 1. CONSTRUCT THE PROMPT correctly
            # We mimic the other script: [BOS] + [Cipher] + [SEP]
            cipher_ids = [int(x) for x in item["ciphertext"].split()]
            true_plain = item["plaintext"]
            
            input_ids = torch.tensor([cfg.bos_token_id] + cipher_ids + [cfg.sep_token_id], dtype=torch.long, device=device).unsqueeze(0)
            
            # 2. GENERATE
            generated_ids = []
            curr_ids = input_ids.clone()

            for _ in range(100): # Generate up to 100 tokens
                T_curr = curr_ids.size(1)
                
                # RWKV-7 alignment fix (multiple of 16)
                pad_needed = (16 - (T_curr % 16)) % 16
                model_input = F.pad(curr_ids, (0, pad_needed), "constant", 0) if pad_needed > 0 else curr_ids
                
                logits = model(model_input)
                next_token = torch.argmax(logits[:, T_curr - 1, :], dim=-1).unsqueeze(0)
                
                token_id = next_token.item()
                if token_id == cfg.eos_token_id or token_id == 0:
                    break
                    
                generated_ids.append(token_id)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)

            # 3. DECODE
            pred_plain = "".join([id_to_char.get(idx, "?") for idx in generated_ids])
            
            print(f"\nSample {i}:")
            print(f"  True Plain: {true_plain[:50]}...")
            print(f"  Pred Plain: {pred_plain}")

if __name__ == "__main__":
    evaluate()