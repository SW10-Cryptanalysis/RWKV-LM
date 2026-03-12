import torch
import torch.nn.functional as F
from config import cfg
from model import get_model
from rwkv7_train_cipher import CipherDataset
from torch.utils.cpp_extension import load

# 1. Load the Kernel (Required to run the model forward pass)
load(
    name="wind_backstepping", 
    sources=['cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'], 
    is_python_module=False, 
    verbose=False, 
    extra_cuda_cflags=cfg.cuda_flags
)

# 2. Setup Device and Model
device = 'cuda'
# get_model() already handles .to('cuda').float() and architecture setup
model = get_model() 

# Load the weights
checkpoint_path = "rwkv7_cipher_final.pth"
try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"--- Weights Loaded from {checkpoint_path} ---")
except FileNotFoundError:
    print(f"Error: {checkpoint_path} not found. Did the training finish?")
    exit()

model.eval()

# 3. Grab a sample
# Note: Using val_dir instead of training_dir is better for evaluation
dataset = CipherDataset(cfg.tokenized_val_dir)
input_ids = dataset[0].unsqueeze(0).to(device)

# 4. Generation Logic
# Prompt length: We give it the cipher part and ask for the decryption
prompt_length = 256 # Adjust based on where your cipher ends and plain text begins
generated = input_ids[:, :prompt_length]

print(f"--- STARTING GENERATION ---")
with torch.no_grad():
    # Generate up to 256 new tokens
    for i in range(256):
        T_current = generated.size(1)
        
        # The RWKV-7 kernel requires input length to be a multiple of chunk_len (16)
        pad_needed = (16 - (T_current % 16)) % 16
        if pad_needed > 0:
            model_input = F.pad(generated, (0, pad_needed), "constant", cfg.pad_token_id)
        else:
            model_input = generated
        
        # Forward pass
        logits = model(model_input)
        
        # We grab the logit for the last ACTUAL token before padding
        # Shape: (Batch, Sequence, Vocab) -> we want index T_current - 1
        next_token_logits = logits[:, T_current - 1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # Stop if we hit the pad token or a specific end-of-text token
        if next_token.item() == cfg.pad_token_id:
            break

# 5. Decode results
def simple_decode(tokens):
    # Filters out non-printable chars for a cleaner console view
    return "".join([chr(t) if 32 <= t <= 126 else f"<{t}>" for t in tokens])

print(f"\nPROMPT (Input IDs {prompt_length} tokens):")
print(simple_decode(generated[0, :prompt_length].tolist()))

print(f"\nMODEL COMPLETION (Decryption Attempt):")
print(simple_decode(generated[0, prompt_length:].tolist()))