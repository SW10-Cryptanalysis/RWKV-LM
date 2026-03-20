from dataclasses import dataclass
from pathlib import Path

# --- PATHS ---
DATA_DIR = Path("/ceph/project/SW10-CausalLM/Ciphers")
OUTPUT_DIR = Path(__file__).parent / "outputs"

TOKENIZED_TRAINING_DIR = DATA_DIR / "tokenized_normal" / "Training"
TOKENIZED_TEST_DIR = DATA_DIR / "tokenized_normal" / "Test"
TOKENIZED_VALIDATION_DIR = DATA_DIR / "tokenized_normal" / "Validation"


@dataclass
class Config:
    # --- ARCHITECTURE ---
    vocab_size: int = 2560  # Matches Mistral tokenizer
    n_embd: int = 1024  # Hidden dimension
    n_layer: int = 12  # Number of transformer layers
    head_size: int = 64  # Size of each attention head
    
    # --- RWKV-7 KERNEL CONFIG ---
    chunk_len: int = 16  # Must divide sequence length evenly
    
    # --- TOKENIZATION ---
    pad_token_id: int = 0
    sep_token_id: int = 2495
    space_token_id: int = 2496
    bos_token_id: int = 2497
    eos_token_id: int = 2498
    char_offset: int = 2499
    
    # --- TRAINING HYPERPARAMETERS ---
    batch_size: int = 16
    sequence_length: int = 512  # Must be multiple of chunk_len (16)
    steps: int = 50000
    learning_rate_init: float = 4e-4  # Initial learning rate
    learning_rate_final: float = 1e-5  # Final learning rate (cosine decay)
    
    # --- GRADIENT OPTIMIZATION ---
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    
    # --- LOGGING ---
    logging_steps: int = 100
    
    # --- PATHS ---
    output_dir: Path = OUTPUT_DIR
    tokenized_training_dir: Path = TOKENIZED_TRAINING_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR
    tokenized_val_dir: Path = TOKENIZED_VALIDATION_DIR
    
    # --- CUDA KERNEL FLAGS ---
    cuda_flags: list = None

    @property
    def dim_att(self) -> int:
        return self.n_embd

    @property
    def dim_ffn(self) -> int:
        # 3.5x is standard for RWKV-7, but 4x is safer for complex logic
        return int(self.n_embd * 3.5)
    
    def __post_init__(self):
        """Validate configuration and set derived values."""
        assert self.sequence_length % self.chunk_len == 0, \
            f"sequence_length ({self.sequence_length}) must be divisible by chunk_len ({self.chunk_len})"
        
        # Set cuda flags if not provided
        if self.cuda_flags is None:
            self.cuda_flags = [
                '-res-usage',
                f'-D_C_={self.head_size}',
                f'-D_CHUNK_LEN_={self.chunk_len}',
                '--use_fast_math',
                '-O3',
                '-Xptxas -O3'
            ]
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)


cfg = Config()
