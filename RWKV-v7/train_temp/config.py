import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path

# --- LOGGING SETUP ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- PATHS ---
DATA_DIR = Path("/ceph/project/SW10-CausalLM/Ciphers")
OUTPUT_DIR = Path(__file__).parent / "outputs"
HOMOPHONE_FILE = "metadata.json"

# Data Directories
TOKENIZED_TRAINING_DIR = DATA_DIR / "tokenized_normal" / "Training"
TOKENIZED_TEST_DIR = DATA_DIR / "tokenized_normal" / "Test"
TOKENIZED_VALIDATION_DIR = DATA_DIR / "tokenized_normal" / "Validation"

@dataclass
class Config:
    # --- ARCHITECTURE ---
    n_embd: int = 1024
    n_layer: int = 12
    head_size: int = 64
    unique_letters: int = 26
    unique_homophones: int = 2503  # Default fallback
    
    # H100 Hopper Optimization: Padded to nearest multiple of 256
    vocab_size: int = 2560 
    
    # --- RWKV-7 KERNEL CONFIG ---
    chunk_len: int = 16 
    
    # --- TRAINING HYPERPARAMETERS ---
    batch_size: int = 64  # Increased for H100
    sequence_length: int = 512
    steps: int = 50000
    learning_rate_init: float = 6e-4  # Slightly higher for larger batch
    learning_rate_final: float = 1e-5
    
    # --- GRADIENT OPTIMIZATION ---
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    
    # --- LOGGING & SYSTEM ---
    logging_steps: int = 10
    output_dir: Path = OUTPUT_DIR
    tokenized_training_dir: Path = TOKENIZED_TRAINING_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR
    tokenized_val_dir: Path = TOKENIZED_VALIDATION_DIR
    
    # --- CUDA KERNEL FLAGS (Targeting H100) ---
    cuda_flags: list = None

    # --- DYNAMIC TOKEN PROPERTIES ---
    @property
    def sep_token_id(self) -> int:
        return self.unique_homophones + 1

    @property
    def space_token_id(self) -> int:
        return self.sep_token_id + 1

    @property
    def bos_token_id(self) -> int:
        return self.space_token_id + 1

    @property
    def eos_token_id(self) -> int:
        return self.bos_token_id + 1

    @property
    def char_offset(self) -> int:
        return self.eos_token_id + 1

    @property
    def dim_att(self) -> int:
        return self.n_embd

    @property
    def dim_ffn(self) -> int:
        return int(self.n_embd * 3.5)
    
    def load_homophones(self) -> None:
        """Load homophone mappings from the metadata file."""
        homophone_path = DATA_DIR / HOMOPHONE_FILE
        if homophone_path.exists():
            try:
                with open(homophone_path) as f:
                    meta = json.load(f)
                    self.unique_homophones = int(meta["max_symbol_id"])
                    logger.info(f"Loaded {self.unique_homophones} homophones from metadata.")
            except Exception as e:
                logger.warning(f"Metadata load failed, using default: {e}")

        # Recalculate vocab_size for H100 Tensor Core alignment (multiple of 256)
        # 178 is the buffer used in your Mistral config
        raw_vocab = self.unique_homophones + self.unique_letters + 178
        self.vocab_size = ((raw_vocab + 255) // 256) * 256
        logger.info(f"Final Vocab Size (Hopper Optimized): {self.vocab_size}")

    def __post_init__(self):
        # 1. Load dynamic metadata first
        self.load_homophones()

        # 2. Validation
        assert self.sequence_length % self.chunk_len == 0, \
            f"sequence_length must be divisible by {self.chunk_len}"
        
        # 3. H100 specific CUDA flags
        if self.cuda_flags is None:
            self.cuda_flags = [
                '-res-usage',
                f'-D_C_={self.head_size}',
                f'-D_CHUNK_LEN_={self.chunk_len}',
                '--use_fast_math',
                '-O3',
                '-Xptxas -O3',
                '--generate-code=arch=compute_90,code=sm_90'
            ]
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

cfg = Config()