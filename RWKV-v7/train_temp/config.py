import os
import json
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path

# --- LOGGING SETUP ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- CLI ARGUMENTS ---
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--without-spaces",
    action="store_true",
    default=False,
    help="If enabled the model trains without space tokens",
)
cli_args, _ = parser.parse_known_args()

MAX_PLAIN_SPACES = 13077
MAX_PLAIN_NORMAL = 10063

# --- PATHS ---
DATA_DIR = Path(__file__).parent.parent.parent.parent / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "Models" / "RWKV" / "RWKV-mono-10k-nosp"
HOMOPHONE_FILE = "metadata.json"

TOKENIZED_TRAINING_DIR = DATA_DIR / "tokenized_normal_monoalphabetic" / "Training"
TOKENIZED_VALIDATION_DIR = DATA_DIR / "tokenized_normal_monoalphabetic" / "Validation"

@dataclass
class Config:
    """RWKV-7 Configuration."""

    # --- ARCHITECTURE (RWKV-7) ---
    n_embd: int = 768
    n_layer: int = 8 
    head_size: int = 64
    chunk_len: int = 32
    
    # --- CIPHER PROPERTIES ---
    buffer: int = 10
    unique_letters: int = 26
    unique_homophones: int = 0
    vocab_size: int = 0 
    use_spaces: bool = False

    # --- TRAINING HYPERPARAMETERS ---
    batch_size: int = 32      # 
    grad_accum: int = 2      # Effective batch 64
    steps: int = 1000
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 3
    seed: int = 1927843
    
    # --- LOGGING & SYSTEM ---
    logging_steps: int = 10
    save_steps: int = 1000

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    tokenized_training_dir: Path = TOKENIZED_TRAINING_DIR
    tokenized_val_dir: Path = TOKENIZED_VALIDATION_DIR

    # --- CUDA KERNEL FLAGS  ---
    cuda_flags: list = field(default_factory=lambda: [
        '-res-usage', '--use_fast_math', '-O3', '-Xptxas -O3',
        '--generate-code=arch=compute_100,code=sm_100' # Target B200
    ])

    @property
    def max_context(self) -> int:
        """Calculate context window and force alignment for CUDA kernels."""
        if self.use_spaces:
            base_len = (MAX_PLAIN_SPACES * 2) + self.buffer 
        else:
            base_len = (MAX_PLAIN_NORMAL * 2) + self.buffer
        
        # Match alignment to the actual chunk_len (32)
        # Formula: ((base_len + (chunk - 1)) // chunk) * chunk
        aligned_len = ((base_len + (self.chunk_len - 1)) // self.chunk_len) * self.chunk_len
        return aligned_len
    
    @property
    def sequence_length(self) -> int:
        """Alias for the data loader."""
        return self.max_context

    @property
    def pad_token_id(self) -> int:
        return 0

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

    # --- HELPER METHODS ---
    def load_homophones(self) -> None:
        """Determines vocab size and token offsets from metadata."""
        homophone_path = DATA_DIR / HOMOPHONE_FILE
        if not homophone_path.exists():
            raise FileNotFoundError(f"Missing metadata: {homophone_path}")

        try:
            with open(homophone_path) as f:
                meta = json.load(f)
                self.unique_homophones = int(meta["max_symbol_id"])
        except OSError as e:
            raise OSError(f"Could not read file: {homophone_path}") from e
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Invalid or missing 'max_symbol_id' in {homophone_path}",
            ) from e

        # Calculate raw size and pad to 64 for L4 Tensor Core alignment
        raw_vocab = self.unique_homophones + self.unique_letters + 178 # Buffer for special tokens
        self.vocab_size = ((raw_vocab + 63) // 64) * 64
        
        logger.info(f"RWKV-7 Config Initialized. Vocab: {self.vocab_size} | Spaces: {self.use_spaces}")

    def __post_init__(self):
        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add head/chunk definitions to flags dynamically
        self.cuda_flags += [f'-D_C_={self.head_size}', f'-D_CHUNK_LEN_={self.chunk_len}']

# Initialize and Load
cfg = Config()
cfg.load_homophones()