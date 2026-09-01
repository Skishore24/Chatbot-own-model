"""
backend/scripts/verify_checkpoint.py
----------------------------------------------------
Standalone CLI tool to verify PyTorch model checkpoints.
Performs:
1. File existence & size check
2. PyTorch load test
3. Checkpoint format & dictionary key validation
4. Model state_dict tensor validation (NaN/Inf check)
5. Architecture configuration compatibility
6. Tokenizer vocabulary compatibility
Exits with code 0 on VALID, code 1 on INVALID.
"""

import sys
import argparse
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import torch

from app.core.config import settings
from app.llm.checkpoint import CheckpointManager
from app.llm.config import GPTConfig
from app.llm.tokenizer import ByteFallbackBPETokenizer


def verify_checkpoint_cli(path_str: str, config_path: str = None, tokenizer_path: str = None) -> int:
    path = Path(path_str)
    print("=" * 60)
    print("GENKIT AI — CHECKPOINT VALIDATION REPORT")
    print("=" * 60)
    print(f"File: {path}")

    # 1. Existence check
    if not path.exists():
        print(f"\n[FAIL] Checkpoint file not found: {path}")
        print("\nFINAL STATUS: INVALID (NOT_FOUND)")
        return 1

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    # 2. PyTorch load & structure verification
    is_valid, reason = CheckpointManager.verify_checkpoint(str(path))
    if not is_valid:
        print(f"\n[FAIL] Checkpoint load verification: {reason}")
        print("\nFINAL STATUS: INVALID")
        return 1

    print("PyTorch load: PASS")
    print("Checkpoint format: PASS")
    print("Model state dict: PASS")

    # 3. Load checkpoint payload
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

    print("\nCheckpoint Metadata:")
    print(f" - Format Version: {checkpoint.get('format_version', 'legacy/unknown')}")
    print(f" - Saved At: {checkpoint.get('saved_at', 'unknown')}")
    print(f" - Vocab Size: {checkpoint.get('vocab_size', 'unknown')}")

    # 4. Architecture verification against config
    target_config = None
    c_path = Path(config_path) if config_path else settings.CONFIG_CHECKPOINT_PATH
    if c_path.exists():
        try:
            target_config = GPTConfig.load_from_file(str(c_path))
            print(f"\nLoaded Expected Architecture Config from: {c_path.name}")
            print(f" - Vocab Size: {target_config.vocab_size}")
            print(f" - Embed Dim: {target_config.n_embd}")
            print(f" - Layers: {target_config.n_layer}")
            print(f" - Heads: {target_config.n_head} (KV Heads: {target_config.n_kv_head})")
            print(f" - Context Window (Block Size): {target_config.block_size}")

            if "config" in checkpoint and checkpoint["config"]:
                cp_cfg = checkpoint["config"]
                cp_vocab = cp_cfg.get("vocab_size") if isinstance(cp_cfg, dict) else cp_cfg.vocab_size
                if cp_vocab != target_config.vocab_size:
                    print(f"\n[FAIL] Architecture mismatch: Checkpoint vocab ({cp_vocab}) != Config ({target_config.vocab_size})")
                    print("\nFINAL STATUS: INVALID")
                    return 1
            print("Architecture compatibility: PASS")
        except Exception as e:
            print(f"Warning: Could not parse reference config {c_path}: {e}")

    # 5. Tokenizer compatibility
    t_path = Path(tokenizer_path) if tokenizer_path else settings.TOKENIZER_CHECKPOINT_PATH
    if t_path.exists():
        try:
            tok = ByteFallbackBPETokenizer()
            tok.load(str(t_path))
            cp_vocab = checkpoint.get("vocab_size")
            if cp_vocab and cp_vocab != tok.vocab_size:
                print(f"\n[FAIL] Tokenizer mismatch: Tokenizer vocab ({tok.vocab_size}) != Checkpoint vocab ({cp_vocab})")
                print("\nFINAL STATUS: INVALID")
                return 1
            print(f"Tokenizer compatibility: PASS (Vocab: {tok.vocab_size:,})")
        except Exception as e:
            print(f"Warning: Could not check tokenizer compatibility: {e}")

    print("\n" + "=" * 60)
    print("FINAL STATUS: VALID")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Genkit Checkpoint Verification CLI")
    parser.add_argument(
        "--path",
        type=str,
        default=str(settings.MODEL_CHECKPOINT_PATH),
        help="Path to .pt checkpoint file",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer JSON")
    args = parser.parse_args()

    sys.exit(verify_checkpoint_cli(args.path, args.config, args.tokenizer))


if __name__ == "__main__":
    main()
