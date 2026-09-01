"""
backend/tests/test_checkpoint.py
----------------------------------------------------
Automated tests for CheckpointManager and verify_checkpoint CLI.
Tests:
1. Valid atomic checkpoint save and verification.
2. Corrupted / truncated checkpoint rejection.
3. Invalid state_dict / NaN parameter detection.
4. Architecture & Vocab size mismatch detection.
5. Backup creation (.bak) upon overwrite.
"""

import os
import tempfile
import torch
import pytest
from pathlib import Path

from app.llm.config import GPTConfig
from app.llm.model import EnterpriseGPTModel
from app.llm.checkpoint import CheckpointManager


@pytest.fixture
def dummy_model_and_config():
    config = GPTConfig(
        vocab_size=2084,
        block_size=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        dropout=0.0,
    )
    model = EnterpriseGPTModel(config)
    return model, config


def test_atomic_checkpoint_save_and_verify(dummy_model_and_config):
    model, config = dummy_model_and_config
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "model_test.pt"

        # 1. Save atomically
        success = CheckpointManager.save_atomic(
            filepath=str(target_path),
            model_state_dict=model.state_dict(),
            config=config,
            vocab_size=config.vocab_size,
            training_metadata={"epoch": 1, "val_loss": 2.5},
        )
        assert success is True
        assert target_path.exists()

        # 2. Verify saved checkpoint
        is_valid, msg = CheckpointManager.verify_checkpoint(str(target_path), expected_config=config)
        assert is_valid is True
        assert "valid" in msg.lower()

        # 3. Load safe
        loaded, status = CheckpointManager.load_safe(str(target_path))
        assert loaded is not None
        assert status == "SUCCESS"
        assert loaded["vocab_size"] == 2084
        assert "model_state_dict" in loaded


def test_corrupted_checkpoint_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupted_path = Path(tmpdir) / "corrupted.pt"

        # Write invalid binary data of realistic size (>1024 bytes)
        with open(corrupted_path, "wb") as f:
            f.write(b"PK\x03\x04" + b"X" * 2048 + b"\xef\xbf\xbd")

        is_valid, msg = CheckpointManager.verify_checkpoint(str(corrupted_path))
        assert is_valid is False
        assert "failed" in msg.lower() or "corrupted" in msg.lower() or "archive" in msg.lower()


def test_nan_param_checkpoint_rejection(dummy_model_and_config):
    model, config = dummy_model_and_config
    state_dict = model.state_dict()

    # Inject NaN into a weight tensor
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].clone()
        state_dict[k][0] = float("nan")
        break

    with tempfile.TemporaryDirectory() as tmpdir:
        nan_path = Path(tmpdir) / "nan_model.pt"

        # Direct torch.save of NaN state dict
        torch.save({"model_state_dict": state_dict, "config": config.to_dict(), "vocab_size": 2084}, str(nan_path))

        is_valid, msg = CheckpointManager.verify_checkpoint(str(nan_path))
        assert is_valid is False
        assert "nan" in msg.lower()


def test_vocab_mismatch_detection(dummy_model_and_config):
    model, config = dummy_model_and_config
    expected_config = GPTConfig(vocab_size=5000, n_embd=64, n_layer=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        cp_path = Path(tmpdir) / "mismatch.pt"
        CheckpointManager.save_atomic(
            filepath=str(cp_path),
            model_state_dict=model.state_dict(),
            config=config,
            vocab_size=2084,
        )

        is_valid, msg = CheckpointManager.verify_checkpoint(str(cp_path), expected_config=expected_config)
        assert is_valid is False
        assert "mismatch" in msg.lower()


def test_backup_rotation_on_overwrite(dummy_model_and_config):
    model, config = dummy_model_and_config
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "model.pt"
        bak_path = Path(f"{str(target_path)}.bak")

        # Save first version
        CheckpointManager.save_atomic(
            filepath=str(target_path),
            model_state_dict=model.state_dict(),
            config=config,
            vocab_size=2084,
            training_metadata={"epoch": 1},
        )
        assert target_path.exists()
        assert not bak_path.exists()

        # Save second version (overwrites target and creates backup)
        CheckpointManager.save_atomic(
            filepath=str(target_path),
            model_state_dict=model.state_dict(),
            config=config,
            vocab_size=2084,
            training_metadata={"epoch": 2},
            keep_backup=True,
        )
        assert target_path.exists()
        assert bak_path.exists()

        bak_data = torch.load(str(bak_path), map_location="cpu", weights_only=False)
        assert bak_data["training_metadata"]["epoch"] == 1
