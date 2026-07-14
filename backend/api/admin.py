"""
api/admin.py
----------------------------------------------------
Genkit AI - Admin API

Features
--------
• Model Status
• GPU Status
• Vector Store Status
• Database Statistics
• Dataset Statistics
• Retrain Model
• Reload Knowledge Base
• Health Check

Author : Genkit AI
"""

import os
import sys
import threading
from datetime import datetime

import torch
from fastapi import APIRouter, HTTPException

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from config import (
    MODEL_DIR,
    DATASET_PATH,
    DEVICE,
    logger,
)

from ai.embeddings.embedding import _store


router = APIRouter(
    prefix="/admin",
    tags=["Admin"]
)


# ============================================================
# MODEL STATUS
# ============================================================

def model_files():

    config_file = os.path.join(
        MODEL_DIR,
        "config.json"
    )

    model_file = os.path.join(
        MODEL_DIR,
        "model.pt"
    )

    vocab_file = os.path.join(
        MODEL_DIR,
        "vocab.json"
    )

    return {

        "config": os.path.exists(config_file),

        "model": os.path.exists(model_file),

        "vocab": os.path.exists(vocab_file)

    }


# ============================================================
# GPU INFORMATION
# ============================================================

def gpu_status():

    if not torch.cuda.is_available():

        return {

            "available": False,

            "device": "CPU"

        }

    return {

        "available": True,

        "device": torch.cuda.get_device_name(0),

        "memory_allocated_mb":
            round(torch.cuda.memory_allocated(0) / 1024 / 1024, 2),

        "memory_reserved_mb":
            round(torch.cuda.memory_reserved(0) / 1024 / 1024, 2),

    }
