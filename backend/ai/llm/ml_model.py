"""
ai/llm/ml_model.py
----------------------------------------------------
Genkit AI - Custom GPT Language Model

Features
--------
✓ Pure PyTorch implementation
✓ Decoder-only Transformer
✓ Multi-Head Causal Attention
✓ Weight Tying
✓ GPU Optimized
✓ Mixed Precision Ready
✓ Flash Attention Support (PyTorch 2.x)
✓ KV Cache Ready
✓ Production Logging

Author : Genkit AI
"""

import os
import json
import math
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Genkit AI Model")


# ============================================================
# GPT CONFIG
# ============================================================

@dataclass
class GPTConfig:
    """
    Configuration for the GPT model.
    """

    vocab_size: int

    block_size: int = 256

    n_embd: int = 256

    n_head: int = 8

    n_layer: int = 8

    dropout: float = 0.1

    bias: bool = True

    flash_attention: bool = True

    gradient_checkpointing: bool = False


# ============================================================
# Utility Functions
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """
    Returns total trainable parameters.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def init_weights(module: nn.Module):
    """
    Standard GPT initialization.
    """

    if isinstance(module, nn.Linear):

        nn.init.normal_(
            module.weight,
            mean=0.0,
            std=0.02,
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):

        nn.init.normal_(
            module.weight,
            mean=0.0,
            std=0.02,
        )


# ============================================================
# Flash Attention Support
# ============================================================

HAS_FLASH_ATTENTION = hasattr(
    torch.nn.functional,
    "scaled_dot_product_attention",
)

if HAS_FLASH_ATTENTION:

    logger.info(
        "Flash Attention detected."
    )

else:

    logger.info(
        "Using standard attention."
    )
# ============================================================
# Multi-Head Causal Self Attention
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self Attention

    Features
    --------
    ✓ Flash Attention (PyTorch 2.x)
    ✓ Standard Attention fallback
    ✓ Dropout
    ✓ Causal Mask
    ✓ GPU Optimized
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        assert (
            config.n_embd % config.n_head == 0
        ), "Embedding dimension must be divisible by number of heads."

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Combined QKV projection
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )

        # Output projection
        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_flash = (
            config.flash_attention
            and HAS_FLASH_ATTENTION
        )

        if not self.use_flash:

            mask = torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size,
                )
            )

            self.register_buffer(
                "mask",
                mask.view(
                    1,
                    1,
                    config.block_size,
                    config.block_size,
                ),
            )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, x):

        B, T, C = x.size()

        qkv = self.c_attn(x)

        q, k, v = qkv.split(
            self.n_embd,
            dim=2,
        )

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        # ====================================================
        # Flash Attention
        # ====================================================

        if self.use_flash:

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        # ====================================================
        # Standard Attention
        # ====================================================

        else:

            att = (
                q @ k.transpose(-2, -1)
            ) / math.sqrt(self.head_dim)

            att = att.masked_fill(
                self.mask[:, :, :T, :T] == 0,
                float("-inf"),
            )

            att = F.softmax(
                att,
                dim=-1,
            )

            att = self.attn_dropout(att)

            y = att @ v

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(
                B,
                T,
                C,
            )
        )

        y = self.c_proj(y)

        y = self.resid_dropout(y)

        return y
    
    # ============================================================
# Feed Forward Network
# ============================================================

class MLP(nn.Module):
    """
    Transformer Feed Forward Network

    Architecture:
        Linear
            ↓
        GELU
            ↓
        Linear
            ↓
        Dropout
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        hidden_dim = 4 * config.n_embd

        self.fc = nn.Linear(
            config.n_embd,
            hidden_dim,
            bias=config.bias,
        )

        self.proj = nn.Linear(
            hidden_dim,
            config.n_embd,
            bias=config.bias,
        )

        self.dropout = nn.Dropout(
            config.dropout,
        )

    def forward(self, x):

        x = self.fc(x)

        # Faster GELU (PyTorch)
        x = F.gelu(
            x,
            approximate="tanh",
        )

        x = self.proj(x)

        x = self.dropout(x)

        return x


# ============================================================
# Transformer Block
# ============================================================

class Block(nn.Module):
    """
    Transformer Decoder Block

        LayerNorm
            ↓
        Self Attention
            ↓
        Residual

        LayerNorm
            ↓
        Feed Forward
            ↓
        Residual
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(
            config.n_embd,
            bias=config.bias,
        )

        self.attn = CausalSelfAttention(
            config,
        )

        self.ln2 = nn.LayerNorm(
            config.n_embd,
            bias=config.bias,
        )

        self.mlp = MLP(
            config,
        )

    def forward(self, x):

        # Self Attention
        x = x + self.attn(
            self.ln1(x)
        )

        # Feed Forward
        x = x + self.mlp(
            self.ln2(x)
        )

        return x

    # ============================================================
# GPT Model
# ============================================================

class GPT(nn.Module):
    """
    Decoder-only GPT Language Model.

    Features
    --------
    ✓ Weight Tying
    ✓ Flash Attention
    ✓ Mixed Precision Ready
    ✓ GPU Optimized
    ✓ Production Ready
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # ----------------------------------------------------
        # Token Embedding
        # ----------------------------------------------------

        self.transformer = nn.ModuleDict({

            "wte": nn.Embedding(
                config.vocab_size,
                config.n_embd,
            ),

            "wpe": nn.Embedding(
                config.block_size,
                config.n_embd,
            ),

            "drop": nn.Dropout(
                config.dropout,
            ),

            "h": nn.ModuleList([
                Block(config)
                for _ in range(config.n_layer)
            ]),

            "ln_f": nn.LayerNorm(
                config.n_embd,
                bias=config.bias,
            ),
        })

        # ----------------------------------------------------
        # Language Model Head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        # ----------------------------------------------------
        # Weight Tying
        # ----------------------------------------------------

        self.lm_head.weight = self.transformer["wte"].weight

        # ----------------------------------------------------
        # Initialize Weights
        # ----------------------------------------------------

        self.apply(init_weights)

        # GPT-2 scaled initialization
        for name, param in self.named_parameters():

            if name.endswith("proj.weight"):

                torch.nn.init.normal_(

                    param,

                    mean=0.0,

                    std=0.02 / math.sqrt(
                        2 * config.n_layer
                    ),

                )

        logger.info(
            "======================================="
        )

        logger.info(
            " Genkit Custom GPT Initialized"
        )

        logger.info(
            "======================================="
        )

        logger.info(
            "Vocabulary Size : %d",
            config.vocab_size,
        )

        logger.info(
            "Embedding Size : %d",
            config.n_embd,
        )

        logger.info(
            "Layers : %d",
            config.n_layer,
        )

        logger.info(
            "Attention Heads : %d",
            config.n_head,
        )

        logger.info(
            "Context Length : %d",
            config.block_size,
        )

        logger.info(
            "Parameters : %s",
            f"{count_parameters(self):,}",
        )

    # --------------------------------------------------------
    # Number of Parameters
    # --------------------------------------------------------

    def get_num_params(self):

        return count_parameters(self)
    
        # --------------------------------------------------------
    # Forward Pass
    # --------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Args
        ----
        idx:
            Token IDs
            Shape : (B, T)

        targets:
            Target token IDs
            Shape : (B, T)

        Returns
        -------
        logits, loss
        """

        device = idx.device

        B, T = idx.size()

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length ({T}) exceeds block size "
                f"({self.config.block_size})"
            )

        # ----------------------------------------------------
        # Position IDs
        # ----------------------------------------------------

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        token_embeddings = self.transformer["wte"](idx)

        position_embeddings = self.transformer["wpe"](pos)

        x = token_embeddings + position_embeddings

        x = self.transformer["drop"](x)

        # ----------------------------------------------------
        # Transformer Blocks
        # ----------------------------------------------------

        use_checkpointing = getattr(self.config, "gradient_checkpointing", False) and self.training

        for block in self.transformer["h"]:
            if use_checkpointing:
                # Use reentrant=False checkpoint API for standard compatibility
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        x = self.transformer["ln_f"](x)

        # ----------------------------------------------------
        # Language Modeling Head
        # ----------------------------------------------------

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            shift_logits = logits[:, :-1, :].contiguous()

            shift_targets = targets[:, 1:].contiguous()

            loss = F.cross_entropy(

                shift_logits.view(
                    -1,
                    shift_logits.size(-1),
                ),

                shift_targets.view(-1),

                ignore_index=-100,

            )

        return logits, loss
        # --------------------------------------------------------
    # Text Generation
    # --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.10,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args
        ----
        idx : Input token ids (B,T)
        max_new_tokens : Number of tokens to generate
        temperature : Sampling temperature
        top_k : Top-k sampling
        repetition_penalty : Reduce repeated words
        eos_token_id : Stop generation if generated

        Returns
        -------
        Tensor (B,T+N)
        """

        self.eval()

        for _ in range(max_new_tokens):

            # ---------------------------------------------
            # Context Window
            # ---------------------------------------------

            if idx.size(1) > self.config.block_size:

                idx_cond = idx[:, -self.config.block_size:]

            else:

                idx_cond = idx

            # ---------------------------------------------
            # Forward
            # ---------------------------------------------

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            # ---------------------------------------------
            # Repetition Penalty
            # ---------------------------------------------

            if repetition_penalty > 1.0:

                for batch in range(idx.size(0)):

                    previous_tokens = torch.unique(idx[batch])

                    logits[
                        batch,
                        previous_tokens,
                    ] /= repetition_penalty

            # ---------------------------------------------
            # Temperature
            # ---------------------------------------------

            temperature = max(
                temperature,
                1e-6,
            )

            logits = logits / temperature

            # ---------------------------------------------
            # Top-k Sampling
            # ---------------------------------------------

            if top_k is not None:

                top_k = min(
                    top_k,
                    logits.size(-1),
                )

                values, _ = torch.topk(
                    logits,
                    top_k,
                )

                logits[
                    logits < values[:, [-1]]
                ] = float("-inf")

            # ---------------------------------------------
            # Probabilities
            # ---------------------------------------------

            probs = F.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                (
                    idx,
                    next_token,
                ),
                dim=1,
            )

            # ---------------------------------------------
            # EOS Stop
            # ---------------------------------------------

            if (
                eos_token_id is not None
                and (next_token == eos_token_id).all()
            ):
                break

        return idx
    
        # --------------------------------------------------------
    # Configure Optimizer
    # --------------------------------------------------------

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas=(0.9, 0.95),
        device_type: str = "cuda",
    ):
        """
        Create AdamW optimizer.

        Separates parameters that should and
        should not receive weight decay.
        """

        param_dict = {
            pn: p
            for pn, p in self.named_parameters()
            if p.requires_grad
        }

        decay_params = []
        no_decay_params = []

        for name, param in param_dict.items():

            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [

            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },

            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        fused_available = (
            device_type == "cuda"
            and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        )

        optimizer = torch.optim.AdamW(

            optim_groups,

            lr=learning_rate,

            betas=betas,

            fused=fused_available,

        )

        logger.info(
            "Optimizer initialized (%s)",
            "Fused AdamW" if fused_available else "AdamW",
        )

        return optimizer

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    def save_pretrained(
        self,
        save_dir: str,
    ):
        """
        Save model checkpoint.
        """

        os.makedirs(
            save_dir,
            exist_ok=True,
        )

        torch.save(

            self.state_dict(),

            os.path.join(
                save_dir,
                "model.pt",
            ),

        )

        config = {

            "vocab_size": self.config.vocab_size,

            "block_size": self.config.block_size,

            "n_embd": self.config.n_embd,

            "n_head": self.config.n_head,

            "n_layer": self.config.n_layer,

            "dropout": self.config.dropout,

            "bias": self.config.bias,

            "flash_attention": self.config.flash_attention,

        }

        with open(

            os.path.join(
                save_dir,
                "config.json",
            ),

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(
                config,
                f,
                indent=4,
            )

        logger.info(
            "Model saved to %s",
            save_dir,
        )

    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cpu",
    ):
        """
        Load model from checkpoint.
        """

        config_path = os.path.join(
            model_dir,
            "config.json",
        )

        model_path = os.path.join(
            model_dir,
            "model.pt",
        )

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:

            cfg = json.load(f)

        config = GPTConfig(**cfg)

        model = cls(config)

        state = torch.load(
            model_path,
            map_location=device,
        )

        # Adapt keys from trained model to match current class architecture
        new_state = {}
        for k, v in state.items():
            if k.endswith(".attn.bias"):
                continue
            new_key = k
            new_key = new_key.replace(".ln_1.", ".ln1.")
            new_key = new_key.replace(".ln_2.", ".ln2.")
            new_key = new_key.replace(".mlp.c_fc.", ".mlp.fc.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
            new_state[new_key] = v
        state = new_state

        model.load_state_dict(state)

        model.to(device)

        model.eval()

        logger.info(
            "Model loaded from %s",
            model_dir,
        )

        return model

    # --------------------------------------------------------
    # Model Summary
    # --------------------------------------------------------

    def summary(self):
        """
        Print model summary.
        """

        logger.info("====================================")
        logger.info("Genkit GPT Summary")
        logger.info("====================================")

        logger.info(
            "Vocabulary : %d",
            self.config.vocab_size,
        )

        logger.info(
            "Layers : %d",
            self.config.n_layer,
        )

        logger.info(
            "Heads : %d",
            self.config.n_head,
        )

        logger.info(
            "Embedding : %d",
            self.config.n_embd,
        )

        logger.info(
            "Context : %d",
            self.config.block_size,
        )

        logger.info(
            "Parameters : %s",
            f"{self.get_num_params():,}",
        )

        logger.info("====================================")
    
class SimpleWordTokenizer:
    """
    Production-ready word tokenizer.

    Features
    --------
    • Custom vocabulary
    • Fast regex tokenizer
    • Batch encoding
    • Batch decoding
    • Vocabulary training
    • Save / Load
    """

    SPECIAL_TOKENS = [
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",
    ]

    def __init__(self, vocab=None):

        vocab = vocab or {}

        self.vocab = {
            str(word): int(idx)
            for word, idx in vocab.items()
        }

        self.inverse_vocab = {
            idx: word
            for word, idx in self.vocab.items()
        }

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"

        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)
        self.bos_token_id = self.vocab.get(self.bos_token, 2)
        self.eos_token_id = self.vocab.get(self.eos_token, 3)

        self.regex = re.compile(
            r"[A-Za-z0-9_]+|[^\w\s]",
            re.UNICODE,
        )

    # ----------------------------------------------------
    # Vocabulary Size
    # ----------------------------------------------------

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ----------------------------------------------------
    # Tokenization
    # ----------------------------------------------------

    def tokenize(self, text: str):

        if not text:
            return []

        return self.regex.findall(text.lower())

    # ----------------------------------------------------
    # Encode
    # ----------------------------------------------------

    def encode(
        self,
        text,
        add_special_tokens=True,
    ):

        ids = []

        if add_special_tokens:
            ids.append(self.bos_token_id)

        for token in self.tokenize(text):
            ids.append(
                self.vocab.get(
                    token,
                    self.unk_token_id,
                )
            )

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids
        # ----------------------------------------------------
    # Decode
    # ----------------------------------------------------

    def decode(
        self,
        ids,
        skip_special_tokens=True,
    ):

        words = []

        for idx in ids:

            idx = int(idx)

            if skip_special_tokens and idx in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            ):
                continue

            words.append(
                self.inverse_vocab.get(
                    idx,
                    self.unk_token,
                )
            )

        sentence = ""

        punctuation = {
            ".", ",", "!", "?",
            ":", ";",
            ")", "]", "}",
            "'",
            "\""
        }

        for token in words:

            if (
                token in punctuation
                or token.startswith("'")
            ):
                sentence += token
            else:
                if sentence:
                    sentence += " "
                sentence += token

        return sentence.strip()

    # ----------------------------------------------------
    # Batch Encode
    # ----------------------------------------------------

    def batch_encode(
        self,
        texts,
        add_special_tokens=True,
    ):

        return [
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
            )
            for text in texts
        ]

    # ----------------------------------------------------
    # Batch Decode
    # ----------------------------------------------------

    def batch_decode(
        self,
        sequences,
        skip_special_tokens=True,
    ):

        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
            )
            for seq in sequences
        ]

    # ----------------------------------------------------
    # Pad Sequences
    # ----------------------------------------------------

    def pad_sequences(
        self,
        sequences,
        max_length=None,
    ):

        if not sequences:
            return []

        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded = []

        for seq in sequences:

            seq = seq[:max_length]

            seq += [
                self.pad_token_id
            ] * (max_length - len(seq))

            padded.append(seq)

        return padded

    # ----------------------------------------------------
    # Attention Mask
    # ----------------------------------------------------

    def attention_mask(
        self,
        sequences,
    ):

        return [
            [
                0 if token == self.pad_token_id else 1
                for token in seq
            ]
            for seq in sequences
        ]
        # ----------------------------------------------------
    # Save Tokenizer
    # ----------------------------------------------------

    def save_pretrained(self, save_dir: str):

        os.makedirs(save_dir, exist_ok=True)

        vocab_file = os.path.join(
            save_dir,
            "vocab.json",
        )

        with open(
            vocab_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.vocab,
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Tokenizer saved to %s",
            vocab_file,
        )

    # ----------------------------------------------------
    # Load Tokenizer
    # ----------------------------------------------------

    @classmethod
    def from_pretrained(cls, save_dir: str):

        vocab_file = os.path.join(
            save_dir,
            "vocab.json",
        )

        if not os.path.exists(vocab_file):
            raise FileNotFoundError(
                f"Vocabulary not found: {vocab_file}"
            )

        with open(
            vocab_file,
            "r",
            encoding="utf-8",
        ) as f:

            vocab = json.load(f)

        return cls(vocab)

    # ----------------------------------------------------
    # Train Vocabulary
    # ----------------------------------------------------

    @classmethod
    def train_on_texts(
        cls,
        texts,
        min_frequency=1,
    ):

        regex = re.compile(
            r"[A-Za-z0-9_]+|[^\w\s]",
            re.UNICODE,
        )

        frequencies = {}

        for text in texts:

            for token in regex.findall(text.lower()):

                frequencies[token] = (
                    frequencies.get(token, 0) + 1
                )

        sorted_tokens = sorted(
            frequencies.items(),
            key=lambda x: (-x[1], x[0])
        )

        vocab = {}

        for token in cls.SPECIAL_TOKENS:
            vocab[token] = len(vocab)

        for token, freq in sorted_tokens:

            if freq < min_frequency:
                continue

            if token not in vocab:
                vocab[token] = len(vocab)

        logger.info(
            "Vocabulary trained (%d tokens)",
            len(vocab),
        )

        return cls(vocab)

        # ----------------------------------------------------
    # Add New Tokens
    # ----------------------------------------------------

    def add_tokens(self, tokens):

        added = 0

        for token in tokens:

            token = str(token).lower()

            if token not in self.vocab:

                idx = len(self.vocab)

                self.vocab[token] = idx
                self.inverse_vocab[idx] = token

                added += 1

        logger.info(
            "Added %d new tokens to vocabulary.",
            added,
        )

        return added

    # ----------------------------------------------------
    # Convert Token -> ID
    # ----------------------------------------------------

    def convert_tokens_to_ids(self, tokens):

        if isinstance(tokens, str):

            return self.vocab.get(
                tokens,
                self.unk_token_id,
            )

        return [
            self.vocab.get(
                token,
                self.unk_token_id,
            )
            for token in tokens
        ]

    # ----------------------------------------------------
    # Convert ID -> Token
    # ----------------------------------------------------

    def convert_ids_to_tokens(self, ids):

        if isinstance(ids, int):

            return self.inverse_vocab.get(
                ids,
                self.unk_token,
            )

        return [
            self.inverse_vocab.get(
                int(idx),
                self.unk_token,
            )
            for idx in ids
        ]

    # ----------------------------------------------------
    # Vocabulary Length
    # ----------------------------------------------------

    def __len__(self):

        return len(self.vocab)

    # ----------------------------------------------------
    # Contains
    # ----------------------------------------------------

    def __contains__(self, token):

        return token in self.vocab

    # ----------------------------------------------------
    # String Representation
    # ----------------------------------------------------

    def __repr__(self):

        return (
            f"SimpleWordTokenizer("
            f"vocab_size={len(self.vocab)}, "
            f"pad={self.pad_token_id}, "
            f"unk={self.unk_token_id}, "
            f"bos={self.bos_token_id}, "
            f"eos={self.eos_token_id})"
        )
