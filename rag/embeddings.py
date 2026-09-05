"""
embeddings.py
=============
Qwen3-Embedding-0.6B wrapper for the RAG system.

Uses raw transformers (no sentence-transformers dependency).
Model: last-token pooling, 1024-dim output, supports 100+ languages.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Local model path
MODEL_PATH = (
    "~/programs/sandyproj/Qwen3-Embedding-0.6B"
)

# Maximum tokens the model supports
MAX_TOKENS = 8192


class EmbeddingModel:
    """
    Load Qwen3-Embedding-0.6B once and reuse for all
    embedding requests.
    """

    def __init__(self, model_path=MODEL_PATH):

        import os

        path = os.path.expanduser(model_path)

        print(
            f"Loading embedding model: {path}..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Embedding model on GPU.")
        else:
            print("Embedding model on CPU.")

        self.model.eval()

        # Hidden size from config
        self.dim = (
            self.model.config.hidden_size
        )

        print(
            f"Embedding model ready "
            f"(dim={self.dim})."
        )

    @torch.no_grad()
    def embed(self, texts):
        """
        Embed a list of texts and return a numpy array
        of shape (len(texts), dim).

        Uses last-token pooling as configured in the
        model's pooling config.
        """

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches to avoid OOM
        batch_size = 32

        for i in range(0, len(texts), batch_size):

            batch = texts[i : i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )

            # Move to same device as model
            device = next(
                self.model.parameters()
            ).device

            encoded = {
                k: v.to(device)
                for k, v in encoded.items()
            }

            outputs = self.model(**encoded)

            # Last token pooling
            embeddings = (
                self._last_token_pool(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                )
            )

            # Normalize
            embeddings = (
                torch.nn.functional.normalize(
                    embeddings, p=2, dim=1
                )
            )

            all_embeddings.append(
                embeddings.cpu().float().numpy()
            )

        return np.concatenate(all_embeddings, axis=0)

    def _last_token_pool(
        self, last_hidden_states, attention_mask
    ):
        """
        Extract the last non-padding token's hidden state.
        """

        sequence_lengths = (
            attention_mask.sum(dim=1) - 1
        )

        batch_size = last_hidden_states.shape[0]

        return last_hidden_states[
            torch.arange(
                batch_size,
                device=last_hidden_states.device,
            ),
            sequence_lengths,
        ]


# Global instance — loaded once, reused everywhere
_embedding_model = None


def get_embedding_model():
    """Return the singleton embedding model."""

    global _embedding_model

    if _embedding_model is None:
        _embedding_model = EmbeddingModel()

    return _embedding_model


def embed_texts(texts):
    """
    Convenience function: embed texts using the
    singleton model.
    """

    model = get_embedding_model()

    return model.embed(texts)


def get_embedding_dim():
    """Return the embedding dimension."""

    model = get_embedding_model()

    return model.dim
