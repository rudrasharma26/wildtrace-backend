"""
Perceiver module for Recurrence over Video Frames (RoVF).
Adapted from Strong-AI-Lab/RoVF-Meerkat-Reidentification models/perceiver_wrapper.py.
Paper: Recurrence over Video Frames (RoVF) for Animal Re-identification (IJCV 2026).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def add_latent_positional_embeddings(latents: torch.Tensor, positional_embeddings: torch.Tensor) -> torch.Tensor:
    if positional_embeddings.size(0) < latents.size(1):
        raise RuntimeError(
            f"positional_embeddings has fewer latent positions than active latents: "
            f"{positional_embeddings.size(0)} < {latents.size(1)}"
        )
    active_pe = positional_embeddings[: latents.size(1)]
    return latents + active_pe.unsqueeze(0).expand(latents.size(0), -1, -1)


class TransformerEncoder(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        norm = nn.LayerNorm(normalized_shape=latent_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=norm)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.transformer_encoder(src, mask=mask)


class CrossAttention(nn.Module):
    def __init__(self, latent_dim: int, data_dim: int, num_heads: int):
        super().__init__()
        self.data_dim = data_dim
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(data_dim, latent_dim)
        self.value_proj = nn.Linear(data_dim, latent_dim)
        self.attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)

    def forward(self, latents: torch.Tensor, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.query_proj(latents)
        if len(data.size()) == 2:
            data = data.view(data.size(0), -1, self.data_dim)
        key = self.key_proj(data)
        value = self.value_proj(data)
        attn_output, attn_weights = self.attention(query, key, value)
        return attn_output, attn_weights


class Perceiver(nn.Module):
    """
    Perceiver recurrent architecture used in RoVF.
    Cross-attends learned/recurrent memory latents to incoming frame spatial tokens,
    then processes through a multi-layer Transformer Encoder.
    """
    def __init__(
        self,
        embedding_dim: int = 384,
        latent_dim: int = 384,
        num_heads: int = 8,
        num_latents: int = 257,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 384,
        use_video_emb: bool = True,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.use_video_emb = use_video_emb

        self.latents_p = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(num_latents, latent_dim)) * (latent_dim**0.5)
        )
        self.latents: Optional[torch.Tensor] = None
        self.video_emb: Optional[torch.Tensor] = None

        pe_nlatents = num_latents + 1 if use_video_emb else num_latents
        self.positional_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(pe_nlatents, latent_dim))
        )

        self.embedding_cross_attention = CrossAttention(latent_dim, embedding_dim, num_heads)
        self.transformer = TransformerEncoder(latent_dim, num_heads, num_transformer_layers, dropout)

        if output_dim is not None:
            self.output_layer = nn.Linear(latent_dim * num_latents, output_dim)
        else:
            self.output_layer = None

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def reset_latents(self):
        self.latents = None
        self.video_emb = None

    def forward(
        self,
        embeddings: torch.Tensor,
        video_emb: Optional[torch.Tensor] = None,
        add_pos_emb: bool = True,
        is_reset_latents: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = embeddings.size(0)

        if self.latents is None:
            if video_emb is not None:
                latents = self.layer_norm2(video_emb)
            else:
                latents = self.latents_p.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            latents = self.latents

        if add_pos_emb:
            latents = add_latent_positional_embeddings(latents, self.positional_embeddings)

        latents = self.dropout1(latents)

        embeddings_norm = self.layer_norm1(embeddings)
        cross_out, attn_weights = self.embedding_cross_attention(latents, embeddings_norm)
        latents = latents + cross_out
        latents = self.dropout3(latents)

        latents = self.transformer(latents)

        if self.output_layer is not None:
            output = self.output_layer(latents.reshape(latents.size(0), -1))
        else:
            output = latents[:, 0, :]

        if is_reset_latents:
            self.reset_latents()
        else:
            self.latents = latents

        return output, attn_weights
