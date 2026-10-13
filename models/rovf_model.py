"""
RoVF (Recurrence over Video Frames) Model Pipeline.
Combines DINOv2 image backbone with Perceiver Recurrent Transformer.
Reference: Recurrence over Video Frames (RoVF) for Animal Re-identification (IJCV 2026).
Official repository: Strong-AI-Lab/RoVF-Meerkat-Reidentification
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoModel
from .perceiver import Perceiver


class RoVFRecurrentModel(nn.Module):
    """
    RoVF Model: DINOv2 backbone + Perceiver recurrent module.
    """
    def __init__(
        self,
        dino_model_name: str = "facebook/dinov2-small",
        embedding_dim: int = 384,
        latent_dim: int = 384,
        num_latents: int = 257,
        num_heads: int = 8,
        num_transformer_layers: int = 2,
        dropout_rate: float = 0.1,
        output_dim: int = 384,
        is_append_avg_emb: bool = True,
        freeze_image_model: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.output_dim = output_dim
        self.is_append_avg_emb = is_append_avg_emb
        self.device_name = device

        print(f"[RoVF] Loading DINOv2 backbone from HuggingFace ({dino_model_name})...")
        self.image_model = AutoModel.from_pretrained(dino_model_name)
        if freeze_image_model:
            for p in self.image_model.parameters():
                p.requires_grad = False
            self.image_model.eval()

        perceiver_cfg = {
            "embedding_dim": embedding_dim,
            "latent_dim": latent_dim,
            "num_heads": num_heads,
            "num_latents": num_latents,
            "num_transformer_layers": num_transformer_layers,
            "dropout": dropout_rate,
            "output_dim": output_dim,
            "use_video_emb": True,
        }
        self.perceiver = Perceiver(**perceiver_cfg)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Checkpoint tracking
        self.checkpoint_loaded: bool = False
        self.checkpoint_path: Optional[str] = None
        self.checkpoint_metadata: Dict[str, Any] = {}

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Loads a fine-tuned RoVF checkpoint (.pt) if available.
        """
        if not os.path.exists(checkpoint_path):
            print(f"[RoVF] Checkpoint file not found: {checkpoint_path}")
            return False

        try:
            print(f"[RoVF] Loading model checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device_name)

            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                self.load_state_dict(state_dict, strict=False)
            elif isinstance(ckpt, dict) and any(k.startswith("perceiver.") for k in ckpt.keys()):
                self.load_state_dict(ckpt, strict=False)
            else:
                self.load_state_dict(ckpt, strict=False)

            self.checkpoint_loaded = True
            self.checkpoint_path = checkpoint_path
            if "metadata" in ckpt:
                self.checkpoint_metadata = {"raw": str(ckpt["metadata"])}
            print(f"[RoVF] Checkpoint successfully loaded into model.")
            return True
        except Exception as e:
            print(f"[RoVF] Warning: Failed to load checkpoint {checkpoint_path}: {e}")
            return False

    def forward_video(
        self,
        video_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """
        Forward pass over a video clip tensor.
        Args:
            video_tensor: FloatTensor of shape [B, T, 3, 224, 224], normalized to ImageNet stats.
        Returns:
            clip_embedding: [B, output_dim] L2-normalized embedding vector.
            recurrent_weights: List of attention weights per frame (length T).
            per_frame_embs: [B, T, output_dim] per-frame CLS embeddings for frame affinity.
        """
        B, T, C, H, W = video_tensor.shape
        self.perceiver.reset_latents()

        frame_tokens_list: List[torch.Tensor] = []
        frame_cls_list: List[torch.Tensor] = []

        with torch.no_grad():
            for t in range(T):
                frame = video_tensor[:, t, :, :, :]
                outputs = self.image_model(frame)
                tokens = outputs.last_hidden_state  # [B, 257, 384]
                frame_tokens_list.append(tokens)
                # CLS token is at index 0
                frame_cls_list.append(tokens[:, 0, :])

            # Stack frames for average spatial tokens: [B, T, 257, 384]
            stacked_tokens = torch.stack(frame_tokens_list, dim=1)
            # Average image embedding across video: [B, 257, 384]
            avg_image_emb = torch.mean(stacked_tokens, dim=1)

            predictions: List[torch.Tensor] = []
            attention_weights: List[float] = []

            for t in range(T):
                pred, attn_w = self.perceiver(
                    embeddings=frame_tokens_list[t],
                    video_emb=avg_image_emb if t == 0 else None,
                    is_reset_latents=False
                )
                predictions.append(pred)
                # Attention weight for frame t (mean across heads and queries)
                attn_score = float(attn_w.mean().item())
                attention_weights.append(attn_score)

            self.perceiver.reset_latents()

            # Final clip prediction from recurrent head
            clip_emb = predictions[-1]

            # If appending average frame embedding (rovf_s_af configuration)
            if self.is_append_avg_emb:
                # Mean over frames and patch tokens -> [B, 384]
                avg_frame_vec = torch.mean(torch.mean(stacked_tokens, dim=1), dim=1)
                clip_emb = clip_emb + avg_frame_vec

            # Normalize to unit sphere (cosine similarity metric)
            clip_emb_norm = F.normalize(clip_emb, p=2, dim=-1)

            # Per frame CLS representations normalized: [B, T, 384]
            per_frame_cls = torch.stack(frame_cls_list, dim=1)
            per_frame_cls = F.normalize(per_frame_cls, p=2, dim=-1)

            # Normalize attention weights so they sum to 1.0 (for visualization)
            tot_w = sum(attention_weights)
            if tot_w > 0:
                normalized_weights = [round(w / tot_w, 4) for w in attention_weights]
            else:
                normalized_weights = [round(1.0 / T, 4)] * T

            return clip_emb_norm, normalized_weights, per_frame_cls
