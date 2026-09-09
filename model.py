from __future__ import annotations

import glob
import logging
import math
import os
from pathlib import Path

import clip
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from beam_search import CaptionGenerator

CHECKPOINT_FORMAT_VERSION = 2


class CaptionClipModel(nn.Module):
    """CLIP visual encoder + projection + LSTM language decoder."""

    def __init__(
        self,
        vocab: list[str],
        clip_model: str = "ViT-B/32",
        embedding_size: int = 512,
        rnn_size: int = 512,
        num_layers: int = 2,
        share_embedding_weights: bool = False,
        use_cuda: bool = True,
    ) -> None:
        super().__init__()
        if share_embedding_weights and embedding_size != rnn_size:
            raise ValueError("Shared embedding/classifier weights require embedding_size == rnn_size")

        self.vocab = list(vocab)
        self.encoder_name = clip_model
        self.embedding_size = int(embedding_size)
        self.rnn_size = int(rnn_size)
        self.num_layers = int(num_layers)
        self.share_embedding_weights = bool(share_embedding_weights)

        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(clip_model, device=device, download_root="./models")
        clip_embedding_dim = int(self.clip_model.visual.output_dim)

        self.projection = nn.Linear(clip_embedding_dim, embedding_size)
        self.rnn = nn.LSTM(embedding_size, rnn_size, num_layers=num_layers)
        self.classifier = nn.Linear(rnn_size, len(self.vocab))
        self.embedder = nn.Embedding(len(self.vocab), embedding_size)
        self.dropout = nn.Dropout(0.2)

        if share_embedding_weights:
            self.classifier.weight = self.embedder.weight

        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

    @property
    def config(self) -> dict:
        return {
            "clip_model": self.encoder_name,
            "embedding_size": self.embedding_size,
            "rnn_size": self.rnn_size,
            "num_layers": self.num_layers,
            "share_embedding_weights": self.share_embedding_weights,
        }

    def forward(self, imgs: Tensor, captions: Tensor, lengths: list[int]) -> tuple[Tensor, object]:
        """Return token logits aligned with `[first_word ... EOS]` targets.

        The projected image is the first recurrent input. Its recurrent output
        predicts the first caption token, which matches the decoding path used
        by beam search. Text inputs then predict subsequent tokens and EOS.
        """

        if captions.dtype != torch.long:
            raise TypeError("captions must be torch.long token IDs")
        if len(lengths) != captions.shape[0]:
            raise ValueError("lengths must contain one entry per batch item")

        image_features = self.clip_model.encode_image(imgs).float()
        image_features = self.projection(image_features).unsqueeze(1)
        text_embeddings = self.dropout(self.embedder(captions))
        fused = torch.cat([image_features, text_embeddings], dim=1)
        recurrent_lengths = [length + 1 for length in lengths]
        packed = pack_padded_sequence(fused, recurrent_lengths, batch_first=True, enforce_sorted=True)
        recurrent_output, state = self.rnn(packed)
        prediction: Tensor = self.classifier(recurrent_output.data)
        return prediction, state

    def generate(
        self,
        image,
        *,
        beam_size: int = 3,
        max_caption_length: int = 30,
        length_normalization_factor: float = 0.0,
        eos_token: str = "EOS",
    ) -> list[str]:
        if not torch.is_tensor(image):
            image_tensor = self.preprocess(image)
        else:
            from torchvision.transforms.functional import to_pil_image

            image_tensor = self.preprocess(to_pil_image(image))

        device = next(self.parameters()).device
        image_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor).float()
            image_features = self.projection(image_features).unsqueeze(0)

        generator = CaptionGenerator(
            embedder=self.embedder,
            rnn=self.rnn,
            classifier=self.classifier,
            eos_id=self.vocab.index(eos_token),
            beam_size=beam_size,
            max_caption_length=max_caption_length,
            length_normalization_factor=length_normalization_factor,
        )
        sentences, _ = generator.beam_search(image_features)

        results: list[str] = []
        for sentence in sentences:
            words = [self.vocab[index] for index in sentence if self.vocab[index] != eos_token]
            deduplicated = [word for index, word in enumerate(words) if index == 0 or word != words[index - 1]]
            results.append(" ".join(deduplicated).replace(" , ,", ",").replace(" , ", ", "))
        return results

    def save_checkpoint(self, filepath: str, save_dir: str, keep_count: int, current_epoch: int) -> None:
        """Save a state-dict checkpoint without pickling the live model object."""

        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": int(current_epoch),
            "vocab": self.vocab,
            "config": self.config,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, filepath)
        logging.debug("Checkpoint saved: %s", filepath)

        if keep_count > 0:
            checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pth"))

            def extract_epoch(filename: str) -> int:
                try:
                    return int(Path(filename).stem.replace("checkpoint_epoch_", ""))
                except ValueError:
                    return 0

            checkpoint_files.sort(key=extract_epoch)
            for old_file in checkpoint_files[:-keep_count]:
                try:
                    os.remove(old_file)
                except OSError as exc:
                    logging.warning("Unable to remove %s: %s", old_file, exc)

    @classmethod
    def from_checkpoint(cls, filename: str | Path, *, device: torch.device) -> "CaptionClipModel":
        try:
            payload = torch.load(filename, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(filename, map_location="cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload or "config" not in payload:
            raise ValueError(
                "Unsupported legacy checkpoint. The old format pickled a live model object; retrain or convert it "
                "in a trusted environment instead of loading arbitrary pickle data in the public inference path."
            )
        if int(payload.get("format_version", 0)) != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"Unsupported checkpoint format_version={payload.get('format_version')!r}")
        model = cls(payload["vocab"], use_cuda=device.type == "cuda", **payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        return model

    def finetune_clip(
        self,
        optimizer,
        current_epoch: int,
        total_epochs: int,
        base_clip_lr: float,
        stages: int = 4,
    ) -> int:
        if total_epochs <= 0 or stages <= 0:
            raise ValueError("total_epochs and stages must be positive")
        stage_length = max(1, math.ceil(total_epochs / stages))
        current_stage = min((current_epoch // stage_length) + 1, stages)

        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        if self.encoder_name.startswith("ViT"):
            encoder_blocks = self.clip_model.visual.transformer.resblocks
        elif self.encoder_name.startswith("RN"):
            encoder_blocks = self.clip_model.visual.layer4
        else:
            raise ValueError(f"Unsupported CLIP encoder architecture: {self.encoder_name}")

        total_blocks = len(encoder_blocks)
        if current_stage == 1:
            self.clip_model.eval()
        elif current_stage == 2:
            count = max(1, total_blocks // 4)
            for block in encoder_blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            self.clip_model.train()
        elif current_stage == 3:
            count = max(1, total_blocks // 2)
            for block in encoder_blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            self.clip_model.train()
        else:
            for parameter in self.clip_model.parameters():
                parameter.requires_grad = True
            self.clip_model.train()

        clip_lr = base_clip_lr * (1 - current_stage / (stages + 1))
        optimizer.param_groups[0]["lr"] = clip_lr
        stage_progress = (current_epoch % stage_length) / stage_length
        if stage_progress > 0.5:
            cosine_factor = 0.5 * (math.cos(2.0 * math.pi * (stage_progress - 0.5)) + 1)
            optimizer.param_groups[0]["lr"] = clip_lr * cosine_factor

        logging.info(
            "CLIP fine-tuning stage %s/%s, epoch %s/%s, lr=%g",
            current_stage,
            stages,
            current_epoch,
            total_epochs - 1,
            optimizer.param_groups[0]["lr"],
        )
        return current_stage
