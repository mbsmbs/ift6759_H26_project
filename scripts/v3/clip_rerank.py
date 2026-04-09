from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class ClipCandidateScore:
    candidate_index: int
    prompt_index: int
    clip_score: float


class CLIPReranker:
    

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        hf_model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = None

        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.processor = None

        try:
            import open_clip 

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.backend = "open_clip"
            print(f"[CLIP] Using open_clip backend on {self.device}")
            return
        except Exception as e:
            print(f"[CLIP] open_clip unavailable, falling back to transformers. Reason: {e}")

        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            self.model = CLIPModel.from_pretrained(hf_model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(hf_model_name)
            self.model.eval()
            self.backend = "transformers"
            print(f"[CLIP] Using transformers backend on {self.device}")
            return
        except Exception as e:
            raise RuntimeError(
                "Could not load any CLIP backend. Install either open_clip_torch or transformers."
            ) from e

    @torch.no_grad()
    def score_crops_against_prompts(
        self,
        crops: Sequence[Image.Image],
        prompts: Sequence[str],
    ) -> np.ndarray:
        """
        Returns:
            scores shape = [num_crops, num_prompts]
        """
        if len(crops) == 0:
            return np.zeros((0, len(prompts)), dtype=np.float32)
        if len(prompts) == 0:
            raise ValueError("Prompts list is empty.")

        if self.backend == "open_clip":
            image_tensors = torch.stack([self.preprocess(img) for img in crops]).to(self.device)
            text_tokens = self.tokenizer(list(prompts)).to(self.device)

            image_features = self.model.encode_image(image_tensors)
            text_features = self.model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ text_features.T
            scores = logits.softmax(dim=-1).detach().cpu().numpy().astype(np.float32)
            return scores

        if self.backend == "transformers":
            inputs = self.processor(
                text=list(prompts),
                images=list(crops),
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            logits = outputs.logits_per_image
            scores = logits.softmax(dim=-1).detach().cpu().numpy().astype(np.float32)
            return scores

        raise RuntimeError("Unknown CLIP backend.")

    def choose_best_prompt_per_crop(
        self,
        crops: Sequence[Image.Image],
        prompts: Sequence[str],
    ) -> List[ClipCandidateScore]:
        scores = self.score_crops_against_prompts(crops=crops, prompts=prompts)
        best: List[ClipCandidateScore] = []

        for candidate_index in range(scores.shape[0]):
            prompt_index = int(np.argmax(scores[candidate_index]))
            clip_score = float(scores[candidate_index, prompt_index])
            best.append(
                ClipCandidateScore(
                    candidate_index=candidate_index,
                    prompt_index=prompt_index,
                    clip_score=clip_score,
                )
            )
        return best

    def rank_candidates(
        self,
        crops: Sequence[Image.Image],
        prompts: Sequence[str],
    ) -> List[ClipCandidateScore]:
        best = self.choose_best_prompt_per_crop(crops=crops, prompts=prompts)
        best = sorted(best, key=lambda x: x.clip_score, reverse=True)
        return best


def crop_xyxy_from_bgr(
    image_bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Image.Image:
    h, w = image_bgr.shape[:2]
    x1i = max(0, min(int(round(x1)), w - 1))
    y1i = max(0, min(int(round(y1)), h - 1))
    x2i = max(0, min(int(round(x2)), w - 1))
    y2i = max(0, min(int(round(y2)), h - 1))

    if x2i <= x1i:
        x2i = min(w - 1, x1i + 1)
    if y2i <= y1i:
        y2i = min(h - 1, y1i + 1)

    crop_bgr = image_bgr[y1i:y2i, x1i:x2i]
    if crop_bgr.size == 0:
        crop_bgr = image_bgr

    crop_rgb = crop_bgr[:, :, ::-1]
    return Image.fromarray(crop_rgb)