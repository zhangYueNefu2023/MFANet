from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _valid_subset(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    valid = valid.reshape(-1).bool()
    if not torch.any(valid):
        return None
    return prediction[valid], target[valid]


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def focal_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def boundary_target(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool3d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1)


class PartialLabelLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        boundary_weight: float = 0.2,
        deep_supervision_weight: float = 0.3,
        liver_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.deep_supervision_weight = deep_supervision_weight
        self.liver_weight = liver_weight

    def _segmentation_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * soft_dice_loss(logits, target) + self.focal_weight * focal_bce_loss(logits, target)

    def _deep_supervision(
        self,
        logits_list: list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not logits_list:
            return target.sum() * 0.0
        total = target.sum() * 0.0
        for logits in logits_list:
            resized_target = F.interpolate(target, size=logits.shape[-3:], mode="nearest")
            total = total + self._segmentation_loss(logits, resized_target)
        return total / len(logits_list)

    def forward(
        self,
        outputs: dict[str, torch.Tensor | list[torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        tumor_logits = outputs["tumor_logits"]
        liver_logits = outputs["liver_logits"]
        boundary_logits = outputs["boundary_logits"]
        assert isinstance(tumor_logits, torch.Tensor)
        assert isinstance(liver_logits, torch.Tensor)
        assert isinstance(boundary_logits, torch.Tensor)

        anchor = tumor_logits.sum() * 0.0
        tumor_loss = anchor
        liver_loss = anchor
        boundary_loss = anchor
        tumor_aux_loss = anchor
        liver_aux_loss = anchor

        selected = _valid_subset(tumor_logits, batch["tumor"], batch["tumor_valid"])
        if selected is not None:
            tumor_prediction, tumor_target = selected
            tumor_loss = self._segmentation_loss(tumor_prediction, tumor_target)
            boundary_selected = _valid_subset(
                boundary_logits, boundary_target(batch["tumor"]), batch["tumor_valid"]
            )
            assert boundary_selected is not None
            boundary_loss = F.binary_cross_entropy_with_logits(*boundary_selected)
            tumor_aux = outputs.get("tumor_aux", [])
            assert isinstance(tumor_aux, list)
            valid = batch["tumor_valid"].reshape(-1).bool()
            tumor_aux_loss = self._deep_supervision([item[valid] for item in tumor_aux], batch["tumor"][valid])

        selected = _valid_subset(liver_logits, batch["liver"], batch["liver_valid"])
        if selected is not None:
            liver_prediction, liver_target = selected
            liver_loss = self._segmentation_loss(liver_prediction, liver_target)
            liver_aux = outputs.get("liver_aux", [])
            assert isinstance(liver_aux, list)
            valid = batch["liver_valid"].reshape(-1).bool()
            liver_aux_loss = self._deep_supervision([item[valid] for item in liver_aux], batch["liver"][valid])

        total = (
            tumor_loss
            + self.liver_weight * liver_loss
            + self.boundary_weight * boundary_loss
            + self.deep_supervision_weight * (tumor_aux_loss + self.liver_weight * liver_aux_loss)
        )
        components = {
            "loss": float(total.detach().cpu()),
            "tumor": float(tumor_loss.detach().cpu()),
            "liver": float(liver_loss.detach().cpu()),
            "boundary": float(boundary_loss.detach().cpu()),
        }
        return total, components
