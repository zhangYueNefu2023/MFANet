from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import Downsample, HybridMixerBlock, PhaseFusion, UpBlock


class Encoder(nn.Module):
    def __init__(self, base: int, dropout: float) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8, base * 12]
        self.widths = widths
        self.stage0 = HybridMixerBlock(widths[0], True, False, dropout)
        self.down1 = Downsample(widths[0], widths[1])
        self.stage1 = HybridMixerBlock(widths[1], True, False, dropout)
        self.down2 = Downsample(widths[1], widths[2])
        self.stage2 = HybridMixerBlock(widths[2], True, True, dropout)
        self.down3 = Downsample(widths[2], widths[3])
        self.stage3 = HybridMixerBlock(widths[3], False, True, dropout)
        self.down4 = Downsample(widths[3], widths[4])
        self.bottleneck = HybridMixerBlock(widths[4], False, True, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s0 = self.stage0(x)
        s1 = self.stage1(self.down1(s0))
        s2 = self.stage2(self.down2(s1))
        s3 = self.stage3(self.down3(s2))
        bottleneck = self.bottleneck(self.down4(s3))
        return bottleneck, [s0, s1, s2, s3]


class Decoder(nn.Module):
    def __init__(self, widths: list[int], dropout: float, conditioned: bool) -> None:
        super().__init__()
        self.conditioned = conditioned
        self.ups = nn.ModuleList(
            [
                UpBlock(widths[4], widths[3], widths[3], dropout),
                UpBlock(widths[3], widths[2], widths[2], dropout),
                UpBlock(widths[2], widths[1], widths[1], dropout),
                UpBlock(widths[1], widths[0], widths[0], dropout),
            ]
        )
        target_widths = [widths[3], widths[2], widths[1], widths[0]]
        self.condition = nn.ModuleList(
            [nn.Conv3d(width + 2, width, 1) for width in target_widths]
        ) if conditioned else nn.ModuleList()
        self.auxiliary = nn.ModuleList([nn.Conv3d(widths[2], 1, 1), nn.Conv3d(widths[1], 1, 1)])

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: list[torch.Tensor],
        prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = bottleneck
        features: list[torch.Tensor] = []
        for index, (up, skip) in enumerate(zip(self.ups, reversed(skips))):
            x = up(x, skip)
            if self.conditioned:
                if prior is None:
                    raise ValueError("The tumour decoder requires a liver probability and uncertainty prior")
                resized = F.interpolate(prior, size=x.shape[-3:], mode="trilinear", align_corners=False)
                x = self.condition[index](torch.cat([x, resized], dim=1))
            features.append(x)
        auxiliary = [self.auxiliary[0](features[1]), self.auxiliary[1](features[2])]
        return features[-1], auxiliary


class MFANetV2(nn.Module):
    """Missing-phase-tolerant 3D network for joint liver and tumour segmentation.

    Input phases are ordered NC, AP, PVP, DP. ``liver_valid`` is required during
    training so that tumour-only cases cannot backpropagate through the liver
    probability and anatomical support paths.
    """

    def __init__(
        self,
        in_channels: int = 1,
        phases: int = 4,
        base_channels: int = 36,
        dropout: float = 0.10,
        phase_dropout: float = 0.25,
        support_floor: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 <= phase_dropout < 1.0:
            raise ValueError("phase_dropout must be in [0, 1)")
        if not 0.0 <= support_floor <= 1.0:
            raise ValueError("support_floor must be in [0, 1]")
        self.in_channels = in_channels
        self.phases = phases
        self.support_floor = support_floor
        self.phase_fusion = PhaseFusion(in_channels, base_channels, phases, phase_dropout)
        self.encoder = Encoder(base_channels, dropout)
        widths = self.encoder.widths
        self.liver_decoder = Decoder(widths, dropout, conditioned=False)
        self.tumor_decoder = Decoder(widths, dropout, conditioned=True)
        self.liver_head = nn.Conv3d(widths[0], 1, 1)
        self.tumor_head = nn.Conv3d(widths[0], 1, 1)
        self.boundary_head = nn.Sequential(
            nn.Conv3d(widths[0], widths[0], 3, padding=1),
            nn.GELU(),
            nn.Conv3d(widths[0], 1, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        phase_mask: torch.Tensor,
        liver_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if x.ndim == 5:
            x = x.unsqueeze(2)
        if x.ndim != 6:
            raise ValueError("Input must have shape [B, P, D, H, W] or [B, P, C, D, H, W]")
        if x.shape[1:3] != (self.phases, self.in_channels):
            raise ValueError(f"Expected {self.phases} phases with {self.in_channels} channel(s) each")
        if phase_mask.shape != x.shape[:2]:
            raise ValueError("phase_mask must have shape [B, P]")
        phase_mask = phase_mask.to(device=x.device, dtype=x.dtype)
        if not torch.all((phase_mask == 0) | (phase_mask == 1)):
            raise ValueError("phase_mask must contain only 0 and 1")
        if not torch.all(phase_mask.sum(dim=1) >= 1):
            raise ValueError("Every case must contain at least one available phase")

        if liver_valid is None:
            if self.training:
                raise ValueError("Pass liver_valid during training, including tumour-only cases")
            liver_valid = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        if liver_valid.numel() != x.shape[0]:
            raise ValueError("liver_valid must have one value per case")
        liver_valid = liver_valid.to(device=x.device).reshape(-1).bool()

        fused, phase_weights = self.phase_fusion(x, phase_mask)
        bottleneck, skips = self.encoder(fused)

        liver_features, liver_aux = self.liver_decoder(bottleneck, skips)
        liver_logits = self.liver_head(liver_features)
        liver_probability = torch.sigmoid(liver_logits)
        liver_uncertainty = 4.0 * liver_probability * (1.0 - liver_probability)
        if not torch.any(liver_valid):
            conditioned_liver = liver_probability.detach()
        elif torch.all(liver_valid):
            conditioned_liver = liver_probability
        else:
            has_liver_label = liver_valid[:, None, None, None, None]
            conditioned_liver = torch.where(
                has_liver_label, liver_probability, liver_probability.detach()
            )
        conditioned_uncertainty = 4.0 * conditioned_liver * (1.0 - conditioned_liver)
        prior = torch.cat([conditioned_liver, conditioned_uncertainty], dim=1)

        tumor_features, tumor_aux = self.tumor_decoder(bottleneck, skips, prior)
        tumor_raw_logits = self.tumor_head(tumor_features)
        support = F.max_pool3d(conditioned_liver, kernel_size=9, stride=1, padding=4)
        support = self.support_floor + (1.0 - self.support_floor) * support
        tumor_probability = (torch.sigmoid(tumor_raw_logits) * support).clamp(1e-5, 1.0 - 1e-5)
        tumor_logits = torch.logit(tumor_probability)

        return {
            "liver_logits": liver_logits,
            "tumor_logits": tumor_logits,
            "tumor_raw_logits": tumor_raw_logits,
            "boundary_logits": self.boundary_head(tumor_features),
            "liver_aux": liver_aux,
            "tumor_aux": tumor_aux,
            "phase_weights": phase_weights,
            "liver_uncertainty": liver_uncertainty,
        }


MFANet = MFANetV2
