from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        super().__init__(*layers)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3, dropout=dropout),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SpectralMixer3D(nn.Module):
    """Global mixing through a compact learnable transform in Fourier space."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.freq = nn.Sequential(
            nn.Conv3d(2 * channels, 2 * channels, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(2 * channels, 2 * channels, 1, bias=False),
        )
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-3:]
        spectrum = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")
        features = torch.cat([spectrum.real, spectrum.imag], dim=1)
        mixed = self.freq(features)
        real, imag = mixed.chunk(2, dim=1)
        # Keep the inverse FFT in float32 when convolution runs under mixed precision.
        restored = torch.fft.irfftn(
            torch.complex(real.float(), imag.float()),
            s=spatial_size,
            dim=(-3, -2, -1),
            norm="ortho",
        ).to(dtype=x.dtype)
        return x + self.scale * self.norm(restored)


class AxialGatedMixer3D(nn.Module):
    """Linear-cost long-kernel mixing along the three spatial axes."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.depth = nn.Conv3d(
            channels, channels, (kernel_size, 1, 1), padding=(pad, 0, 0), groups=channels
        )
        self.height = nn.Conv3d(
            channels, channels, (1, kernel_size, 1), padding=(0, pad, 0), groups=channels
        )
        self.width = nn.Conv3d(
            channels, channels, (1, 1, kernel_size), padding=(0, 0, pad), groups=channels
        )
        self.value = nn.Conv3d(channels, channels, 1)
        self.gate = nn.Conv3d(channels, channels, 1)
        self.output = nn.Conv3d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        context = self.depth(z) + self.height(z) + self.width(z)
        update = self.value(context) * torch.sigmoid(self.gate(z))
        return x + self.scale * self.output(update)


class HybridMixerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        use_spectral: bool,
        use_axial: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.local = ResidualConvBlock(channels, dropout)
        self.spectral = SpectralMixer3D(channels) if use_spectral else nn.Identity()
        self.axial = AxialGatedMixer3D(channels) if use_axial else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.axial(self.spectral(self.local(x)))


class PhaseFusion(nn.Module):
    """Shared phase stem followed by mask-aware attention pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        phases: int,
        phase_dropout: float,
    ) -> None:
        super().__init__()
        self.phases = phases
        self.phase_dropout = phase_dropout
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, out_channels, 3),
            ResidualConvBlock(out_channels),
        )
        self.phase_embedding = nn.Parameter(torch.zeros(1, phases, out_channels, 1, 1, 1))
        nn.init.trunc_normal_(self.phase_embedding, std=0.02)
        self.score = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, max(out_channels // 4, 4), 1),
            nn.GELU(),
            nn.Conv3d(max(out_channels // 4, 4), 1, 1),
        )

    def _drop_phases(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.phase_dropout <= 0:
            return mask
        keep = (torch.rand_like(mask) > self.phase_dropout).to(mask.dtype) * mask
        for batch_index in range(mask.shape[0]):
            if keep[batch_index].sum() == 0:
                retained = torch.multinomial(mask[batch_index], 1)
                keep[batch_index, retained] = 1
        return keep

    def forward(self, x: torch.Tensor, phase_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 6:
            raise ValueError("Expected input with shape [B, P, C, D, H, W]")
        batch, phases, channels, depth, height, width = x.shape
        if phases != self.phases:
            raise ValueError(f"Expected {self.phases} phases, received {phases}")
        effective_mask = self._drop_phases(phase_mask.float())
        encoded = self.stem(x.reshape(batch * phases, channels, depth, height, width))
        encoded = encoded.reshape(batch, phases, encoded.shape[1], depth, height, width)
        encoded = encoded + self.phase_embedding
        raw_scores = self.score(encoded.reshape(batch * phases, encoded.shape[2], depth, height, width))
        raw_scores = raw_scores.reshape(batch, phases)
        raw_scores = raw_scores.masked_fill(effective_mask <= 0, -1e4)
        weights = torch.softmax(raw_scores, dim=1) * effective_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused = (encoded * weights[:, :, None, None, None, None]).sum(dim=1)
        return fused, weights


class Downsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels, out_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvNormAct(out_channels + skip_channels, out_channels, 3, dropout=dropout),
            ResidualConvBlock(out_channels, dropout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = self.project(x)
        return self.fuse(torch.cat([x, skip], dim=1))
