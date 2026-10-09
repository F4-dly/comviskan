"""Dynamic attention layer used by the custom YOLO26 model graphs."""

from __future__ import annotations

import torch
from torch import nn


class DynamicAttention(nn.Module):
    """Input-conditioned channel and spatial attention with a residual path."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid(),
        )
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channel_weights = self.channel_gate(inputs)
        spatial_descriptor = torch.cat(
            (inputs.mean(dim=1, keepdim=True), inputs.amax(dim=1, keepdim=True)), dim=1
        )
        spatial_weights = self.spatial_gate(spatial_descriptor)
        attended = inputs * channel_weights * spatial_weights
        return inputs + self.output(attended)


def register_dynamic_attention() -> None:
    """Expose the custom layer to Ultralytics YAML model parsing."""
    from ultralytics.nn import tasks

    tasks.DynamicAttention = DynamicAttention
