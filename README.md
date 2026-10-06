# MFANet core modules

This is the model-only PyTorch implementation of the MFANet architecture described in the revised manuscript. It contains the phase fusion, spatial/spectral/axial encoder, liver and tumor decoders, soft anatomical support, boundary head, and partial-label objective. It contains no dataset loader, image registration, training loop, metric calculation, plotting, or pretrained weights.

## Files

- `mfanet_core/blocks.py`: convolutional, Fourier, axial, phase-fusion, and upsampling blocks
- `mfanet_core/model.py`: encoder, decoders, and MFANet forward pass
- `mfanet_core/losses.py`: class-validity-masked Dice, focal, boundary, and auxiliary losses

Install the sole dependency with `pip install -r requirements.txt`. Import from the repository root:

```python
import torch
from mfanet_core import MFANet, PartialLabelLoss

model = MFANet()  # four phases, 36 base channels
criterion = PartialLabelLoss()

# Phase order: non-contrast, arterial, portal venous, delayed.
# Tensors below illustrate the API; they are not patient data.
x = torch.randn(1, 4, 32, 32, 32)
phase_mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.float32)
liver_valid = torch.tensor([0], dtype=torch.bool)
tumor_valid = torch.tensor([1], dtype=torch.bool)
liver_target = torch.zeros(1, 1, 32, 32, 32)
tumor_target = torch.zeros(1, 1, 32, 32, 32)

model.train()
outputs = model(x, phase_mask, liver_valid=liver_valid)
loss, components = criterion(outputs, {
    "liver": liver_target,
    "tumor": tumor_target,
    "liver_valid": liver_valid,
    "tumor_valid": tumor_valid,
})
loss.backward()
```

The default model is large; a smaller `base_channels` can be used for interface checks. In evaluation mode, `model(x, phase_mask)` returns the same output dictionary without requiring `liver_valid`. Use `torch.sigmoid(outputs["liver_logits"])` and `torch.sigmoid(outputs["tumor_logits"])` for probabilities. `tumor_logits` already includes soft liver support; `tumor_raw_logits` is before support. The input must have at least one available phase per case. Both shapes `[B, 4, D, H, W]` and `[B, 4, 1, D, H, W]` are accepted.

For training, pass `liver_valid` to the model and both `liver_valid` and `tumor_valid` to the loss. A case without a curated liver contour must have `liver_valid=0`; the tumor branch then receives a detached liver prior. The loss ignores that case's liver target. Do not treat an automatically generated or missing organ mask as a curated liver reference.

This package is a code extract aligned to the current manuscript's architecture description. It does not include experimental data, trained checkpoints, or evidence for the manuscript's reported performance.


Synthetic-tensor checks covered a full-width forward pass, missing-phase invariance, phase-dropout retention, mixed-validity gradients, partial-label loss masking, and CPU automatic mixed precision. They do not establish the manuscript's measured performance or GPU memory and runtime figures.
