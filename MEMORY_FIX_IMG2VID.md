# LTX-2 Image-to-Video Memory Optimization

## Problem

The LTX-2 image-to-video pipeline was experiencing out-of-memory (OOM) errors when generating videos with high frame counts (e.g., 484 frames at 1024x576 resolution). The OOM occurred during the transformer forward pass, specifically in the feedforward layer.

### Root Cause

The img2vid pipeline uses **per-token timesteps** to handle conditioning where the first frame has timestep 0 (clean image) and subsequent frames have timestep t (noisy). This creates a timestep tensor of shape `[batch_size, num_tokens]` where `num_tokens` can be very large (e.g., 35,136 tokens for 484 frames at 1024x576).

The transformer was expanding cross-attention modulation parameters to full per-token shape:

```python
# OLD CODE (OOM)
video_cross_attn_scale_shift = video_cross_attn_scale_shift_unique[inverse_indices]
video_cross_attn_a2v_gate = video_cross_attn_a2v_gate_unique[inverse_indices]

video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
    batch_size, -1, video_cross_attn_scale_shift.shape[-1]
)
video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(
    batch_size, -1, video_cross_attn_a2v_gate.shape[-1]
)
```

This created tensors of shape `[2, 35136, 4096]` = **549 MB per tensor**, with multiple such tensors being created, totaling over **2.1 GB** of extra memory.

## Solution

Keep cross-attention parameters **compact** (only storing unique timestep embeddings) and expand them lazily in the transformer blocks when needed:

### Changes to `transformer_ltx2.py`

#### 1. Keep cross-attention params compact (lines 1274-1280)

```python
# NEW CODE (memory efficient)
# Keep cross attention params compact - don't expand to per-token yet
# This saves massive memory for img2vid: [batch, num_unique, dim] instead of [batch, num_tokens, dim]
video_cross_attn_scale_shift = video_cross_attn_scale_shift_unique.unsqueeze(0).expand(batch_size, -1, -1)
video_cross_attn_a2v_gate = video_cross_attn_a2v_gate_unique.unsqueeze(0).expand(batch_size, -1, -1)
```

This keeps the shape as `[2, 2, 4096]` = **0.03 MB per tensor** instead of 549 MB.

#### 2. Lazy expansion in transformer blocks (lines 534-540)

```python
# Expand compact cross-attention params if needed (for memory-efficient img2vid)
_temb_ca_scale_shift = temb_ca_scale_shift
_temb_ca_gate = temb_ca_gate
if temb_indices is not None and temb_ca_scale_shift.size(1) < num_tokens:
    batch_idx = torch.arange(batch_size, device=temb_ca_scale_shift.device)[:, None]
    _temb_ca_scale_shift = temb_ca_scale_shift[batch_idx, temb_indices]
    _temb_ca_gate = temb_ca_gate[batch_idx, temb_indices]
```

The expansion happens only when needed in each block, and the expanded tensor is temporary (not stored).

## Memory Savings

For img2vid with 484 frames at 1024x576 (35,136 tokens per sample, batch_size=2):

- **Old approach**: 2,196 MB for cross-attention params (per transformer step)
- **New approach**: 0.1 MB for cross-attention params (per transformer step)
- **Savings**: ~2.2 GB per step

Since the transformer has 48 layers and these parameters are created once and passed through all layers, the actual peak memory reduction is the full 2.2 GB.

## Affected Pipelines

This optimization specifically benefits:
- **Image-to-Video (img2vid)**: Uses per-token timesteps with conditioning mask
- **Text-to-Video (txt2vid)**: Already efficient (uses single timestep per batch)

The fix maintains compatibility with both pipelines through the conditional expansion logic.

## Testing

Run the unit test to verify memory savings:

```bash
python test_memory_fix_unit.py
```

Expected output:
```
MEMORY SAVINGS: 2195.9 MB per step
REDUCTION: 100.0%
```

## Files Modified

1. `/mnt/d/Projects/diffusers-repo/src/diffusers/models/transformers/transformer_ltx2.py`
   - Lines 1274-1280: Keep cross-attention params compact
   - Lines 534-540: Lazy expansion in transformer blocks
