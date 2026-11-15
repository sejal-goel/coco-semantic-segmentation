"""The two models (Part B-1).

  deeplabv3plus  - CNN: ResNet-50 encoder + atrous spatial pyramid pooling
                   + decoder (Chen et al., 2018), via segmentation_models_pytorch
  segformer_b3   - Transformer: MiT-B3 hierarchical encoder + all-MLP decoder
                   (Xie et al., 2021), via HuggingFace transformers

Both are wrapped so forward() returns an logits tensor of shape
(N, num_classes, H, W) - identical interface, identical pipeline.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class SegformerWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/mit-b3', num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, x):
        out = self.net(pixel_values=x).logits          # (N, C, H/4, W/4)
        return F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)


class DeepLabWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet',
                                     in_channels=3, classes=num_classes)

    def forward(self, x):
        return self.net(x)


def build_model(name, num_classes=C.NUM_CLASSES):
    if name == 'deeplabv3plus':
        return DeepLabWrapper(num_classes)
    if name == 'segformer_b3':
        return SegformerWrapper(num_classes)
    raise ValueError(f"unknown model {name}")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model, img_size=C.IMG_SIZE, device='cpu'):
    """GFLOPs for one forward pass at the training input size (uses thop)."""
    try:
        from thop import profile
        m = model.to(device).eval()
        x = torch.randn(1, 3, img_size, img_size, device=device)
        with torch.no_grad():
            flops, _ = profile(m, inputs=(x,), verbose=False)
        return flops / 1e9
    except Exception as e:                       # noqa: BLE001
        print(f"FLOP measurement failed ({e}); record as None and mention it.")
        return None


def measure_inference_time(model, img_size=C.IMG_SIZE, device='cuda', n=30):
    """Mean milliseconds per image, batch size 1, after warmup."""
    m = model.to(device).eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(5):
            m(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            m(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
    return dt * 1000.0


def model_card(name, device='cuda', img_size=C.IMG_SIZE):
    """The Part-B-1 reporting table for one model."""
    model = build_model(name)
    total, trainable = count_params(model)
    card = {
        'model': name,
        'input_size': img_size,
        'params_total': int(total),
        'params_trainable': int(trainable),
        'gflops': measure_flops(model, img_size, device),
        'inference_ms_per_image': measure_inference_time(model, img_size, device),
    }
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return card
