"""Compile and exercise gsplat CUDA before accepting a workstation/notebook setup."""
import json

import torch
from gsplat import rasterization


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is unavailable.")
    device = "cuda:0"
    means = torch.tensor([[0.0, 0.0, 3.0]], device=device, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
    scales = torch.tensor([[0.2, 0.2, 0.2]], device=device, requires_grad=True)
    opacities = torch.tensor([0.9], device=device, requires_grad=True)
    colors = torch.tensor([[0.8, 0.2, 0.1]], device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor([[[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]]], device=device)
    image, alpha, _ = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, 64, 64
    )
    (image.mean() + alpha.mean()).backward()
    print(json.dumps({"gsplat_cuda": True, "image_shape": list(image.shape), "gpu": torch.cuda.get_device_name(0)}))


if __name__ == "__main__":
    main()
