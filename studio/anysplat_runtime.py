"""Windows-friendly, PLY-only adapter for the pinned official AnySplat source."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from studio.config import ANYSPLAT_DIR, ANYSPLAT_ROOT

SH_C0 = 0.28209479177387814
INPUT_SIZE = 448


def _install_inference_shims(torch):
    """Replace optional Linux rendering/training extensions with native Torch paths.

    AnySplat inference needs xFormers attention and torch-scatter in the upstream
    environment. PyTorch SDPA/scatter_reduce provide equivalent inference operations
    on current Windows CUDA wheels. The unused CUDA renderer is stubbed because this
    application exports Gaussians and renders them in WebGL.
    """
    # Avoid executing the training dataset package __init__ when inference imports
    # its lightweight type/shim modules.
    if "src.dataset" not in sys.modules:
        dataset = types.ModuleType("src.dataset")
        dataset.__path__ = [str(ANYSPLAT_ROOT / "src" / "dataset")]
        sys.modules["src.dataset"] = dataset

    if "torch_scatter" not in sys.modules:
        scatter = types.ModuleType("torch_scatter")

        def scatter_add(src, index, dim=0, dim_size=None):
            if dim != 0:
                raise NotImplementedError("AnySplat compatibility scatter supports dim=0")
            size = int(index.max().item()) + 1 if index.numel() else 0
            size = max(size, int(dim_size or 0))
            out = torch.zeros((size, *src.shape[1:]), dtype=src.dtype, device=src.device)
            return out.index_add_(0, index.long(), src)

        def scatter_max(src, index, dim=0, dim_size=None):
            if dim != 0:
                raise NotImplementedError("AnySplat compatibility scatter supports dim=0")
            size = int(index.max().item()) + 1 if index.numel() else 0
            size = max(size, int(dim_size or 0))
            shape = (size, *src.shape[1:])
            values = torch.full(shape, -torch.inf, dtype=src.dtype, device=src.device)
            expanded = index.long().view(-1, *([1] * (src.ndim - 1))).expand_as(src)
            values.scatter_reduce_(0, expanded, src, reduce="amax", include_self=True)
            return values, torch.zeros(shape, dtype=torch.long, device=src.device)

        scatter.scatter_add, scatter.scatter_max = scatter_add, scatter_max
        sys.modules["torch_scatter"] = scatter

    if "xformers.ops" not in sys.modules:
        xformers = types.ModuleType("xformers")
        ops = types.ModuleType("xformers.ops")

        def memory_efficient_attention(q, k, v, op=None, p=0.0, **_):
            # Upstream tensors are B,N,H,D; SDPA uses B,H,N,D.
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=float(p),
            )
            return out.transpose(1, 2)

        marker = type("_FlashOp", (), {})
        ops.memory_efficient_attention = memory_efficient_attention
        ops.fmha = types.SimpleNamespace(flash=types.SimpleNamespace(FwOp=marker, BwOp=marker))
        xformers.ops = ops
        sys.modules["xformers"] = xformers
        sys.modules["xformers.ops"] = ops

    # The decoder only exists for upstream video rendering. This app never calls it.
    decoder_name = "src.model.decoder.decoder_splatting_cuda"
    if decoder_name not in sys.modules:
        from dataclasses import dataclass

        decoder = types.ModuleType(decoder_name)

        @dataclass
        class DecoderSplattingCUDACfg:
            name: str
            background_color: list[float]
            make_scale_invariant: bool

        class DecoderSplattingCUDA(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg

            def forward(self, *_args, **_kwargs):
                raise RuntimeError("The upstream CUDA video decoder is disabled; use the local WebGL viewer.")

        decoder.DecoderSplattingCUDACfg = DecoderSplattingCUDACfg
        decoder.DecoderSplattingCUDA = DecoderSplattingCUDA
        sys.modules[decoder_name] = decoder

    # Imported by the shared Gaussian adapter, but not called by AnySplat's pose-free path.
    rotation_name = "src.misc.sh_rotation"
    if rotation_name not in sys.modules:
        rotation = types.ModuleType(rotation_name)
        rotation.rotate_sh = lambda harmonics, _rotations: harmonics
        sys.modules[rotation_name] = rotation


def _source_ready():
    return (ANYSPLAT_ROOT / "src" / "model" / "model" / "anysplat.py").is_file()


def model_ready():
    return _source_ready() and (ANYSPLAT_DIR / "config.json").is_file() and (ANYSPLAT_DIR / "model.safetensors").is_file()


def preprocess_image(path: Path):
    """Apply EXIF orientation, RGB conversion, resize and centre crop to 448 square."""
    import torch

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        scale = INPUT_SIZE / min(width, height)
        resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
        left = (resized.width - INPUT_SIZE) // 2
        top = (resized.height - INPUT_SIZE) // 2
        image = resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def select_inputs(paths: list[Path], limit: int) -> list[Path]:
    """Evenly preserve coverage when a VRAM budget uses fewer uploaded frames."""
    if limit >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, limit, dtype=int)
    return [paths[int(i)] for i in indices]


def automatic_view_limit(total_vram_gb: float, uploaded: int) -> int:
    # Conservative defaults for a 1B-parameter model at fixed 448x448 input.
    if total_vram_gb <= 8.5:
        safe = 2
    elif total_vram_gb <= 12.5:
        safe = 4
    elif total_vram_gb <= 16.5:
        safe = 6
    elif total_vram_gb <= 24.5:
        safe = 10
    else:
        safe = 16
    return max(1, min(uploaded, safe))


def load_model(device="cuda"):
    import torch
    from safetensors.torch import load_file

    if not model_ready():
        raise RuntimeError("AnySplat is not installed. Run Setup NVIDIA Workstation.cmd, then retry.")
    if str(ANYSPLAT_ROOT) not in sys.path:
        sys.path.insert(0, str(ANYSPLAT_ROOT))
    _install_inference_shims(torch)

    # Prevent an unnecessary second multi-gigabyte VGGT download. The AnySplat
    # checkpoint already contains the full fine-tuned encoder state.
    from src.model.encoder.vggt.models.vggt import VGGT
    VGGT.from_pretrained = classmethod(lambda cls, *_args, **_kwargs: cls())
    from src.model.model.anysplat import AnySplat

    config = json.loads((ANYSPLAT_DIR / "config.json").read_text(encoding="utf-8"))
    encoder_cfg = AnySplat._decode_arg(AnySplat._hub_mixin_init_parameters["encoder_cfg"].annotation, config["encoder_cfg"])
    decoder_cfg = AnySplat._decode_arg(AnySplat._hub_mixin_init_parameters["decoder_cfg"].annotation, config["decoder_cfg"])
    with torch.device("meta"):
        model = AnySplat(encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg)
    state = load_file(str(ANYSPLAT_DIR / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"AnySplat checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected tensors")
    del state
    # Non-persistent buffers are absent from safetensors and remain on meta.
    for module in model.modules():
        for name, buffer in tuple(module._buffers.items()):
            if buffer is None or not buffer.is_meta:
                continue
            if name == "_resnet_mean":
                value = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
            elif name == "_resnet_std":
                value = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
            elif name == "sh_mask":
                value = torch.ones(buffer.shape, dtype=torch.float32)
                for degree in range(1, int(value.numel() ** 0.5)):
                    value[degree**2:(degree + 1)**2] = 0.1 * 0.25**degree
            else:
                raise RuntimeError(f"Unsupported AnySplat non-persistent buffer: {name}")
            module._buffers[name] = value
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def to_viewer_gaussians(gaussians, max_gaussians=2_000_000):
    """Convert official AnySplat tensors to the app's viewer/PLY array."""
    means = gaussians.means[0].detach().float().cpu().numpy()
    scales = gaussians.scales[0].detach().float().cpu().numpy()
    rotations = gaussians.rotations[0].detach().float().cpu().numpy()  # x,y,z,w
    harmonics = gaussians.harmonics[0, :, :, 0].detach().float().cpu().numpy()
    opacities = gaussians.opacities[0].detach().float().cpu().numpy().reshape(-1)

    valid = np.isfinite(means).all(1) & np.isfinite(scales).all(1)
    valid &= np.isfinite(rotations).all(1) & np.isfinite(harmonics).all(1) & np.isfinite(opacities)
    valid &= (scales > 0).all(1) & (opacities > 0.005)
    indices = np.flatnonzero(valid)
    if len(indices) > max_gaussians:
        # Preserve the strongest surface evidence and keep output/viewer memory bounded.
        keep = np.argpartition(opacities[indices], -max_gaussians)[-max_gaussians:]
        indices = np.sort(indices[keep])
    if not len(indices):
        raise ValueError("AnySplat produced no valid visible Gaussians.")

    means, scales = means[indices], scales[indices]
    rotations, harmonics, opacities = rotations[indices], harmonics[indices], opacities[indices]
    g = np.zeros((len(indices), 16), dtype=np.float32)
    g[:, :3] = means * np.array([1, -1, -1], dtype=np.float32)
    g[:, 3] = np.clip(opacities, 0.001, 0.999)
    g[:, 4:7] = scales
    # Global 180-degree X rotation multiplied by upstream x,y,z,w quaternions.
    x, y, z, w = rotations.T
    g[:, 8:12] = np.stack((-x, w, -z, y), axis=-1)
    g[:, 12:15] = np.clip(harmonics * SH_C0 + 0.5, 0, 1)
    return g
