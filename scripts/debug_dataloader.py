#!/usr/bin/env python3
import io
import os
import sys

import numpy as np
import torch
import webdataset as wds
from omegaconf import OmegaConf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_modules import WebSignalDataModule
from sjepa.extractors import ConvFeatureExtractor
from sjepa.masking import (
    MultiBlockMaskMaker,
    RandomClusterMaskMaker,
    RandomMaskMaker,
    TimeInverseBlockMasker,
)

MASKERS = {
    "random-masker": RandomMaskMaker,
    "random-cluster-masker": RandomClusterMaskMaker,
    "time-inverse-masker": TimeInverseBlockMasker,
    "block-masker": MultiBlockMaskMaker,
    "multi-block-masker": MultiBlockMaskMaker,
}


def _print_tensor(name, value):
    if value is None:
        print(f"{name}: None")
        return
    if isinstance(value, torch.Tensor):
        print(
            f"{name}: shape={tuple(value.shape)} dtype={value.dtype} "
            f"min={value.min().item():.4g} max={value.max().item():.4g}"
        )
        return
    print(f"{name}: type={type(value)}")


def _mask_line(mask, on="X", off="."):
    return "".join(on if v else off for v in mask.tolist())


def _visualize_masks(ctx_masks, target_indices, max_len=120):
    # Expect shapes like:
    # ctx_masks: [B, S, T] or [B, T]
    # target_indices: [B, S, G, T] or [B, G, T]
    ctx = ctx_masks
    tgt = target_indices

    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(1)  # -> [B, 1, T]
    if tgt.dim() == 3:
        tgt = tgt.unsqueeze(1)  # -> [B, 1, G, T]

    b = 0
    s = 0
    ctx_line = ctx[b, s]
    tgt_lines = tgt[b, s]

    T = ctx_line.shape[-1]
    show_len = min(T, max_len)

    idx_line = "".join(str(i % 10) for i in range(show_len))
    print("mask index:", idx_line)
    # ctx_mask is True for masked positions. Show masked vs visible explicitly.
    print("context   :", _mask_line(ctx_line[:show_len], on="M", off="V"))
    for g in range(tgt_lines.shape[0]):
        print(f"target {g:02d}:", _mask_line(tgt_lines[g, :show_len], on="T", off="."))
    print("")
    print(
        f"masked_context_total: {int(ctx_line.sum().item())} / {int(ctx_line.numel())}"
    )
    for g in range(tgt_lines.shape[0]):
        print(
            f"masked_target_{g:02d}_total: "
            f"{int(tgt_lines[g].sum().item())} / {int(tgt_lines[g].numel())}"
        )


def _effective_stride_and_rf(conv_layers_spec):
    """
    Compute effective stride and receptive field (in input samples) for a stack of
    1D conv layers with no padding/dilation.

    conv_layers_spec: list[(dim, kernel, stride)]
    """
    stride = 1
    receptive_field = 1
    for _, kernel, layer_stride in conv_layers_spec:
        receptive_field = receptive_field + (kernel - 1) * stride
        stride *= layer_stride
    return stride, receptive_field


def _span_seconds(
    num_tokens: int, stride_seconds: float, receptive_field_seconds: float
) -> float:
    if num_tokens <= 0:
        return 0.0
    return receptive_field_seconds + (num_tokens - 1) * stride_seconds


def peek_raw(path):
    ds = (
        wds.WebDataset(path, resampled=False)
        .to_tuple("__key__", "signal.npy", "features.npy", "label", "video_path")
    )
    key, signal_bytes, feat_bytes, label, vpath = next(iter(ds))
    signal = np.load(io.BytesIO(signal_bytes))
    features = np.load(io.BytesIO(feat_bytes))

    print("Raw sample")
    print("key:", key)
    print("label:", label)
    print("video_path:", vpath)
    print("signal shape:", signal.shape, "min/max:", signal.min(), signal.max())
    print("features shape:", features.shape)


def peek_dataloader(cfg_data, cfg_extractor, cfg_masker, batch_size, num_workers, pin_memory):
    conv_layers_spec = eval(cfg_extractor.conv_layers_spec)
    extractor = ConvFeatureExtractor(
        conv_layers_spec=conv_layers_spec,
        in_channels=cfg_data.in_channels,
        depthwise=cfg_extractor.depthwise,
    )
    input_len = int(cfg_data.sr * cfg_data.process_seconds)
    nr_patches = extractor.total_patches(input_len)

    stride_samples, rf_samples = _effective_stride_and_rf(conv_layers_spec)
    stride_seconds = stride_samples / float(cfg_data.sr)
    rf_seconds = rf_samples / float(cfg_data.sr)

    print("Tokenization")
    print(
        f"input_len_samples: {input_len} (sr={cfg_data.sr} Hz, window={cfg_data.process_seconds}s)"
    )
    print(f"tokens (S): {nr_patches}")
    print(f"effective_stride: {stride_samples} samples ({stride_seconds:.4g}s)")
    print(f"receptive_field: {rf_samples} samples ({rf_seconds:.4g}s)")

    masker_name = getattr(cfg_masker, "name", None)
    masker_class = MASKERS.get(masker_name)
    if masker_class is None:
        raise ValueError(
            f"Unsupported masker name: {masker_name}. Available: {list(MASKERS.keys())}"
        )
    masker_kwargs = OmegaConf.to_container(cfg_masker, resolve=True)
    masker = masker_class(**masker_kwargs)

    print("Masker")
    print(f"name: {masker_name}")
    if masker_name == "time-inverse-masker":
        ctx_len = int(getattr(cfg_masker, "context_mask_length", 0))
        tgt_len = int(getattr(cfg_masker, "target_length", 0))
        print(
            f"context_mask_length: {ctx_len} tokens "
            f"(~{_span_seconds(ctx_len, stride_seconds, rf_seconds):.4g}s/span)"
        )
        print(
            f"target_length: {tgt_len} tokens "
            f"(~{_span_seconds(tgt_len, stride_seconds, rf_seconds):.4g}s/span)"
        )
    elif masker_name in ("block-masker", "multi-block-masker"):
        ctx_d = float(getattr(cfg_masker, "context_cluster_d", 0.0))
        ctx_u = float(getattr(cfg_masker, "context_cluster_u", 0.0))
        tgt_d = float(getattr(cfg_masker, "target_cluster_d", 0.0))
        tgt_u = float(getattr(cfg_masker, "target_cluster_u", 0.0))
        ctx_tokens = (int(nr_patches * ctx_d), int(nr_patches * ctx_u))
        tgt_tokens = (int(nr_patches * tgt_d), int(nr_patches * tgt_u))
        print(
            f"context_cluster: {ctx_d:g}..{ctx_u:g} -> {ctx_tokens[0]}..{ctx_tokens[1]} tokens "
            f"(~{_span_seconds(ctx_tokens[0], stride_seconds, rf_seconds):.4g}.."
            f"{_span_seconds(ctx_tokens[1], stride_seconds, rf_seconds):.4g}s)"
        )
        print(
            f"target_cluster: {tgt_d:g}..{tgt_u:g} -> {tgt_tokens[0]}..{tgt_tokens[1]} tokens "
            f"(~{_span_seconds(tgt_tokens[0], stride_seconds, rf_seconds):.4g}.."
            f"{_span_seconds(tgt_tokens[1], stride_seconds, rf_seconds):.4g}s)"
        )
    else:
        print(
            "note: this masker samples (mostly) non-contiguous token sets; "
            "token-count->seconds is not a single span."
        )

    dm = WebSignalDataModule(
        base_data_dir=cfg_data.base_data_dir,
        val_data_dir=cfg_data.val_data_dir,
        batch_size=batch_size,
        masker=masker,
        nr_samples_per_audio=cfg_data.samples_per_audio,
        nr_time_points=nr_patches,
        in_channels=cfg_data.in_channels,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    (
        audio,
        noise,
        source_rir,
        noise_rirs,
        snr,
        ctx_masks,
        target_indices,
        ctx_and_target_masks,
    ) = batch

    names = [
        "audio",
        "noise",
        "source_rir",
        "noise_rirs",
        "snr",
        "ctx_masks",
        "target_indices",
        "ctx_and_target_masks",
    ]

    print("Dataloader batch")
    for name, value in zip(names, batch):
        _print_tensor(name, value)
    print("")
    if getattr(peek_dataloader, "_visualize", True):
        _visualize_masks(ctx_masks, target_indices, max_len=120)


def main():
    # ----------- Edit these values as needed -----------
    
    data_cfg = "configs/data/custom_signal.yaml"
    extractor_cfg = "configs/extractor/ConvFeatureExtractor_signal.yaml"
    masker_cfg = "configs/masker/MultiBlock.yaml"
    batch_size = 2
    num_workers = 0
    pin_memory = False
    raw_only = False
    loader_only = False
    visualize_masks = True
    # ----------------------------------------------------

    cfg_data = OmegaConf.load(data_cfg)
    cfg_extractor = OmegaConf.load(extractor_cfg)
    cfg_masker = OmegaConf.load(masker_cfg)

    if not loader_only:
        peek_raw(cfg_data.base_data_dir)
        print("")

    if not raw_only:
        peek_dataloader._visualize = visualize_masks
        peek_dataloader(
            cfg_data,
            cfg_extractor,
            cfg_masker,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


if __name__ == "__main__":
    main()
