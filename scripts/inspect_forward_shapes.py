#!/usr/bin/env python3
import sys
from ast import literal_eval
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from data_modules import WebSignalDataModule
    from sjepa.extractors import ConvFeatureExtractor
    from sjepa.jepa import JEPA
    from sjepa.masking import TimeInverseBlockMasker
    from sjepa.types import TransformerEncoderCFG, TransformerLayerCFG

    cfg_data = OmegaConf.load(str(ROOT / "configs/data/custom_signal.yaml"))
    cfg_ext = OmegaConf.load(
        str(ROOT / "configs/extractor/ConvFeatureExtractor_signal.yaml")
    )
    cfg_mask = OmegaConf.load(str(ROOT / "configs/masker/TimeInverseBlock.yaml"))
    cfg_tr = OmegaConf.load(str(ROOT / "configs/trainer/default_trainer.yaml"))

    extractor = ConvFeatureExtractor(
        conv_layers_spec=literal_eval(cfg_ext.conv_layers_spec),
        in_channels=cfg_data.in_channels,
        depthwise=cfg_ext.depthwise,
    )

    masker = TimeInverseBlockMasker(**cfg_mask)

    dm = WebSignalDataModule(
        base_data_dir=cfg_data.base_data_dir,
        val_data_dir=cfg_data.val_data_dir,
        batch_size=2,
        masker=masker,
        nr_samples_per_audio=cfg_data.samples_per_audio,
        nr_time_points=extractor.total_patches(
            int(cfg_data.sr * cfg_data.process_seconds)
        ),
        in_channels=cfg_data.in_channels,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    enc_layer = TransformerLayerCFG.create()
    enc_cfg = TransformerEncoderCFG.create()
    dec_layer = TransformerLayerCFG.create(d_model=384)
    dec_cfg = TransformerEncoderCFG.create()

    model = JEPA(
        feature_extractor=extractor,
        transformer_encoder_layers_cfg=enc_layer,
        transformer_encoder_cfg=enc_cfg,
        transformer_decoder_layers_cfg=dec_layer,
        transformer_decoder_cfg=dec_cfg,
        in_channels=cfg_data.in_channels,
        resample_sr=cfg_data.sr,
        original_sr=cfg_data.original_sr,
        process_audio_seconds=cfg_data.process_seconds,
        nr_samples_per_audio=cfg_data.samples_per_audio,
        use_gradient_checkpointing=cfg_tr.get("use_gradient_checkpointing", False),
        compile_modules=False,
        average_top_k_layers=cfg_tr.get("average_top_k_layers", 12),
        is_spectrogram=False,
        clean_data_ratio=cfg_data.get("clean_data_ratio", 0.0),
        size=cfg_tr.get("size", "base"),
    )

    audio_input, ctx_masks, target_indices, ctx_and_target_masks = model.prepare_batch(
        batch
    )
    audio_input = audio_input.float()
    print("audio_input", audio_input.shape)
    print("ctx_masks", ctx_masks.shape)
    print("target_indices", target_indices.shape)
    print("ctx_and_target_masks", ctx_and_target_masks.shape)

    out = model(audio_input, ctx_masks, target_indices, ctx_and_target_masks)
    print("local_features", out["local_features"].shape)
    print("contextual_features", out["contextual_features"].shape)
    print("preds", out["preds"].shape)
    print("targets", out["targets"].shape)


if __name__ == "__main__":
    main()
