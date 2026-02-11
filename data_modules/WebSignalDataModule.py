import io

import numpy as np
import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader


class WebSignalDataModule(pl.LightningDataModule):
    """
    DataModule for non-audio 1D signals stored as WebDataset shards.

    Expects each sample to contain a `signal.npy` entry (e.g., from
    preprocess_to_shards_9p8s.py). Returns the same tuple structure as
    WebAudioDataModuleLMDB so the training loop is unchanged.
    """

    def __init__(
        self,
        base_data_dir: str,
        val_data_dir: str,
        batch_size: int = 32,
        masker=None,
        nr_samples_per_audio: int = 1,
        nr_time_points: int = 100,
        in_channels: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.datapath = base_data_dir
        self.val_path = val_data_dir
        self.batch_size = batch_size
        self.masker = masker
        self.nr_samples_per_audio = nr_samples_per_audio
        self.nr_time_points = nr_time_points
        self.in_channels = in_channels
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

    @staticmethod
    def _to_tensor(signal):
        if isinstance(signal, bytes):
            signal = np.load(io.BytesIO(signal))
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)
        else:
            signal = torch.as_tensor(signal)
        return signal.float().squeeze()

    def _extract_signal(self, sample):
        if isinstance(sample, dict):
            signal = (
                sample.get("signal.npy")
                or sample.get("signal")
                or sample.get("npy")
            )
        else:
            signal = sample[0]
        if signal is None:
            raise KeyError(
                "Could not find signal in sample. Expected key 'signal.npy' (or 'signal'/'npy')."
            )
        return self._to_tensor(signal)

    def _augment_sample(self, sample):
        signal = self._extract_signal(sample)

        context_mask, target_indices, ctx_and_target_masks = self.masker(
            batch_size=self.nr_samples_per_audio,
            n_times=self.nr_time_points,
            in_channels=self.in_channels,
        )

        return (
            signal,
            None,
            None,
            None,
            None,
            context_mask,
            target_indices,
            ctx_and_target_masks,
        )

    def make_web_dataset(self, path: str, shuffle: int):
        dataset = (
            wds.WebDataset(
                path,
                resampled=True,
                nodesplitter=wds.shardlists.split_by_node,
                workersplitter=wds.shardlists.split_by_worker,
                shardshuffle=False,
            )
            .repeat()
            .shuffle(shuffle)
            .map(self._augment_sample)
            .batched(self.batch_size)
        )
        return dataset

    def setup(self, stage: str):
        if stage == "fit":
            self.signal_train = self.make_web_dataset(self.datapath, shuffle=1000)

    def train_dataloader(self):
        loader_kwargs = {
            "batch_size": None,
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        loader = DataLoader(self.signal_train, **loader_kwargs)
        return loader
