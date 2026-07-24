from torch.utils.data import DataLoader
import pytorch_lightning as pl
import webdataset as wds
from .dataset_functions import pre_process_audio_visage
import torch 

def to_torch(sample):
    return torch.from_numpy(sample[0])

class ViSageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_data_dir: str,
        batch_size: int = 32,
        sr: int = 32000,
        masker=None,
        nr_samples_per_audio: int = 16,
        nr_patches: int = 200,
        **kwargs
    ):
        """Initialize the data module with shared noise data."""
        super().__init__()
        self.datapath = base_data_dir
        self.batch_size = batch_size
        self.sr = sr
        self.masker = masker
        self.nr_samples_per_audio = nr_samples_per_audio
        self.nr_patches = nr_patches
    
    def _augment_sample(self, sample):
        """Augment sample with noise and RIR data."""
        
        audio, audio_sr = sample[0]
        audio = pre_process_audio_visage(audio, audio_sr, self.sr)

        context_idx = self.masker(
            local_features=None,
            batch_size=self.nr_samples_per_audio,
            n_times=self.nr_patches
        )
        return audio, context_idx

    def make_web_dataset(self, path: str, shuffle: int):
        """Create a WebDataset pipeline for audio processing."""
        dataset = (
            wds.WebDataset(
                path,
                resampled=True,
                nodesplitter=wds.shardlists.split_by_node,
                workersplitter=wds.shardlists.split_by_worker,
                shardshuffle=False
            )
            .repeat()
            .shuffle(shuffle)
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .to_tuple("flac")
            .map(self._augment_sample)
            .batched(self.batch_size)
        )
        return dataset

    def setup(self, stage: str):
        """Set up datasets for training."""
        if stage == "fit":
            self.audio_train = self.make_web_dataset(
                self.datapath, shuffle=1000
            )

    def train_dataloader(self):
        """Return the training DataLoader."""
        loader = DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=True,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )
        return loader