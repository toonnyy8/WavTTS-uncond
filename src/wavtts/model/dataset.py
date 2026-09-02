import json
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm



class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=16_000,
        wav_frame_len: int = 160,
        wav_frame_hop: int | None = None,
        target_rms: float = 0.1,  # per-utterance loudness normalization; 0 disables
        random_frame_offset: bool = False,  # framing-phase augmentation, see __getitem__
        **_,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.wav_frame_len = wav_frame_len
        # batches are budgeted in model tokens, and overlapping frames produce one token
        # per hop, not per frame. Keeping the frame length here would undercount by
        # wav_frame_len / wav_frame_hop and blow up memory by the same factor.
        self.wav_frame_hop = wav_frame_len if wav_frame_hop is None else wav_frame_hop
        self.target_rms = target_rms
        self.random_frame_offset = random_frame_offset

        self._resamplers = {}

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.wav_frame_hop
        return self.data[index]["duration"] * self.target_sample_rate / self.wav_frame_hop

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            duration = row["duration"]

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        audio, source_sample_rate = torchaudio.load(audio_path)

        # make sure mono input
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # resample if necessary
        if source_sample_rate != self.target_sample_rate:
            if source_sample_rate not in self._resamplers:
                self._resamplers[source_sample_rate] = torchaudio.transforms.Resample(
                    source_sample_rate, self.target_sample_rate
                )
            audio = self._resamplers[source_sample_rate](audio)

        # framing-phase augmentation: drop a random 0..hop-1 samples off the front, so the
        # clip meets the frame grid at a different phase every time it is drawn.
        #
        # The front end slices on a grid anchored at sample 0 (DiT._wav_to_tokens) and
        # input_embed projects the raw samples through a Linear whose weights are tied to
        # their position in the frame, so the model has no shift equivariance at all: delay
        # a waveform by one sample and its whole token decomposition changes. Without this,
        # every clip is seen under one arbitrary alignment for the entire run -- the same
        # 149510 waveform/phase pairs, epoch after epoch -- and the grid becomes something
        # to memorize. The offset turns that into hop distinct views per clip, and it is a
        # pure delay, so the augmentation is exactly perceptually neutral.
        #
        # The range is the hop, not the frame length: framing is periodic in the hop, and an
        # offset of a whole hop is the same set of frames shifted by one token.
        #
        # Cropping rather than left-padding with zeros, for two reasons. Zeros would put a
        # random-length silence and then an abrupt onset *inside* the first frame -- a
        # manufactured discontinuity of just the kind this is meant to keep the model from
        # learning -- and they would grow the token count by one, which get_frame_len (a
        # duration estimate, read before the audio is) would not see. Cropping can only
        # shorten, so that estimate stays an upper bound and the frame budget stays safe.
        # The cost is at most hop-1 samples, 9.9 ms at a hop of 160.
        if self.random_frame_offset and self.wav_frame_hop > 1:
            offset = int(torch.randint(0, self.wav_frame_hop, (1,)))
            offset = min(offset, max(audio.shape[-1] - 1, 0))
            if offset > 0:
                audio = audio[..., offset:]

        # loudness: every utterance enters training at the same RMS, so the equal-power
        # mixing augmentation blends two comparable sources instead of one drowning the
        # other. No peak guard: speech runs a crest factor around 7, so clamping to
        # +-0.99 at target_rms 1.0 would fire on every clip and hand back exactly the
        # per-clip loudness this is here to remove. At the old target_rms of 0.1 it still
        # fired on a fifth of the corpus, pulling those clips as low as 0.052. The
        # waveform leaves the +-1 domain deliberately; peak_normalize() is what anything
        # writing it to a file or feeding a pretrained model calls first.
        # DC is removed first, so the RMS below is the signal's standard deviation rather
        # than sqrt(mean^2 + var). A recording carrying converter bias would otherwise
        # have that bias counted as loudness and get scaled down for it, and the offset
        # would survive into training as a constant the model has to learn to emit.
        if self.target_rms > 0:
            audio = audio - audio.mean()
            rms = audio.pow(2).mean().sqrt()
            if rms > 1e-5:
                audio = audio * (self.target_rms / rms)

        return {
            "wav": audio.squeeze(0),
        }


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    waveform_kwargs: dict = dict(),
) -> CustomDataset:
    """
    WavTTS only supports raw waveform datasets.
    dataset_type:
      - "CustomDataset": use default data path data/{dataset_name}
        (include any legacy tokenizer suffix, e.g. Emilia_ZH_EN_pinyin, in the name)
      - "CustomDatasetPath": pass the full path to a prepared dataset
    """

    print("Loading dataset ...")

    if audio_type != "raw":
        raise ValueError("WavTTS only supports raw waveform datasets; audio_type must be 'raw'.")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("wavtts").joinpath(f"../../data/{dataset_name}"))
        try:
            train_dataset = load_from_disk(f"{rel_data_path}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(train_dataset, durations=durations, **waveform_kwargs)

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(train_dataset, durations=durations, **waveform_kwargs)

    else:
        raise ValueError(f"Unsupported dataset_type for WavTTS wav-only training: {dataset_type}")

    return train_dataset


# collation


def collate_fn(batch):
    wavs = [item["wav"] for item in batch]
    wav_lengths = torch.LongTensor([w.shape[0] for w in wavs])
    max_wav_len = wav_lengths.max().item()

    padded_wavs = []
    for w in wavs:
        pad_len = max_wav_len - w.shape[0]
        padded_wavs.append(F.pad(w, (0, pad_len), value=0.0))

    return dict(
        wav=torch.stack(padded_wavs),  # [B, T_wav]
        wav_lengths=wav_lengths,
    )
