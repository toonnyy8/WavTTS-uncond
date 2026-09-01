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
        target_rms: float = 0.1,  # per-utterance loudness normalization; 0 disables
        rand_frame_offset: bool = False,  # sub-frame grid jitter; see __getitem__
        **_,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.wav_frame_len = wav_frame_len
        self.target_rms = target_rms
        self.rand_frame_offset = rand_frame_offset

        self._resamplers = {}

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.wav_frame_len
        return self.data[index]["duration"] * self.target_sample_rate / self.wav_frame_len

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

        # Sub-frame offset. Tokenisation is a stride-wav_frame_len reshape
        # (DiT._wav_to_tokens): no window, no overlap, so shifting a clip by less than one
        # frame yields a token sequence the model has no architectural way to relate back to
        # the unshifted one. Without this a clip hands back byte-identical 160-sample vectors
        # on all 177 epochs, which is exactly what a 664M-parameter model can memorise; with
        # it the realisation differs every draw, at the cost of under 10 ms of leading silence.
        # The corpus already covers every grid alignment across its 149k clips, so this buys
        # decorrelation rather than a symmetry the data hides.
        #
        # Trimming, not zero-padding: padding would make "clips open on digital silence" a
        # systematic feature of the data, and would let a clip spill into one more frame than
        # get_frame_len() promised DynamicBatchSampler, whose budget rounds no frames up.
        # Batch composition stays deterministic under the run seed either way -- the sampler
        # reads durations, which this does not touch.
        if self.rand_frame_offset and audio.shape[-1] > self.wav_frame_len:
            offset = int(torch.randint(0, self.wav_frame_len, (1,)).item())
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
