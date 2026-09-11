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
        is_fake: bool = False,  # DDO: this whole dir is p_ref's output, not recorded speech
        **_,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.wav_frame_len = wav_frame_len
        self.target_rms = target_rms
        self.rand_frame_offset = rand_frame_offset
        # A tag, not a switch: nothing below this line branches on it. The fake pool is
        # written in the real corpus' own directory format precisely so it can come through
        # this same __getitem__ with the same waveform_kwargs -- the RMS normalization and
        # the sub-frame jitter below are what erase "loudness" and "sits exactly on the frame
        # grid" as features a discriminator could separate real from fake on, without ever
        # touching speech quality (spec 3.5, shortcuts 2 and 3).
        self.is_fake = is_fake

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
            "is_fake": self.is_fake,
        }


class TaggedConcatDataset(Dataset):
    """Real corpus followed by `fake_repeat` copies of the fake pool, for DDO.

    The paper's diffusion setting pairs real and fake 1:1 (50k generated images against
    50k CIFAR images). A fake pool that size is unaffordable here -- 244.6 h of 32-NFE
    sampling -- so the ratio is enforced by *sampling* rather than by pool size: the fake
    indices are repeated until the two halves contribute comparable frame counts
    (spec 3.5). Only the waveform repeats. Every draw still gets a fresh (t, eps) in
    CFM.forward and a fresh sub-frame offset in CustomDataset.__getitem__, so a repeated
    clip is never a repeated training example -- the price is that the fake half is seen
    ~5x as often as the real half within one round, which is the pool-overfitting risk
    the spec's risk table tracks via ddo/delta_fake against gen/utmos.

    Concatenation rather than interleaving: DynamicBatchSampler sorts the whole index
    space by frame length and packs neighbours, so real and fake of similar length land
    in the same batch on their own. It only needs get_frame_len(idx) and
    sampler.data_source, which is why this is a plain Dataset and frame-wise batching
    needs no change at all.
    """

    def __init__(self, real: CustomDataset, fake: CustomDataset, fake_repeat: int = 1):
        if fake_repeat < 1:
            raise ValueError(f"fake_repeat must be >= 1, got {fake_repeat}")
        if real.target_sample_rate != fake.target_sample_rate:
            raise ValueError(
                f"real and fake datasets disagree on sample rate "
                f"({real.target_sample_rate} vs {fake.target_sample_rate}); the fake pool must be "
                "generated at the model's rate so both halves go through one loading path"
            )
        self.real = real
        self.fake = fake
        self.fake_repeat = int(fake_repeat)

        # DynamicBatchSampler and Trainer.train() read these off the dataset object, so the
        # concat has to forward them rather than inherit from nothing.
        self.target_sample_rate = real.target_sample_rate
        self.wav_frame_len = real.wav_frame_len

    def __len__(self):
        return len(self.real) + len(self.fake) * self.fake_repeat

    def _route(self, index: int) -> tuple[CustomDataset, int]:
        n_real = len(self.real)
        if index < n_real:
            return self.real, index
        # modulo, not block-repeat: which copy an index falls in is irrelevant, and wrapping
        # keeps the mapping stable if fake_repeat changes between rounds.
        return self.fake, (index - n_real) % len(self.fake)

    def __getitem__(self, index):
        ds, idx = self._route(index)
        return ds[idx]

    def get_frame_len(self, index):
        ds, idx = self._route(index)
        frame_len = ds.get_frame_len(idx)
        if ds is self.fake:
            frame_len += self._grid_dither(index)
        return frame_len

    def _grid_dither(self, index: int) -> float:
        """A deterministic sub-frame offset, in [0, 1) frames, added to fake lengths only.

        Without it, DDO trains on one-sided batches. DynamicBatchSampler sorts the whole
        index space by get_frame_len and packs neighbours, and Python's sort is stable, so
        equal keys keep index order -- every real row first, then every fake row. Real
        lengths are `samples / wav_frame_len` for arbitrary recorded sample counts, so they
        are dense floats; generated lengths are whole frames by construction, so they all
        pile onto the same handful of integer keys. Each such pile is contiguous and larger
        than one frame budget, and comes out as a run of all-fake batches. Simulated on the
        real corpus geometry (149.5k clips, a 20% pool repeated 5x, 4800 frames/GPU) that is
        **93% one-sided batches**: the global real:fake ratio is exactly right and the
        per-batch ratio, which is the one the loss sees, is 0 or 1 almost every step.

        That breaks two things the spec asks for by name: the per-batch real:fake balance
        (3.5) and the shared time draws between paired real and fake rows (3.5's common
        random numbers), which cannot pair anything in a batch with only one kind of row.

        Spreading the fake keys back across the sub-frame positions the real half already
        occupies interleaves them: the same simulation drops to 7.9% one-sided, the
        remainder being the very long clips where a batch holds one or two rows anyway.

        This only ever over-reports a length, and by less than a single frame out of a
        budget of thousands, so the sampler stays conservative and no batch grows. It is
        deterministic in the index, because DynamicBatchSampler builds its batches once.
        Nothing outside the DDO path calls this.
        """
        return ((index * 2654435761) % self.wav_frame_len) / self.wav_frame_len


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


def _total_frames(dataset: CustomDataset) -> float:
    return float(sum(dataset.get_frame_len(i) for i in range(len(dataset))))


def load_ddo_dataset(
    real_name: str,
    fake_name: str,
    waveform_kwargs: dict,
    real_fake_ratio: float = 1.0,
    dataset_type: str = "CustomDataset",
) -> TaggedConcatDataset:
    """Real corpus + offline fake pool, as one dataset for DDO finetuning.

    Both halves go through load_dataset() with the *same* waveform_kwargs. That is the
    single most important line in this file for DDO: the discriminator here is the model
    itself, and it will happily drive Delta apart on any feature that separates the two
    pools, whether or not that feature has anything to do with speech quality. Loudness,
    frame-grid phase and file precision are all neutralised by sharing this one loading
    path (spec 3.5); length distribution is neutralised upstream, by gen_fake_pool.py
    drawing its durations from the real duration.json.

    real_fake_ratio is the target real:fake *frame* ratio per batch (1.0 = the paper's
    1:1 pairing). Since only whole repeats of the pool are available, the achieved ratio
    is printed rather than promised.
    """
    real = load_dataset(real_name, dataset_type=dataset_type, waveform_kwargs=waveform_kwargs)
    fake = load_dataset(
        fake_name,
        dataset_type=dataset_type,
        waveform_kwargs={**dict(waveform_kwargs), "is_fake": True},
    )

    real_frames = _total_frames(real)
    fake_frames = _total_frames(fake)
    if fake_frames <= 0:
        raise ValueError(f"fake dataset '{fake_name}' has no frames")
    if real_fake_ratio <= 0:
        raise ValueError(f"real_fake_ratio must be > 0, got {real_fake_ratio}")

    fake_repeat = max(1, round(real_frames / (fake_frames * real_fake_ratio)))
    achieved = real_frames / (fake_frames * fake_repeat)
    hours = real.wav_frame_len / real.target_sample_rate / 3600
    print(
        f"DDO dataset: real '{real_name}' {len(real)} clips / {real_frames * hours:.2f} h, "
        f"fake '{fake_name}' {len(fake)} clips / {fake_frames * hours:.2f} h x{fake_repeat} repeats; "
        f"real:fake frame ratio {achieved:.3f} (requested {real_fake_ratio:.3f})"
    )
    return TaggedConcatDataset(real, fake, fake_repeat=fake_repeat)


# collation


def collate_fn(batch):
    wavs = [item["wav"] for item in batch]
    wav_lengths = torch.LongTensor([w.shape[0] for w in wavs])
    max_wav_len = wav_lengths.max().item()

    padded_wavs = []
    for w in wavs:
        pad_len = max_wav_len - w.shape[0]
        padded_wavs.append(F.pad(w, (0, pad_len), value=0.0))

    out = dict(
        wav=torch.stack(padded_wavs),  # [B, T_wav]
        wav_lengths=wav_lengths,
    )

    # item.get, not item["is_fake"]: callers hand this raw dicts (the smoke tests do), and
    # anything that predates DDO has no such key. The key is emitted only when the batch
    # actually carries the tag, so a non-DDO run's batch dict keeps exactly the shape it
    # had -- the trainer reads it back with batch.get("is_fake"), and CFM.forward treats a
    # missing flag as all-real anyway.
    if any("is_fake" in item for item in batch):
        out["is_fake"] = torch.tensor([bool(item.get("is_fake", False)) for item in batch])

    return out
