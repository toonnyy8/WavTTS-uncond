import math

import pytest
import torch
from torch import nn


def test_import():
    from wavtts.model import CFM, DiT  # noqa: F401

    assert torch.tensor(1.0).item() == 1.0


def _reinit_nonzero(model):
    # zero-init 的 AdaLN/proj_out 會讓輸出恆為 0，正負分支無差異；測試前擾動權重
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)


def test_dit_forward_shape():
    from wavtts.model.backbones.dit import STATE_CLEAN, DiT

    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    x = torch.randn(2, 1600)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)
    out = dit(x=x, state=state, time=torch.tensor(0.5))
    assert out.shape == (2, 1600)
    assert torch.isfinite(out).all()


def test_dit_cfg_infer_packs_clean_and_null():
    from wavtts.model.backbones.dit import STATE_CLEAN, DiT

    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    _reinit_nonzero(dit)
    x = torch.randn(2, 1600)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)
    out = dit(x=x, state=state, time=torch.tensor(0.5), cfg_infer=True)
    assert out.shape == (4, 1600)
    pos, neg = torch.chunk(out, 2, dim=0)
    assert not torch.allclose(pos, neg)  # 正負分支必須產生不同輸出


def test_dit_state_changes_output():
    from wavtts.model.backbones.dit import NUM_STATES, STATE_CLEAN, STATE_NULL, DiT

    assert (STATE_CLEAN, STATE_NULL, NUM_STATES) == (0, 1, 2)  # mixed no longer has a state
    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    _reinit_nonzero(dit)
    x = torch.randn(1, 1600)
    t = torch.tensor(0.5)
    out_clean = dit(x=x, state=torch.tensor([STATE_CLEAN]), time=t)
    out_null = dit(x=x, state=torch.tensor([STATE_NULL]), time=t)
    assert not torch.allclose(out_clean, out_null)


def make_model(use_aux_mel_loss=False, **kwargs):
    from wavtts.model import CFM, DiT

    torch.manual_seed(0)
    transformer = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    defaults = dict(
        waveform_kwargs={"wav_frame_len": 160},
        prediction="x_pred",
        loss_space="v",
        t_eps=0.02,
        use_aux_mel_loss=use_aux_mel_loss,
        aux_mel_loss_weight=0.05,
        sample_rate=16000,
        latents_scale=1.0,
    )
    defaults.update(kwargs)
    return CFM(transformer=transformer, **defaults)


def test_mix_augment_applies_only_to_flagged():
    model = make_model()
    torch.manual_seed(0)
    x = torch.randn(4, 3200)
    lens = torch.full((4,), 3200, dtype=torch.long)

    flags = torch.tensor([True, False, True, False])
    x_aug = model._mix_augment(x, lens, flags)
    assert not torch.allclose(x_aug[0], x[0])
    assert torch.equal(x_aug[1], x[1])  # unflagged samples pass through untouched
    assert torch.equal(x_aug[3], x[3])
    assert torch.equal(model._mix_augment(x, lens, torch.zeros(4, dtype=torch.bool)), x)


def _observed_states(model, batch=4096):
    from wavtts.model.backbones.dit import STATE_CLEAN

    seen = {}
    orig_forward = model.transformer.forward

    def spy(*, x, state, **kwargs):
        seen["state"] = state.clone()
        seen["x"] = x.clone()
        # the real backbone on a 4096-sample batch would be absurd; the labels are
        # decided before it is called, so a zero of the right shape is enough
        return torch.zeros_like(x)

    model.transformer.forward = spy
    model(torch.randn(batch, 1600) * 0.1)
    model.transformer.forward = orig_forward
    return seen["state"], STATE_CLEAN


def test_label_split_is_half_clean_half_null():
    from wavtts.model.backbones.dit import STATE_CLEAN, STATE_NULL

    model = make_model(state_null_prob=0.5, p_mix=0.5)
    torch.manual_seed(0)
    state, _ = _observed_states(model)
    null_frac = (state == STATE_NULL).float().mean().item()
    assert null_frac == pytest.approx(0.5, abs=0.03)
    assert (state == STATE_CLEAN).float().mean().item() == pytest.approx(0.5, abs=0.03)


def test_only_null_samples_are_mixed():
    from wavtts.model.backbones.dit import STATE_NULL

    # the clean branch must stay pure single-speaker speech: a mixed sample labelled
    # clean would have CFG guiding *toward* speaker inconsistency
    model = make_model(state_null_prob=0.5, p_mix=1.0)
    seen = {}
    orig_mix = model._mix_augment

    def spy_mix(x1, lens, mix_flags):
        seen["mix"] = mix_flags.clone()
        return orig_mix(x1, lens, mix_flags)

    model._mix_augment = spy_mix
    model.transformer.forward = lambda *, x, state, **kw: (
        seen.__setitem__("state", state.clone()),
        torch.zeros_like(x),
    )[1]

    torch.manual_seed(0)
    model(torch.randn(64, 1600) * 0.1)

    is_null = seen["state"] == STATE_NULL
    assert is_null.any() and not is_null.all()  # the split actually happened
    assert not (seen["mix"] & ~is_null).any()  # no clean sample was mixed
    assert torch.equal(seen["mix"], is_null)  # p_mix=1.0 -> every null sample was


def test_mix_augment_concat_prefix_preserved():
    model = make_model(p_concat=1.0)
    torch.manual_seed(0)
    x = torch.randn(2, 3200)
    lens = torch.full((2,), 3200, dtype=torch.long)
    x_aug = model._mix_augment(x, lens, torch.ones(2, dtype=torch.bool))
    # 切換點最早在 0.3*3200=960，之前的內容必須原封不動（no leaky：前段就是原語者）
    assert torch.equal(x_aug[:, :900], x[:, :900])


def test_mix_augment_batch_of_one_is_noop():
    model = make_model()
    x = torch.randn(1, 3200)
    lens = torch.full((1,), 3200, dtype=torch.long)
    x_aug = model._mix_augment(x, lens, torch.ones(1, dtype=torch.bool))
    assert torch.equal(x_aug, x)  # roll partner would be the sample itself


def test_train_step_backward():
    model = make_model()
    torch.manual_seed(0)
    wav = torch.randn(4, 16000) * 0.1
    lens = torch.tensor([16000, 12000, 16000, 8000])
    loss, loss_dict = model(wav, lens=lens)
    assert torch.isfinite(loss)
    assert set(loss_dict) == {"total_loss", "flow_loss", "aux_mel_loss", "rep_loss"}
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_train_step_with_aux_mel_loss():
    model = make_model(use_aux_mel_loss=True)
    torch.manual_seed(0)
    wav = torch.randn(2, 16000) * 0.1
    loss, loss_dict = model(wav)
    assert torch.isfinite(loss)
    assert loss_dict["aux_mel_loss"].item() != 0.0
    loss.backward()


@pytest.mark.parametrize("cfg_strength", [2.0, 0.0])
def test_sample_shapes(cfg_strength):
    model = make_model()
    out, trajectory = model.sample(8000, batch=2, steps=2, cfg_strength=cfg_strength, seed=0)
    assert out.shape == (2, 8000)
    assert torch.isfinite(out).all()
    assert trajectory.shape[0] == 3  # steps+1 個時間點


def test_sample_duration_not_multiple_of_frame_len():
    model = make_model()
    out, _ = model.sample(8123, batch=1, steps=2, seed=0)
    assert out.shape == (1, 8123)


def test_collate_wav_only():
    from wavtts.model.dataset import collate_fn

    batch = [{"wav": torch.randn(1000)}, {"wav": torch.randn(800)}]
    out = collate_fn(batch)
    assert set(out.keys()) == {"wav", "wav_lengths"}
    assert out["wav"].shape == (2, 1000)
    assert out["wav_lengths"].tolist() == [1000, 800]
    assert torch.equal(out["wav"][1, 800:], torch.zeros(200))


def test_dataset_normalizes_loudness(tmp_path):
    import torchaudio

    from wavtts.model.dataset import CustomDataset

    t = torch.linspace(0, 1, 16000)
    signals = {
        "loud": 0.8 * torch.sin(2 * torch.pi * 220 * t),
        "quiet": 0.01 * torch.sin(2 * torch.pi * 220 * t),
        "peaky": torch.zeros(16000).index_fill_(0, torch.arange(0, 16000, 400), 1.0),  # high crest factor
        "biased": 0.3 * torch.sin(2 * torch.pi * 220 * t) + 0.2,  # converter DC offset
    }
    rows = []
    for name, wav in signals.items():
        path = tmp_path / f"{name}.wav"
        torchaudio.save(str(path), wav.unsqueeze(0), 16000)
        rows.append({"audio_path": str(path), "duration": 1.0})

    ds = CustomDataset(rows, durations=[1.0] * len(rows), target_rms=1.0)
    for i, name in enumerate(signals):
        wav = ds[i]["wav"]
        # every clip lands exactly on target, the peaky one included: there is no peak
        # clamp to hand a fraction of the corpus its loudness back
        assert wav.pow(2).mean().sqrt().item() == pytest.approx(1.0, rel=1e-5), name
    # and the waveform is expected to leave +-1 -- the impulse train's crest factor is 20
    assert ds[2]["wav"].abs().max().item() > 1.0

    # DC is removed before the RMS is taken, so the offset is not counted as loudness:
    # the biased clip has the same AC level as an unbiased one, and lands on target
    biased = ds[3]["wav"]
    assert biased.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert biased.pow(2).mean().sqrt().item() == pytest.approx(1.0, rel=1e-5)
    # and the stronger form: the bias becomes invisible. "loud" and "biased" are the same
    # 220 Hz tone at different amplitudes, one of them offset. After DC removal and RMS
    # normalization they have to come out as the same waveform -- which they cannot if
    # the offset is counted as loudness, since it would scale "biased" down by 1.37x.
    assert torch.allclose(ds[0]["wav"], biased, atol=2e-3)

    off = CustomDataset(rows, durations=[1.0] * len(rows), target_rms=0.0)
    assert off[0]["wav"].pow(2).mean().sqrt().item() == pytest.approx(0.8 / 2**0.5, abs=0.01)


def test_config_instantiates_model_and_trains_one_step():
    from importlib.resources import files as pkg_files

    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from wavtts.model import CFM

    cfg = OmegaConf.load(str(pkg_files("wavtts").joinpath("configs/WavTTS.yaml")))
    arch = OmegaConf.to_container(cfg.model.arch, resolve=True)
    arch.update(dim=64, depth=2, heads=2)  # 縮小以便 CPU 測試；參數名必須與 DiT 簽名一致
    cfm_kwargs = OmegaConf.to_container(cfg.model.cfm, resolve=True)
    model_cls = get_class(f"wavtts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(**arch, wav_frame_len=cfg.model.waveform.wav_frame_len),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **cfm_kwargs,
    )
    torch.manual_seed(0)
    loss, loss_dict = model(torch.randn(2, 16000) * 0.1)
    assert torch.isfinite(loss)
    loss.backward()


def test_sample_uncond_cli(tmp_path):
    from importlib.resources import files as pkg_files

    import torchaudio
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from wavtts.model import CFM

    # 縮小版 config
    cfg = OmegaConf.load(str(pkg_files("wavtts").joinpath("configs/WavTTS.yaml")))
    cfg.model.arch.dim = 64
    cfg.model.arch.depth = 2
    cfg.model.arch.heads = 2
    cfg_path = tmp_path / "tiny.yaml"
    OmegaConf.save(cfg, str(cfg_path))

    # 對應的隨機權重 checkpoint
    arch = OmegaConf.to_container(cfg.model.arch, resolve=True)
    model_cls = get_class(f"wavtts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(**arch, wav_frame_len=cfg.model.waveform.wav_frame_len),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **OmegaConf.to_container(cfg.model.cfm, resolve=True),
    )
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, str(ckpt_path))

    from wavtts.infer.sample_uncond import main

    out_dir = tmp_path / "out"
    main(
        [
            "--ckpt", str(ckpt_path),
            "--config", str(cfg_path),
            "--duration_sec", "0.5",
            "--num", "2",
            "--steps", "2",
            "--seed", "0",
            "--out_dir", str(out_dir),
            "--device", "cpu",
        ]
    )
    wav_files = sorted(out_dir.glob("*.wav"))
    assert len(wav_files) == 2
    loaded, sr = torchaudio.load(str(wav_files[0]))
    assert loaded.shape == (1, 8000)  # 0.5s @ 16k
    assert sr == 16000

    # Test EMA checkpoint loading path
    ema_state = {f"ema_model.{k}": v for k, v in model.state_dict().items()}
    ema_state["initted"] = torch.tensor(True)
    ema_state["step"] = torch.tensor(0)
    ema_ckpt_path = tmp_path / "model_ema.pt"
    torch.save({"ema_model_state_dict": ema_state}, str(ema_ckpt_path))
    out_dir2 = tmp_path / "out_ema"
    main(
        [
            "--ckpt", str(ema_ckpt_path),
            "--config", str(cfg_path),
            "--duration_sec", "0.5",
            "--num", "1",
            "--steps", "2",
            "--seed", "0",
            "--out_dir", str(out_dir2),
            "--device", "cpu",
        ]
    )
    assert len(sorted(out_dir2.glob("*.wav"))) == 1


@pytest.mark.parametrize("cfg_strength", [2.0, 0.0])
def test_sample_dpmpp(cfg_strength):
    model = make_model()
    out, trajectory = model.sample(8000, batch=2, steps=4, cfg_strength=cfg_strength, solver="dpmpp", seed=0)
    assert out.shape == (2, 8000)
    assert torch.isfinite(out).all()
    assert trajectory.shape[0] == 5  # steps+1 states


def test_sample_rejects_unknown_solver():
    model = make_model()
    with pytest.raises(ValueError):
        model.sample(8000, steps=2, solver="bogus")


def test_sample_seed_is_isolated_and_deterministic():
    model = make_model()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    out1, _ = model.sample(3200, steps=2, seed=7)
    after = torch.rand(3)
    assert torch.equal(expected, after)  # sampling with seed must not touch global RNG
    out2, _ = model.sample(3200, steps=2, seed=7)
    assert torch.equal(out1, out2)  # same seed, same clip


def test_vanilla_rope_never_wraps_over_the_trained_position_range():
    """gamma=4 is safe on the stock spectrum: no dim completes a turn and aliases.

    Randomized positions push a 2979-frame clip out to 11916. Vanilla RoPE at
    base=10000 turns its slowest dim once every 47117 frames (471 s), so two positions
    that far apart never collide — which is why this branch needs no spectrum surgery.
    """
    from x_transformers.x_transformers import RotaryEmbedding

    dim_head, max_position = 64, 4 * 2979
    inv_freq = RotaryEmbedding(dim_head).inv_freq
    turns = max_position * inv_freq / (2 * math.pi)
    assert turns.min() < 1.0  # slowest dim has not come back around
    assert turns.min() == pytest.approx(0.253, abs=0.01)


def test_randomized_positions_sorted_unique_and_in_range():
    from wavtts.model.rope import randomized_positions

    torch.manual_seed(0)
    n, gamma = 50, 4.0
    pos = randomized_positions(batch=64, seq_len=n, gamma=gamma, device=torch.device("cpu"))
    assert pos.shape == (64, n)
    assert (pos[:, 1:] > pos[:, :-1]).all()  # strictly increasing: order preserved, no repeats
    assert pos.min() >= 0 and pos.max() < math.ceil(n * gamma)
    assert not torch.equal(pos[0], pos[1])  # independent draw per sample


def test_gamma_of_one_leaves_positions_contiguous():
    from wavtts.model.rope import randomized_positions

    pos = randomized_positions(batch=2, seq_len=50, gamma=1.0, device=torch.device("cpu"))
    assert torch.equal(pos[0], torch.arange(50))
    assert torch.equal(pos[1], torch.arange(50))


def test_every_batch_spans_contiguous_through_gamma():
    """The whole point of the random bound: no curriculum, so one batch holds both ends.

    A row's span `pos[-1] - pos[0]` tracks its own upper bound, so the spread of spans
    across a batch is the spread of stretch factors the model sees in a single update.
    """
    from wavtts.model.rope import randomized_positions

    torch.manual_seed(0)
    n, gamma = 100, 4.0
    pos = randomized_positions(batch=512, seq_len=n, gamma=gamma, device=torch.device("cpu"))
    span = (pos[:, -1] - pos[:, 0]).float()

    assert span.min() < 1.2 * n  # some rows are essentially contiguous
    assert span.max() > 3.0 * n  # others are stretched near gamma
    assert span.max() < gamma * n  # and never past it
    # the draw is uniform over the bound, so the mean sits near the middle of [1, gamma]
    assert 2.0 * n < span.mean() < 3.0 * n


def _rpe_dit(**overrides):
    from wavtts.model.backbones.dit import DiT

    kwargs = dict(
        dim=64,
        depth=2,
        heads=2,
        dim_head=32,
        ff_mult=2,
        wav_frame_len=160,
        logn_ref_len=100,
    )
    kwargs.update(overrides)
    return DiT(**kwargs)


def test_dit_forward_runs_far_past_the_reference_length():
    from wavtts.model.backbones.dit import STATE_CLEAN

    torch.manual_seed(0)
    dit = _rpe_dit()
    _reinit_nonzero(dit)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)

    for num_samples in (1600, 160 * 400):  # 10 frames, then 4x the 100-frame reference
        out = dit(x=torch.randn(2, num_samples), state=state, time=torch.tensor(0.5))
        assert out.shape == (2, num_samples)
        assert torch.isfinite(out).all()


def test_rpe_is_training_only_and_changes_output():
    from wavtts.model.backbones.dit import STATE_CLEAN

    torch.manual_seed(0)
    dit = _rpe_dit(rpe_gamma=4.0)
    _reinit_nonzero(dit)
    x = torch.randn(2, 1600)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)

    dit.eval()
    with torch.no_grad():
        assert torch.equal(dit(x=x, state=state, time=torch.tensor(0.5)), dit(x=x, state=state, time=torch.tensor(0.5)))

    dit.train()
    with torch.no_grad():
        a = dit(x=x, state=state, time=torch.tensor(0.5))
        b = dit(x=x, state=state, time=torch.tensor(0.5))
    assert not torch.allclose(a, b)  # fresh random bounds and positions every training forward


def test_logn_scaling_sharpens_but_never_flattens():
    from wavtts.model.modules import AttnProcessor

    proc = AttnProcessor(logn_ref_len=100)
    assert proc._logit_scale(100) == pytest.approx(1.0)  # at the reference length, a no-op
    assert proc._logit_scale(400) > 1.0  # longer than the reference -> sharper logits
    # clamped below the reference: flattening a short clip's softmax is not the job,
    # and unclamped it would push a 25-key softmax toward uniform
    assert proc._logit_scale(25) == pytest.approx(1.0)
    assert proc._logit_scale(2) == pytest.approx(1.0)
    assert AttnProcessor()._logit_scale(4000) == 1.0  # disabled by default


def test_logn_reads_each_sample_length_from_the_mask_not_the_padded_width():
    """A sample's temperature must not depend on who it was batched with.

    Collation pads every row to the longest in the batch, so the padded width is a
    property of the batch, not of the clip. Taking `n` from there would give the same
    audio a different temperature depending on its batch-mates, and a third one at
    inference where it arrives unpadded.
    """
    from wavtts.model.modules import AttnProcessor

    proc = AttnProcessor(logn_ref_len=100)
    mask = torch.zeros(3, 400, dtype=torch.bool)
    for row, real_len in enumerate((400, 200, 25)):
        mask[row, :real_len] = True

    scale = proc._logit_scale(400, mask)
    assert scale.shape == (3,)
    for row, real_len in enumerate((400, 200, 25)):
        assert scale[row].item() == pytest.approx(proc._logit_scale(real_len), rel=1e-6)
    assert scale[2].item() == pytest.approx(1.0)  # still clamped per row

    # an all-true mask has to agree with the scalar form, or padding-free batches would
    # silently change behaviour
    full = torch.ones(2, 400, dtype=torch.bool)
    assert proc._logit_scale(400, full)[0].item() == pytest.approx(proc._logit_scale(400), rel=1e-6)


def test_a_padded_row_attends_as_if_it_were_alone():
    """End-to-end: batching a short clip beside a long one must not change its output."""
    from wavtts.model.backbones.dit import STATE_CLEAN, DiT

    torch.manual_seed(0)
    dit = DiT(
        dim=64,
        depth=2,
        heads=2,
        dim_head=32,
        ff_mult=2,
        wav_frame_len=160,
        attn_mask_enabled=True,
        logn_ref_len=100,
    ).eval()
    _reinit_nonzero(dit)

    short, long = 160 * 30, 160 * 400  # 30 frames padded out to 400
    torch.manual_seed(1)
    clip = torch.randn(1, short)

    with torch.no_grad():
        alone = dit(
            x=clip,
            state=torch.full((1,), STATE_CLEAN, dtype=torch.long),
            time=torch.tensor(0.5),
            lens=torch.tensor([short]),
        )
        batched = dit(
            x=torch.cat([torch.nn.functional.pad(clip, (0, long - short)), torch.randn(1, long)]),
            state=torch.full((2,), STATE_CLEAN, dtype=torch.long),
            time=torch.tensor(0.5),
            lens=torch.tensor([short, long]),
        )
    assert torch.allclose(alone[0], batched[0, :short], atol=1e-5)


def test_logn_reaches_the_attention_softmax():
    from wavtts.model.backbones.dit import STATE_CLEAN, DiT

    # 10 frames against a 4-frame reference: scale is log(10)/log(4) = 1.66, well off 1.0
    common = dict(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    torch.manual_seed(0)
    off = DiT(**common)
    torch.manual_seed(0)
    on = DiT(**common, logn_ref_len=4)
    _reinit_nonzero(off)
    torch.manual_seed(0)
    _reinit_nonzero(on)

    x = torch.randn(2, 1600)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)
    with torch.no_grad():
        a = off(x=x, state=state, time=torch.tensor(0.5))
        b = on(x=x, state=state, time=torch.tensor(0.5))
    assert not torch.allclose(a, b)  # folded into the softmax scale, it must still bite


def test_log_samples_secs_pairs_with_seeds():
    from wavtts.model.trainer import pair_sample_lengths

    assert pair_sample_lengths([0, 1, 2, 3], [5, 15, 30, 60]) == [5, 15, 30, 60]
    assert pair_sample_lengths([0, 1, 2, 3], [5]) == [5, 5, 5, 5]  # pads: no seed left bare
    assert pair_sample_lengths([0, 1], [5, 15, 30, 60]) == [5, 15]  # extras dropped
    assert pair_sample_lengths([0, 1], None) == [5.0, 5.0]


def test_default_config_rope_is_unchanged_when_disabled():
    from wavtts.model.backbones.dit import STATE_CLEAN, DiT

    torch.manual_seed(0)
    plain = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    torch.manual_seed(0)
    explicit = DiT(
        dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160, rpe_gamma=1.0
    )
    x = torch.randn(2, 1600)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(
            plain(x=x, state=state, time=torch.tensor(0.5)), explicit(x=x, state=state, time=torch.tensor(0.5))
        )


def test_rng_state_survives_checkpoint_roundtrip(tmp_path):
    import random

    torch.manual_seed(999)
    random.seed(999)
    state = dict(python=random.getstate(), torch=torch.get_rng_state(), cuda=None)
    expected = (torch.rand(4), random.random())

    ckpt = tmp_path / "model_last.pt"
    torch.save({"rng_state": state}, ckpt)
    torch.manual_seed(1)  # scramble, as a fresh process would
    random.seed(1)

    loaded = torch.load(ckpt, weights_only=True)["rng_state"]  # the trainer's load path
    torch.set_rng_state(loaded["torch"])
    random.setstate(tuple(loaded["python"]))
    assert torch.equal(torch.rand(4), expected[0])
    assert random.random() == expected[1]


def test_signal_metrics():
    from wavtts.train.metrics import clipping_rate, rms, silence_ratio

    silent = torch.zeros(16000)
    assert silence_ratio(silent) == 1.0
    assert clipping_rate(silent) == 0.0

    flat = torch.ones(16000)  # DC: the least peaky signal there is, crest factor 1
    assert silence_ratio(flat) == 0.0
    assert clipping_rate(flat) == 0.0
    assert abs(rms(flat) - 1.0) < 1e-6

    # clipping_rate now reads crest factor, so it takes a genuine spike to move it
    spiky = torch.full((16000,), 0.01)
    spiky[:16] = 1.0
    assert clipping_rate(spiky) == pytest.approx(16 / 16000)


def test_mel_figure():
    import matplotlib.pyplot as plt

    from wavtts.train.metrics import mel_figure

    fig = mel_figure(torch.randn(8000))
    assert fig is not None
    plt.close(fig)


# --- full attention ----------------------------------------------------------


def test_attention_output_projection_is_not_widened():
    """Full attention emits head_dim per head; nothing here concatenates two of them."""
    attn = _rpe_dit().transformer_blocks[0].attn
    assert attn.to_out[0].in_features == attn.inner_dim


def test_padding_is_masked_out_of_the_softmax():
    from wavtts.model.modules import Attention, AttnProcessor

    torch.manual_seed(0)
    attn = Attention(processor=AttnProcessor(attn_mask_enabled=True), dim=32, heads=2, dim_head=16)
    x = torch.randn(1, 8, 32)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, :5] = True

    with torch.no_grad():
        out = attn(x, mask=mask)
        x2 = x.clone()
        x2[:, 5:] += 10.0  # only the padded frames change
        out2 = attn(x2, mask=mask)
    assert torch.allclose(out[:, :5], out2[:, :5], atol=1e-5)


# --- waveform scale ------------------------------------------------------------


def test_peak_normalize_only_scales_down():
    from wavtts.model.utils import peak_normalize

    loud = torch.randn(1, 4000) * 3.0  # training scale: peak well outside +-1
    out = peak_normalize(loud)
    assert out.abs().max().item() == pytest.approx(0.99, abs=1e-6)
    # shape preserved, only gain changed
    assert torch.allclose(out / out.abs().max(), loud / loud.abs().max(), atol=1e-6)

    quiet = torch.randn(1, 4000) * 0.01  # already inside: loudness must survive
    assert torch.equal(peak_normalize(quiet), quiet)
    assert torch.equal(peak_normalize(torch.zeros(1, 100)), torch.zeros(1, 100))


def test_signal_metrics_are_scale_invariant():
    """target_rms is a config knob, so a metric keyed to an absolute level breaks
    silently the moment it moves. These must read the same at any gain."""
    from wavtts.train.metrics import clipping_rate, silence_ratio

    torch.manual_seed(0)
    wav = torch.randn(16000)
    wav[:4000] *= 1e-4  # a stretch quiet enough to count as silence
    wav[9000] = wav.abs().max() * 30  # one spike, well past any sane crest factor

    for gain in (0.1, 1.0, 9.0):  # the old domain, the new one, and an absurd one
        assert silence_ratio(wav * gain) == pytest.approx(silence_ratio(wav), rel=1e-9)
        assert clipping_rate(wav * gain) == pytest.approx(clipping_rate(wav), rel=1e-9)
    assert clipping_rate(wav) > 0.0  # the spike is actually caught
    assert silence_ratio(wav) > 0.0

    assert silence_ratio(torch.zeros(16000)) == 1.0  # degenerate input, no divide by zero
    assert clipping_rate(torch.zeros(16000)) == 0.0


def test_clean_only_training_labels_everything_clean_and_never_mixes():
    from wavtts.model.backbones.dit import STATE_CLEAN

    model = make_model(state_null_prob=0.0, p_mix=0.5)
    mixed = []
    model._mix_augment = lambda x1, lens, flags: (mixed.append(flags.any().item()), x1)[1]
    torch.manual_seed(0)
    state, _ = _observed_states(model, batch=512)
    assert (state == STATE_CLEAN).all()
    assert mixed == [False]  # only null samples may mix, and there are none


def test_guidance_is_off_when_the_null_branch_was_never_trained():
    # a clean-only model's null branch is untrained noise; sample() must ignore
    # cfg_strength rather than steer with it
    clean_only = make_model(state_null_prob=0.0)
    _reinit_nonzero(clean_only.transformer)
    a, _ = clean_only.sample(1600, batch=1, steps=2, cfg_strength=2.0, seed=0)
    b, _ = clean_only.sample(1600, batch=1, steps=2, cfg_strength=0.0, seed=0)
    assert torch.equal(a, b)

    with_cfg = make_model(state_null_prob=0.5)
    _reinit_nonzero(with_cfg.transformer)
    c, _ = with_cfg.sample(1600, batch=1, steps=2, cfg_strength=2.0, seed=0)
    d, _ = with_cfg.sample(1600, batch=1, steps=2, cfg_strength=0.0, seed=0)
    assert not torch.allclose(c, d)  # otherwise the check above is vacuous


def test_clean_config_instantiates_model_and_trains_one_step():
    from importlib.resources import files as pkg_files

    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from wavtts.model import CFM

    cfg = OmegaConf.load(str(pkg_files("wavtts").joinpath("configs/WavTTS_clean.yaml")))
    assert cfg.model.cfm.state_null_prob == 0.0
    arch = OmegaConf.to_container(cfg.model.arch, resolve=True)
    arch.update(dim=64, depth=2, heads=2)
    cfm_kwargs = OmegaConf.to_container(cfg.model.cfm, resolve=True)
    model_cls = get_class(f"wavtts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(**arch, wav_frame_len=cfg.model.waveform.wav_frame_len),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **cfm_kwargs,
    )
    torch.manual_seed(0)
    loss, _ = model(torch.randn(2, 16000) * 0.1)
    assert torch.isfinite(loss)
    loss.backward()


# ---------------------------------------------------------------- self-flow


def _make_self_flow(depth=4, **kwargs):
    import copy

    from wavtts.model import CFM, DiT

    torch.manual_seed(0)
    transformer = DiT(dim=64, depth=depth, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    model = CFM(
        transformer=transformer,
        waveform_kwargs={"wav_frame_len": 160},
        prediction="x_pred",
        loss_space="v",
        t_eps=0.02,
        sample_rate=16000,
        self_flow=True,
        **kwargs,
    )
    _reinit_nonzero(model)
    teacher = copy.deepcopy(model.transformer).requires_grad_(False)
    return model, teacher


def test_self_flow_layer_selection():
    # l = 0.3D, k = 0.7D, the paper's l=8/k=20 at D=28
    model, _ = _make_self_flow(depth=28)
    assert (model.student_layer, model.teacher_layer) == (8, 20)


def test_self_flow_rejects_inverted_layers():
    import pytest

    with pytest.raises(ValueError, match="student layer < teacher layer"):
        _make_self_flow(student_layer_frac=0.7, teacher_layer_frac=0.3)


def test_self_flow_requires_a_teacher():
    # silently falling back to plain flow matching would waste a whole training run
    import pytest

    model, _ = _make_self_flow()
    with pytest.raises(ValueError, match="requires a teacher"):
        model(torch.randn(2, 16000) * 0.1)


def test_self_flow_train_step_backward():
    model, teacher = _make_self_flow()
    torch.manual_seed(0)
    wav = torch.randn(3, 16000) * 0.1
    lens = torch.tensor([16000, 9000, 16000])

    loss, loss_dict = model(wav, lens=lens, teacher=teacher)
    assert torch.isfinite(loss)
    # cosine similarity is bounded, so the term it contributes is too
    assert -1.0 <= loss_dict["rep_loss"].item() <= 1.0
    assert loss_dict["rep_loss"].item() != 0.0

    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.rep_proj.parameters())
    assert all(p.grad is None for p in teacher.parameters())  # teacher is EMA-driven, never trained


def test_self_flow_noises_tokens_at_two_levels():
    # the whole method rests on the input being heterogeneously noised
    model, _ = _make_self_flow()
    torch.manual_seed(0)
    tau_tok, tau_wav, tau_clean = model._dual_timestep(4, 100, 16000, torch.float32, torch.device("cpu"))

    assert tau_tok.shape == (4, 100)
    assert tau_wav.shape == (4, 16000)
    # exactly two distinct levels per row, and the teacher's is the cleaner (larger) one
    for row in range(4):
        levels = tau_tok[row].unique()
        assert len(levels) == 2
        assert torch.isclose(tau_clean[row], levels.max())
    # each token's level is held across all wav_frame_len of its samples
    assert torch.equal(tau_wav.view(4, 100, 160)[:, :, 0], tau_tok)
    assert (tau_wav.view(4, 100, 160) == tau_tok.unsqueeze(-1)).all()


def test_per_token_time_matches_scalar_time_when_uniform():
    # the adaLN rework must not change what a plain scalar timestep does
    from wavtts.model import DiT

    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160, rpe_gamma=1.0)
    _reinit_nonzero(dit)
    dit.eval()

    x = torch.randn(2, 16000) * 0.1
    state = torch.zeros(2, dtype=torch.long)
    scalar_t = torch.tensor([0.3, 0.7])
    per_token_t = scalar_t.unsqueeze(-1).expand(2, 100).contiguous()

    with torch.no_grad():
        a = dit(x=x, state=state, time=scalar_t)
        b = dit(x=x, state=state, time=per_token_t)
    assert torch.allclose(a, b, atol=1e-5)


def test_mask_block_holds_one_level_across_the_block():
    # our tokens are 10 ms raw-waveform frames, not the paper's 40 ms latents, so the mask
    # is drawn per block to restore the paper's noise-pattern time scale
    model, _ = _make_self_flow(mask_block=4)
    torch.manual_seed(0)
    tau_tok, _, _ = model._dual_timestep(4, 100, 16000, torch.float32, torch.device("cpu"))

    blocks = tau_tok.view(4, 25, 4)
    assert (blocks == blocks[:, :, :1]).all()  # one level per 4-token block
    assert len(tau_tok[0].unique()) == 2  # still exactly the two timesteps


def test_mask_block_handles_a_ragged_last_block():
    model, _ = _make_self_flow(mask_block=4)
    torch.manual_seed(0)
    tau_tok, tau_wav, _ = model._dual_timestep(2, 13, 2080, torch.float32, torch.device("cpu"))
    assert tau_tok.shape == (2, 13)  # 13 = 3 whole blocks + 1, truncated not padded
    assert tau_wav.shape == (2, 2080)
    assert (tau_tok.view(2, 13)[:, :12].reshape(2, 3, 4) == tau_tok[:, :12].reshape(2, 3, 4)[:, :, :1]).all()


def test_mask_block_1_is_the_unblocked_draw():
    # the default must stay bit-identical to the per-token draw
    model, _ = _make_self_flow(mask_block=1)
    torch.manual_seed(7)
    a, _, _ = model._dual_timestep(3, 50, 8000, torch.float32, torch.device("cpu"))
    torch.manual_seed(7)
    t = model._sample_time(3, dtype=torch.float32, device=torch.device("cpu"))
    s = model._sample_time(3, dtype=torch.float32, device=torch.device("cpu"))
    m = torch.rand((3, 50)) < model.mask_ratio
    assert torch.equal(a, torch.where(m, s.unsqueeze(-1), t.unsqueeze(-1)))


def test_mask_block_run_length_matches_the_paper_time_scale():
    # 4 x 10 ms tokens reproduce the paper's 80 ms mean run; per-token gives 20 ms
    def mean_run_ms(block, hz=100):
        model, _ = _make_self_flow(mask_block=block)
        torch.manual_seed(0)
        tau, _, _ = model._dual_timestep(1, 20000, 20000 * 160, torch.float32, torch.device("cpu"))
        row = tau[0]
        changes = (row[1:] != row[:-1]).sum().item() + 1
        return len(row) / changes * 1000 / hz

    assert 15 < mean_run_ms(1) < 25  # ~20 ms, the unblocked draw
    assert 65 < mean_run_ms(4) < 95  # ~80 ms, the paper's latent-rate scale
