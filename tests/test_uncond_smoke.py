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
        latents_scale=9.0,
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
    assert set(loss_dict) == {"total_loss", "flow_loss", "aux_mel_loss"}
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
    }
    rows = []
    for name, wav in signals.items():
        path = tmp_path / f"{name}.wav"
        torchaudio.save(str(path), wav.unsqueeze(0), 16000)
        rows.append({"audio_path": str(path), "duration": 1.0})

    ds = CustomDataset(rows, durations=[1.0] * len(rows), target_rms=0.1)
    for i in range(len(rows)):
        wav = ds[i]["wav"]
        assert wav.abs().max() <= 0.99 + 1e-6
        rms = wav.pow(2).mean().sqrt().item()
        assert rms == pytest.approx(0.1, abs=0.02) or wav.abs().max() == pytest.approx(0.99, abs=1e-3)

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


def test_yarn_inv_freq_interpolates_only_slow_dims():
    from wavtts.model.rope import yarn_inv_freq

    dim, base, native = 72, 10000.0, 3000
    vanilla = yarn_inv_freq(dim, base, 1.0, native)
    scaled = yarn_inv_freq(dim, base, 4.0, native)

    # highest-frequency dims extrapolate untouched, lowest are divided by the full scale
    assert torch.allclose(scaled[0], vanilla[0])
    assert torch.allclose(scaled[-1], vanilla[-1] / 4.0)
    # and the ramp between is monotone in how much interpolation each dim receives
    ratio = vanilla / scaled
    assert torch.all(ratio[1:] >= ratio[:-1] - 1e-6)
    assert torch.allclose(yarn_inv_freq(dim, base, 1.0, native), 1.0 / base ** (torch.arange(0, dim, 2) / dim))


def test_yarn_attention_factor():
    from wavtts.model.rope import yarn_attention_factor

    assert yarn_attention_factor(1.0) == 1.0  # no scaling, no temperature change
    assert yarn_attention_factor(4.0) == pytest.approx(0.1 * torch.tensor(4.0).log().item() + 1.0)


def test_randomized_positions_sorted_unique_and_in_range():
    from wavtts.model.rope import randomized_positions

    torch.manual_seed(0)
    pos = randomized_positions(batch=4, seq_len=50, max_len=500, device=torch.device("cpu"))
    assert pos.shape == (4, 50)
    assert (pos[:, 1:] > pos[:, :-1]).all()  # strictly increasing: order preserved, no repeats
    assert pos.min() >= 0 and pos.max() < 500
    assert not torch.equal(pos[0], pos[1])  # independent draw per sample, the default
    shared = randomized_positions(4, 50, 500, torch.device("cpu"), per_sample=False)
    assert shared.shape == (1, 50)  # one draw broadcast over the batch, freqs stay [1, n, d]
    # no room to spread -> plain contiguous positions
    tight = randomized_positions(batch=2, seq_len=50, max_len=50, device=torch.device("cpu"))
    assert torch.equal(tight[0], torch.arange(50))


def test_rpe_max_len_modes():
    from wavtts.model.rope import rpe_max_len

    assert rpe_max_len("relative", seq_len=455, length_scale=2.0, native_ctx=3000) == 910
    assert rpe_max_len("relative", seq_len=2979, length_scale=2.0, native_ctx=3000) == 5958
    assert rpe_max_len("absolute", seq_len=455, length_scale=2.0, native_ctx=3000) == 6000
    with pytest.raises(ValueError):
        rpe_max_len("nope", seq_len=1, length_scale=1.0, native_ctx=1)


def _yarn_dit(**overrides):
    from wavtts.model.backbones.dit import DiT

    kwargs = dict(
        dim=64,
        depth=2,
        heads=2,
        dim_head=32,
        ff_mult=2,
        wav_frame_len=160,
        rope_type="yarn",
        yarn_scale=2.0,
        yarn_native_ctx=100,
        logn_ref_len=100,
    )
    kwargs.update(overrides)
    return DiT(**kwargs)


def test_yarn_dit_forward_and_extrapolates_past_native_ctx():
    from wavtts.model.backbones.dit import STATE_CLEAN

    torch.manual_seed(0)
    dit = _yarn_dit()
    _reinit_nonzero(dit)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)

    for num_samples in (1600, 160 * 400):  # 10 frames, then 4x the 100-frame native ctx
        out = dit(x=torch.randn(2, num_samples), state=state, time=torch.tensor(0.5))
        assert out.shape == (2, num_samples)
        assert torch.isfinite(out).all()


def test_rpe_is_training_only_and_changes_output():
    from wavtts.model.backbones.dit import STATE_CLEAN

    torch.manual_seed(0)
    dit = _yarn_dit(rpe="relative", rpe_length_scale=2.0)
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
    assert not torch.allclose(a, b)  # fresh random positions every training forward


def test_set_yarn_scale_retunes_freqs_and_temperature():
    torch.manual_seed(0)
    dit = _yarn_dit()
    before = dit.rotary_embed.inv_freq.clone()

    dit.set_yarn_scale(4.0)
    assert not torch.allclose(dit.rotary_embed.inv_freq, before)
    assert dit.transformer_blocks[0].attn.processor.attn_temperature == pytest.approx(
        0.1 * torch.tensor(4.0).log().item() + 1.0
    )

    with pytest.raises(ValueError):
        _yarn_dit(rope_type="default").set_yarn_scale(4.0)


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


def test_rpe_curriculum_steps_on_updates():
    from wavtts.model.trainer import Trainer

    trainer = object.__new__(Trainer)
    trainer.rpe_curriculum = [[0, 1.0], [10, 1.5], [20, 2.0]]
    trainer._rpe_length_scale = None
    seen = []
    trainer.accelerator = type(
        "A", (), {"unwrap_model": staticmethod(lambda m: m), "is_main_process": False}
    )()
    trainer.model = type("M", (), {"transformer": type("T", (), {"set_rpe_length_scale": seen.append})()})()

    for update in range(0, 25):
        trainer._advance_rpe_curriculum(update)
    assert seen == [1.0, 1.5, 2.0]  # steps once per milestone, never per update


def test_default_config_rope_is_unchanged_when_disabled():
    from wavtts.model.backbones.dit import STATE_CLEAN, DiT

    torch.manual_seed(0)
    plain = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    torch.manual_seed(0)
    explicit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160, rope_type="default", rpe="off")
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
    loud = torch.ones(16000)
    assert silence_ratio(loud) == 0.0
    assert clipping_rate(loud) == 1.0
    assert abs(rms(loud) - 1.0) < 1e-6


def test_mel_figure():
    import matplotlib.pyplot as plt

    from wavtts.train.metrics import mel_figure

    fig = mel_figure(torch.randn(8000))
    assert fig is not None
    plt.close(fig)


# --- NoPE + bidirectional causal attention -----------------------------------


def _nope_dit(**overrides):
    from wavtts.model.backbones.dit import DiT

    kwargs = dict(
        dim=64,
        depth=2,
        heads=2,
        dim_head=32,
        ff_mult=2,
        wav_frame_len=160,
        rope_type="none",
        attn_mode="bidir_causal",
        logn_ref_len=100,
    )
    kwargs.update(overrides)
    return DiT(**kwargs)


def test_nope_builds_no_rotary_embedding():
    dit = _nope_dit()
    assert dit.rotary_embed is None


@pytest.mark.parametrize("mode", ["bidir_causal", "bidir_causal_split"])
def test_bidir_causal_dit_runs_and_is_position_sensitive(mode):
    from wavtts.model.backbones.dit import STATE_CLEAN

    torch.manual_seed(0)
    dit = _nope_dit(attn_mode=mode).eval()
    _reinit_nonzero(dit)
    x = torch.randn(1, 160 * 20)
    state = torch.full((1,), STATE_CLEAN, dtype=torch.long)
    with torch.no_grad():
        out = dit(x=x, state=state, time=torch.tensor(0.5))
        # reversing the frames must change the answer: with no positional encoding the
        # causal mask is the only thing that can tell the model where a frame sits
        rev = dit(x=x.view(1, 20, 160).flip(1).reshape(1, -1), state=state, time=torch.tensor(0.5))
    assert out.shape == (1, 160 * 20)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, rev.view(1, 20, 160).flip(1).reshape(1, -1), atol=1e-4)


def test_only_the_causal_modes_break_permutation_equivariance():
    """The control that makes the test above meaningful.

    Full attention with no rope is permutation-equivariant, so it carries no position
    at all; the causal modes are not. (In the assembled DiT this is not the whole story
    — `ConvPositionEmbedding` in the input embedding already supplies local relative
    position over a 61-frame window whatever `rope_type` says. So NoPE here means
    "local position from the conv, long-range position from the mask", not "none".)
    """
    from wavtts.model.modules import AttnProcessor

    torch.manual_seed(0)
    n, h, d = 6, 2, 4
    q, k, v = (torch.randn(1, h, n, d) for _ in range(3))
    perm = torch.randperm(n)

    full = AttnProcessor(attn_mode="full")
    with torch.no_grad():
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0)
        permuted = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, perm], k[:, :, perm], v[:, :, perm], scale=1.0
        )
    assert full.attn_mode == "full"
    assert torch.allclose(ref[:, :, perm], permuted, atol=1e-5)

    causal = AttnProcessor(attn_mode="bidir_causal")
    out = causal._bidir_causal(q, k, v, None, 1.0)
    out_perm = causal._bidir_causal(q[:, :, perm], k[:, :, perm], v[:, :, perm], None, 1.0)
    assert not torch.allclose(out[:, :, perm], out_perm, atol=1e-4)


def test_forward_stream_cannot_see_the_future():
    """Split mode: the past-facing heads must ignore anything after their own position."""
    from wavtts.model.modules import AttnProcessor

    torch.manual_seed(0)
    proc = AttnProcessor(attn_mode="bidir_causal_split")
    n, h, d = 6, 2, 4
    q = torch.randn(1, h, n, d)
    k = torch.randn(1, h, n, d)
    v = torch.randn(1, h, n, d)
    out = proc._bidir_causal(q, k, v, None, 1.0)

    k2, v2 = k.clone(), v.clone()
    k2[:, :, -1] += 10.0  # perturb only the last frame
    v2[:, :, -1] += 10.0
    out2 = proc._bidir_causal(q, k2, v2, None, 1.0)

    fwd_head, bwd_head = 0, h // 2
    # the past-facing head is untouched everywhere except at the perturbed frame itself
    assert torch.allclose(out[0, fwd_head, :-1], out2[0, fwd_head, :-1], atol=1e-5)
    # the future-facing head sees it from every position
    assert not torch.allclose(out[0, bwd_head, 0], out2[0, bwd_head, 0], atol=1e-4)


def test_logn_factor_uses_the_per_query_visible_count():
    from wavtts.model.modules import AttnProcessor

    proc = AttnProcessor(attn_mode="bidir_causal", logn_ref_len=100)
    visible = torch.tensor([1.0, 2.0, 50.0, 100.0, 400.0])
    f = proc._logn_factor(visible)
    # clamped at 1 below the reference, exactly 1 at it, above 1 past it — never flatten
    assert torch.allclose(f[:4], torch.ones(4), atol=1e-6)
    assert f[4] > 1.0
    assert torch.isclose(f[4], torch.tensor(math.log(400) / math.log(100)))
    # monotone in the count
    assert (f[1:] >= f[:-1]).all()


def test_bidir_causal_masked_path_matches_the_unpadded_one():
    """The explicit-mask branch and the flip fast path must agree on the real frames."""
    from wavtts.model.modules import AttnProcessor

    torch.manual_seed(0)
    proc = AttnProcessor(attn_mode="bidir_causal", logn_ref_len=4)
    n, real, h, d = 8, 5, 2, 4
    q, k, v = (torch.randn(1, h, n, d) for _ in range(3))
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[:, :real] = True

    padded = proc._bidir_causal(q, k, v, mask, 1.0)
    trimmed = proc._bidir_causal(q[:, :, :real], k[:, :, :real], v[:, :, :real], None, 1.0)
    assert torch.allclose(padded[:, :, :real], trimmed, atol=1e-5)


def test_bidir_causal_rejects_flash_backend():
    from wavtts.model.modules import AttnProcessor

    with pytest.raises(ValueError, match="needs attn_backend"):
        AttnProcessor(attn_mode="bidir_causal", attn_backend="flash_attn")


def test_bidir_causal_split_needs_even_heads():
    from wavtts.model.modules import AttnProcessor

    proc = AttnProcessor(attn_mode="bidir_causal_split")
    q, k, v = (torch.randn(1, 3, 4, 2) for _ in range(3))
    with pytest.raises(ValueError, match="even head count"):
        proc._bidir_causal(q, k, v, None, 1.0)
