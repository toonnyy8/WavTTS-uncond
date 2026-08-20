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


def test_dit_cfg_infer_packs_pos_neg():
    from wavtts.model.backbones.dit import STATE_CLEAN, STATE_MIXED, DiT

    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    _reinit_nonzero(dit)
    x = torch.randn(2, 1600)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)
    neg = torch.full((2,), STATE_MIXED, dtype=torch.long)
    out = dit(x=x, state=state, time=torch.tensor(0.5), cfg_infer=True, neg_state=neg)
    assert out.shape == (4, 1600)
    pos, negp = torch.chunk(out, 2, dim=0)
    assert not torch.allclose(pos, negp)  # 不同 state 必須產生不同輸出


def test_dit_state_changes_output():
    from wavtts.model.backbones.dit import STATE_CLEAN, STATE_MIXED, DiT

    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    _reinit_nonzero(dit)
    x = torch.randn(1, 1600)
    t = torch.tensor(0.5)
    out_clean = dit(x=x, state=torch.tensor([STATE_CLEAN]), time=t)
    out_mixed = dit(x=x, state=torch.tensor([STATE_MIXED]), time=t)
    assert not torch.allclose(out_clean, out_mixed)


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


def test_mix_augment_labels_and_content():
    from wavtts.model.backbones.dit import STATE_CLEAN, STATE_MIXED

    model = make_model(p_mix=1.0)
    torch.manual_seed(0)
    x = torch.randn(4, 3200)
    lens = torch.full((4,), 3200, dtype=torch.long)
    x_aug, state = model._mix_augment(x, lens)
    assert (state == STATE_MIXED).all()
    assert not torch.allclose(x_aug, x)

    model.p_mix = 0.0
    x_aug, state = model._mix_augment(x, lens)
    assert (state == STATE_CLEAN).all()
    assert torch.equal(x_aug, x)


def test_mix_augment_concat_prefix_preserved():
    model = make_model(p_mix=1.0, p_concat=1.0)
    torch.manual_seed(0)
    x = torch.randn(2, 3200)
    lens = torch.full((2,), 3200, dtype=torch.long)
    x_aug, _state = model._mix_augment(x, lens)
    # 切換點最早在 0.3*3200=960，之前的內容必須原封不動（no leaky：前段就是原語者）
    assert torch.equal(x_aug[:, :900], x[:, :900])


def test_mix_augment_batch_of_one_is_noop():
    from wavtts.model.backbones.dit import STATE_CLEAN

    model = make_model(p_mix=1.0)
    x = torch.randn(1, 3200)
    lens = torch.full((1,), 3200, dtype=torch.long)
    x_aug, state = model._mix_augment(x, lens)
    assert torch.equal(x_aug, x)
    assert (state == STATE_CLEAN).all()


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


@pytest.mark.parametrize(
    "cfg_strength,negative",
    [(2.0, "mixed"), (2.0, "null"), (0.0, "mixed")],
)
def test_sample_shapes(cfg_strength, negative):
    model = make_model()
    out, trajectory = model.sample(
        8000, batch=2, steps=2, cfg_strength=cfg_strength, negative=negative, seed=0
    )
    assert out.shape == (2, 8000)
    assert torch.isfinite(out).all()
    assert trajectory.shape[0] == 3  # steps+1 個時間點


def test_sample_duration_not_multiple_of_frame_len():
    model = make_model()
    out, _ = model.sample(8123, batch=1, steps=2, seed=0)
    assert out.shape == (1, 8123)


def test_sample_rejects_unknown_negative():
    model = make_model()
    with pytest.raises(ValueError):
        model.sample(8000, steps=2, negative="bogus")


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
    assert pos.shape == (1, 50)  # broadcast over the batch, so freqs stay [1, n, d]
    assert (pos[:, 1:] > pos[:, :-1]).all()  # strictly increasing: order preserved, no repeats
    assert pos.min() >= 0 and pos.max() < 500
    per_sample = randomized_positions(4, 50, 500, torch.device("cpu"), per_sample=True)
    assert per_sample.shape == (4, 50)
    assert not torch.equal(per_sample[0], per_sample[1])
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


def test_logn_scaling_is_length_dependent():
    from wavtts.model.modules import AttnProcessor

    proc = AttnProcessor(logn_ref_len=100)
    assert proc._logit_scale(100) == pytest.approx(1.0)  # at the reference length, a no-op
    assert proc._logit_scale(400) > 1.0  # longer than trained -> hotter logits
    assert proc._logit_scale(25) < 1.0
    assert AttnProcessor()._logit_scale(4000) == 1.0  # disabled by default


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
