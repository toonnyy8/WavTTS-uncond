"""CPU-only tests for the DDO finetuning path.

The one that matters most is test_delta_is_exactly_zero_when_ref_equals_theta: it is the
only thing standing between a working DDO round and a silently useless one, because both
failure modes it covers (unshared RoPE positions, Delta computed in bf16) produce no
error at all -- just training that does nothing.
"""

import math
from copy import deepcopy

import pytest
import torch
from torch import nn


LOG2 = math.log(2.0)


def _reinit_nonzero(model):
    # DiT zero-inits AdaLN and proj_out, so a fresh model outputs exactly 0 for every
    # input: Delta would be trivially zero and the test would prove nothing.
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)


def _make_dit(**overrides):
    from wavtts.model.backbones.dit import DiT

    kwargs = dict(
        dim=64,
        depth=2,
        heads=2,
        dim_head=32,
        ff_mult=2,
        wav_frame_len=160,
        # Every repo config sets dropout to 0.0, and DDO requires it: dropout makes
        # log p_theta differ between forwards of the same row, while log p_ref was
        # computed under some other mask, so Delta picks up the mask difference
        # (spec S3.6). The DiT default is 0.1, hence the explicit override.
        dropout=0.0,
    )
    kwargs.update(overrides)
    return DiT(**kwargs)


# ---------------------------------------------------------------- Task A1: positions


def test_positions_override_training_randomization():
    from wavtts.model.backbones.dit import STATE_CLEAN

    torch.manual_seed(0)
    dit = _make_dit(rpe_gamma=4.0)
    _reinit_nonzero(dit)
    dit.train()

    x = torch.randn(3, 1600)
    state = torch.full((3,), STATE_CLEAN, dtype=torch.long)
    time = torch.tensor(0.5)
    positions = torch.arange(10).expand(3, 10).contiguous()

    with torch.no_grad():
        a = dit(x=x, state=state, time=time, positions=positions)
        b = dit(x=x, state=state, time=time, positions=positions)
        c = dit(x=x, state=state, time=time)
        d = dit(x=x, state=state, time=time)

    assert torch.equal(a, b)  # given positions win over self.training: no fresh randperm
    assert not torch.allclose(c, d)  # and without them the augmentation still fires


def test_positions_none_is_bit_identical_to_before():
    from wavtts.model.backbones.dit import STATE_CLEAN
    from wavtts.model.rope import randomized_positions

    torch.manual_seed(0)
    dit = _make_dit(rpe_gamma=4.0)
    _reinit_nonzero(dit)

    x = torch.randn(2, 1600)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)
    time = torch.tensor(0.5)

    # Training, positions=None: must still draw its own positions the way it always did,
    # from the global RNG, in the same order -- replaying the draw by hand and passing it
    # in has to land on the same bits.
    dit.train()
    torch.manual_seed(1234)
    with torch.no_grad():
        drawn = dit(x=x, state=state, time=time)
    torch.manual_seed(1234)
    positions = randomized_positions(2, 10, 4.0, x.device)
    with torch.no_grad():
        replayed = dit(x=x, state=state, time=time, positions=positions)
    assert torch.equal(drawn, replayed)

    # Eval, positions=None: contiguous positions, unchanged and deterministic.
    dit.eval()
    with torch.no_grad():
        e1 = dit(x=x, state=state, time=time)
        e2 = dit(x=x, state=state, time=time)
    assert torch.equal(e1, e2)
    with torch.no_grad():
        contiguous = dit(x=x, state=state, time=time, positions=torch.arange(10).expand(2, 10))
    assert torch.allclose(e1, contiguous, atol=1e-6)


def test_positions_are_duplicated_for_cfg_infer():
    from wavtts.model.backbones.dit import STATE_CLEAN

    torch.manual_seed(0)
    dit = _make_dit(rpe_gamma=4.0)
    _reinit_nonzero(dit)
    dit.train()

    x = torch.randn(2, 1600)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)
    positions = torch.arange(10).expand(2, 10).contiguous()
    with torch.no_grad():
        out = dit(x=x, state=state, time=torch.tensor(0.5), cfg_infer=True, positions=positions)
    assert out.shape == (4, 1600)  # positions got catted alongside h, not left at b rows
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------- Task A2: the loss


def test_loss_at_zero_delta_is_log2():
    from wavtts.model.ddo import ddo_loss

    delta = torch.zeros(8)
    is_fake = torch.tensor([False] * 4 + [True] * 4)
    for alpha in (0.5, 1.0, 4.0):
        loss, _ = ddo_loss(delta, is_fake, alpha=alpha, beta=1.0)
        # theta == theta_ref is the analytic starting point of every round: Delta is
        # identically 0 and the loss sits in the middle of sigmoid's linear region.
        assert loss.item() == pytest.approx((1.0 + alpha) * LOG2 / max(alpha, 1.0))


def test_real_term_decreases_when_delta_grows():
    from wavtts.model.ddo import ddo_loss

    is_fake = torch.zeros(4, dtype=torch.bool)
    losses = [ddo_loss(torch.full((4,), d), is_fake, alpha=1.0, beta=1.0)[0].item() for d in (-1.0, 0.0, 1.0, 4.0)]
    assert losses == sorted(losses, reverse=True)


def test_fake_term_decreases_when_delta_drops():
    from wavtts.model.ddo import ddo_loss

    is_fake = torch.ones(4, dtype=torch.bool)
    losses = [ddo_loss(torch.full((4,), d), is_fake, alpha=1.0, beta=1.0)[0].item() for d in (1.0, 0.0, -1.0, -4.0)]
    assert losses == sorted(losses, reverse=True)


def test_alpha_scales_only_the_fake_term():
    from wavtts.model.ddo import ddo_loss

    delta = torch.tensor([0.7, 0.7, -0.3, -0.3])
    is_fake = torch.tensor([False, False, True, True])

    _, s1 = ddo_loss(delta, is_fake, alpha=1.0, beta=1.0)
    _, s3 = ddo_loss(delta, is_fake, alpha=3.0, beta=1.0)

    # Both terms are reported after the 1/max(alpha, 1) gradient-scale normalisation, so
    # the fake:real ratio is where alpha shows up -- and it shows up only there.
    r1 = s1["ddo/loss_fake"].item() / s1["ddo/loss_real"].item()
    r3 = s3["ddo/loss_fake"].item() / s3["ddo/loss_real"].item()
    assert r3 / r1 == pytest.approx(3.0)
    assert s3["ddo/loss_real"].item() == pytest.approx(s1["ddo/loss_real"].item() / 3.0)


def test_loss_is_the_sum_of_the_reported_terms():
    from wavtts.model.ddo import ddo_loss

    delta = torch.tensor([0.7, 0.2, -0.3, -1.1])
    is_fake = torch.tensor([False, False, True, True])
    loss, stats = ddo_loss(delta, is_fake, alpha=2.0, beta=0.5)
    assert loss.item() == pytest.approx(stats["ddo/loss_real"].item() + stats["ddo/loss_fake"].item())


def test_empty_side_contributes_zero():
    from wavtts.model.ddo import ddo_loss

    delta = torch.tensor([0.7, 0.2, -0.4])
    is_fake = torch.zeros(3, dtype=torch.bool)
    loss, stats = ddo_loss(delta, is_fake, alpha=2.0, beta=1.0)

    expected = -torch.nn.functional.logsigmoid(delta).mean() / 2.0
    assert loss.item() == pytest.approx(expected.item())
    assert loss.dtype == delta.dtype and loss.device == delta.device
    for key in ("ddo/loss_fake", "ddo/delta_fake", "ddo/margin", "ddo/acc"):
        assert math.isnan(stats[key].item())
    assert not math.isnan(stats["ddo/delta_real"].item())


def test_acc_is_half_at_zero_delta():
    from wavtts.model.ddo import ddo_loss

    delta = torch.zeros(6)
    is_fake = torch.tensor([False] * 3 + [True] * 3)
    _, stats = ddo_loss(delta, is_fake, alpha=1.0, beta=1.0)
    assert stats["ddo/acc"].item() == pytest.approx(0.5)
    assert stats["ddo/margin"].item() == pytest.approx(0.0)
    assert stats["ddo/delta_std"].item() == pytest.approx(0.0)


def test_acc_counts_each_side_half():
    from wavtts.model.ddo import ddo_loss

    delta = torch.tensor([1.0, 1.0, -1.0, 1.0])  # real: 2/2 right, fake: 1/2 right
    is_fake = torch.tensor([False, False, True, True])
    _, stats = ddo_loss(delta, is_fake, alpha=1.0, beta=1.0)
    assert stats["ddo/acc"].item() == pytest.approx(0.75)
    assert stats["ddo/margin"].item() == pytest.approx(1.0 - 0.0)


def test_stats_are_detached():
    from wavtts.model.ddo import ddo_loss

    delta = torch.tensor([0.5, -0.5], requires_grad=True)
    loss, stats = ddo_loss(delta, torch.tensor([False, True]), alpha=1.0, beta=1.0)
    assert loss.requires_grad
    assert not any(v.requires_grad for v in stats.values())


def test_load_ref_state_dict_prefers_ema(tmp_path):
    from wavtts.model.ddo import load_ref_state_dict

    ckpt = {
        "model_state_dict": {"w": torch.zeros(2)},
        "ema_model_state_dict": {
            "initted": torch.tensor(True),
            "step": torch.tensor(7),
            "ema_model.w": torch.ones(2),
        },
    }
    path = tmp_path / "model_last.pt"
    torch.save(ckpt, path)
    sd = load_ref_state_dict(str(path))
    assert set(sd) == {"w"}
    assert torch.equal(sd["w"], torch.ones(2))

    torch.save({"model_state_dict": {"w": torch.zeros(2)}}, path)
    assert set(load_ref_state_dict(str(path))) == {"w"}

    torch.save({"w": torch.zeros(2)}, path)
    assert set(load_ref_state_dict(str(path))) == {"w"}


# ---------------------------------------------------------------- Task A3: CFM path


def _make_cfm(*, rpe_gamma=1.0, state_null_prob=0.0, use_aux_mel_loss=False, dit_kwargs=None, **overrides):
    from wavtts.model import CFM

    transformer = _make_dit(rpe_gamma=rpe_gamma, **(dit_kwargs or {}))
    _reinit_nonzero(transformer)
    kwargs = dict(
        waveform_kwargs={"wav_frame_len": 160},
        prediction="x_pred",
        loss_space="v",
        t_eps=0.02,
        state_null_prob=state_null_prob,
        use_aux_mel_loss=use_aux_mel_loss,
        aux_mel_loss_weight=0.05,
        sample_rate=16000,
        latents_scale=1.0,
    )
    kwargs.update(overrides)
    return CFM(transformer=transformer, **kwargs)


def _attach(model, *, alpha=1.0, beta=1.0, **kwargs):
    ref = deepcopy(model)
    model.attach_ddo_ref(ref, alpha=alpha, beta=beta, **kwargs)
    return ref


@pytest.mark.parametrize(
    "dit_kwargs",
    [
        {},
        # the production config's flags: masked attention, and activation checkpointing on
        # theta that attach_ddo_ref switches off on p_ref -- the two forwards must still agree
        {"attn_mask_enabled": True, "checkpoint_activations": True},
    ],
)
def test_delta_is_exactly_zero_when_ref_equals_theta(dit_kwargs):
    """The load-bearing test: p_ref is a bit-for-bit copy of p_theta, so Delta must be 0.

    It fails if the two models draw their own randomized RoPE positions (spec S3.6) and
    it fails if Delta is computed in bf16 (spec S3.7). Both of those are invisible in
    production -- the round just quietly does nothing.
    """
    torch.manual_seed(0)
    alpha = 3.0
    model = _make_cfm(rpe_gamma=4.0, state_null_prob=0.0, dit_kwargs=dit_kwargs)
    _attach(model, alpha=alpha, beta=1.0)
    model.train()  # randomized positions are a training-mode augmentation

    wav = torch.randn(6, 1600) * 0.1
    lens = torch.tensor([1600, 1280, 1600, 960, 1600, 1440])
    is_fake = torch.tensor([False, False, False, True, True, True])

    loss, loss_dict = model(wav, lens=lens, is_fake=is_fake)

    assert loss_dict["ddo/delta_real"].item() == 0.0
    assert loss_dict["ddo/delta_fake"].item() == 0.0
    assert loss_dict["ddo/delta_std"].item() == 0.0
    assert loss_dict["ddo/acc"].item() == pytest.approx(0.5)
    # state_null_prob = 0 means no anchor rows, so the total is the DDO term alone.
    assert loss.item() == pytest.approx((1.0 + alpha) * LOG2 / max(alpha, 1.0), rel=1e-6)


def test_delta_is_nonzero_once_ref_differs():
    torch.manual_seed(0)
    model = _make_cfm(rpe_gamma=4.0, state_null_prob=0.0)
    ref = _attach(model)
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    model.train()

    wav = torch.randn(6, 1600) * 0.1
    is_fake = torch.tensor([False, False, False, True, True, True])
    _, loss_dict = model(wav, is_fake=is_fake)
    assert abs(loss_dict["ddo/delta_real"].item()) > 0.0
    assert loss_dict["ddo/delta_std"].item() > 0.0


def test_ref_is_invisible():
    model = _make_cfm()
    keys_before = set(model.state_dict())
    params_before = len(list(model.parameters()))
    buffers_before = len(list(model.buffers()))
    children_before = len(list(model.children()))

    _attach(model)

    assert set(model.state_dict()) == keys_before
    assert len(list(model.parameters())) == params_before
    assert len(list(model.buffers())) == buffers_before
    assert len(list(model.children())) == children_before
    assert model.ddo_ref is not None
    assert _make_cfm().ddo_ref is None


def _spy_transformer(model, seen):
    orig = model.transformer.forward

    def spy(*, x, state, time, mask=None, cfg_infer=False, lens=None, positions=None):
        seen.setdefault("state", []).append(state.clone())
        seen.setdefault("time", []).append(time.clone())
        return orig(x=x, state=state, time=time, mask=mask, cfg_infer=cfg_infer, lens=lens, positions=positions)

    model.transformer.forward = spy


def test_fake_rows_are_clean_and_unmixed():
    from wavtts.model.backbones.dit import STATE_CLEAN

    torch.manual_seed(0)
    model = _make_cfm(state_null_prob=1.0, p_mix=1.0)  # every real row would be null+mixed
    _attach(model)
    model.train()

    seen = {}
    _spy_transformer(model, seen)
    orig_mix = model._mix_augment

    def mix_spy(x1, lens, mix_flags):
        seen["mix_flags"] = mix_flags.clone()
        return orig_mix(x1, lens, mix_flags)

    model._mix_augment = mix_spy

    is_fake = torch.tensor([False, True, False, True, True, False])
    model(torch.randn(6, 1600) * 0.1, is_fake=is_fake)

    assert not seen["mix_flags"][is_fake].any()  # fakes are never mixed
    assert (seen["state"][0][is_fake] == STATE_CLEAN).all()  # and never null


def test_real_and_fake_rows_share_time_draws():
    torch.manual_seed(0)
    model = _make_cfm(state_null_prob=0.0)
    _attach(model)
    model.train()

    seen = {}
    _spy_transformer(model, seen)

    is_fake = torch.tensor([False, True, True, False, True])
    model(torch.randn(5, 1600) * 0.1, is_fake=is_fake)

    time = seen["time"][0]
    real_idx = (~is_fake).nonzero(as_tuple=True)[0]
    fake_idx = is_fake.nonzero(as_tuple=True)[0]
    paired = real_idx[torch.arange(fake_idx.numel()) % real_idx.numel()]
    assert torch.equal(time[fake_idx], time[paired])


def test_real_and_fake_rows_share_noise_draws():
    """The paper's common random numbers are (t, eps), not t alone. The rows of a batch are
    padded to one width, so eps has the same shape on both sides and a fake row can reuse
    its partner real row's noise at the same sample positions."""
    torch.manual_seed(0)
    model = _make_cfm(state_null_prob=0.0)
    is_fake = torch.tensor([False, True, True, False, True])
    x1 = torch.randn(5, 1600)

    fake_idx, partner = model._ddo_pair_rows(is_fake)
    assert torch.equal(fake_idx, torch.tensor([1, 2, 4]))
    assert torch.equal(partner, torch.tensor([0, 3, 0]))  # by position, wrapping around

    x0 = model._ddo_sample_noise(x1, is_fake)
    assert x0.shape == x1.shape
    assert torch.equal(x0[fake_idx], x0[partner])
    assert not torch.equal(x0[0], x0[3])  # the real rows keep independent draws

    # a one-sided batch pairs nothing and still gets a full batch of noise
    fake_idx, partner = model._ddo_pair_rows(torch.zeros(5, dtype=torch.bool))
    assert fake_idx.numel() == 0 and partner.numel() == 0
    assert model._ddo_sample_noise(x1, torch.zeros(5, dtype=torch.bool)).shape == x1.shape


def test_ddo_backward_gives_grads_to_theta_only():
    torch.manual_seed(0)
    model = _make_cfm(rpe_gamma=4.0, state_null_prob=0.5)
    ref = _attach(model)
    model.train()

    is_fake = torch.tensor([False, False, True, True])
    loss, _ = model(torch.randn(4, 1600) * 0.1, is_fake=is_fake)
    loss.backward()

    assert all(p.grad is None for p in ref.parameters())
    assert all(not p.requires_grad for p in ref.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize(
    "flags, state_null_prob",
    [
        ([False, False, False, False], 0.0),  # all real, clean arm: no anchor rows
        ([True, True, True, True], 0.5),  # all fake: the real term is empty
        ([False, False, False, False], 1.0),  # all anchor: the DDO term is empty
        ([False, True, False, True], 0.5),  # the normal mix
    ],
)
def test_all_real_and_all_fake_batches_do_not_crash(flags, state_null_prob):
    torch.manual_seed(0)
    model = _make_cfm(rpe_gamma=4.0, state_null_prob=state_null_prob)
    _attach(model, alpha=2.0, beta=0.5)
    model.train()

    loss, loss_dict = model(torch.randn(4, 1600) * 0.1, is_fake=torch.tensor(flags))
    assert torch.isfinite(loss)
    loss.backward()

    expected = {
        "total_loss",
        "flow_loss",
        "aux_mel_loss",
        "anchor_loss",
        "ddo/loss_real",
        "ddo/loss_fake",
        "ddo/delta_real",
        "ddo/delta_fake",
        "ddo/delta_std",
        "ddo/margin",
        "ddo/acc",
    }
    assert set(loss_dict) == expected  # the trainer logs these keys blind; nan means "absent"


def test_anchor_rows_carry_the_pretraining_loss():
    torch.manual_seed(0)
    model = _make_cfm(rpe_gamma=4.0, state_null_prob=1.0)
    _attach(model, anchor_weight=2.0)
    model.train()

    torch.manual_seed(7)
    wav = torch.randn(4, 1600) * 0.1
    loss, loss_dict = model(wav, is_fake=torch.zeros(4, dtype=torch.bool))

    # Every row is null here, so there is nothing for the discriminator to look at and
    # the total is the weighted anchor alone.
    assert math.isnan(loss_dict["ddo/delta_real"].item())
    assert loss.item() == pytest.approx(loss_dict["anchor_loss"].item())
    assert loss_dict["anchor_loss"].item() == pytest.approx(2.0 * loss_dict["flow_loss"].item(), rel=1e-5)


def test_aux_mel_rides_along_with_the_anchor_only():
    torch.manual_seed(0)
    model = _make_cfm(rpe_gamma=4.0, state_null_prob=0.5, use_aux_mel_loss=True)
    _attach(model)
    model.train()

    is_fake = torch.tensor([False, False, True, True, False, True])
    loss, loss_dict = model(torch.randn(6, 3200) * 0.1, is_fake=is_fake)
    assert torch.isfinite(loss)
    # The mel term is a perceptual regularizer, not part of the ELBO, so it stays out of
    # Delta and lives only on the anchor rows (spec S3.1).
    assert loss_dict["aux_mel_loss"].item() > 0.0
    assert loss_dict["anchor_loss"].item() > loss_dict["aux_mel_loss"].item()


def test_attach_rejects_an_unknown_delta_normalize():
    model = _make_cfm()
    with pytest.raises(ValueError):
        model.attach_ddo_ref(deepcopy(model), alpha=1.0, beta=1.0, delta_normalize="rms")


def test_delta_normalize_sum_scales_by_row_length():
    torch.manual_seed(0)
    model = _make_cfm(rpe_gamma=4.0, state_null_prob=0.0)
    ref = _attach(model, delta_normalize="mean")
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    wav = torch.randn(2, 1600) * 0.1
    lens = torch.tensor([1600, 1600])
    is_fake = torch.tensor([False, True])

    model.eval()  # no randomized positions, so both runs see identical inputs
    torch.manual_seed(3)
    _, mean_dict = model(wav, lens=lens, is_fake=is_fake)
    model.ddo_delta_normalize = "sum"
    torch.manual_seed(3)
    _, sum_dict = model(wav, lens=lens, is_fake=is_fake)

    assert sum_dict["ddo/delta_real"].item() == pytest.approx(1600 * mean_dict["ddo/delta_real"].item(), rel=1e-4)


def test_forward_without_ddo_is_unchanged():
    torch.manual_seed(0)
    model = _make_cfm(rpe_gamma=4.0, state_null_prob=0.5)
    twin = deepcopy(model)
    model.train()
    twin.train()

    wav = torch.randn(4, 1600) * 0.1
    lens = torch.tensor([1600, 1280, 960, 1600])

    torch.manual_seed(11)
    loss_a, dict_a = model(wav, lens=lens)
    torch.manual_seed(11)
    loss_b, dict_b = twin(wav, lens=lens, is_fake=torch.tensor([False, True, False, True]))

    # No reference attached: is_fake is ignored and the loss is the pretraining one,
    # down to the bit, with exactly the three keys it always had.
    assert set(dict_a) == set(dict_b) == {"total_loss", "flow_loss", "aux_mel_loss"}
    assert torch.equal(loss_a, loss_b)


def test_per_sample_loss_upcasts_before_squaring():
    """The cast has to happen on the inputs, not on the result.

    Under autocast the backbone hands back bf16 and everything downstream of a mixed
    bf16/fp32 op happens to promote, so a missing .float() hides here and only shows up
    the day something upstream starts feeding bf16 on both sides -- by which point Delta
    is quantization noise and nothing raises. Feed it bf16 on every input instead.
    """
    torch.manual_seed(0)
    model = _make_cfm()
    wav = torch.randn(2, 1600) * 0.1
    time = torch.rand(2)
    mask = torch.ones(2, 1600, dtype=torch.bool)

    out = model._per_sample_loss(
        wav.bfloat16(),
        wav.bfloat16(),
        wav.bfloat16(),
        time.bfloat16(),
        mask,
    )
    assert out.dtype == torch.float32


def test_delta_is_fp32_under_bf16_autocast():
    import wavtts.model.cfm as cfm_module

    torch.manual_seed(0)
    model = _make_cfm(rpe_gamma=4.0, state_null_prob=0.5)
    _attach(model)
    model.train()

    seen = {}
    orig = cfm_module.ddo_loss

    def capture(delta, is_fake, **kwargs):
        seen["delta"] = delta
        return orig(delta, is_fake, **kwargs)

    cfm_module.ddo_loss = capture
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss_dict = model(torch.randn(4, 1600) * 0.1, is_fake=torch.tensor([False, False, True, True]))
    finally:
        cfm_module.ddo_loss = orig

    assert seen["delta"].dtype == torch.float32
    assert loss_dict["ddo/delta_real"].dtype == torch.float32
    assert loss_dict["total_loss"].dtype == torch.float32
