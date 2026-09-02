import math

import torch

from wavtts.infer.spk_cond import CCG, _renoise, generate, generate_long, integrate, invert, prepare_ref, ref_trajectory


class _MeanOfRefFlow:
    """Stub CFM. v(x, t) is the mean of the first `ref_len` samples, broadcast over the
    whole sequence — so the generated half's outcome is a closed-form function of which
    reference state was pinned in at each step, which is exactly what needs checking."""

    wav_frame_hop = 160
    latents_scale = 1.0
    t_eps = 0.02

    def __init__(self, ref_len):
        self.ref_len = ref_len

    def velocity(self, x, t, *, cfg_strength=0.0, state=None):
        return x[:, : self.ref_len].mean(dim=1, keepdim=True).expand_as(x)


class _ConstFlow:
    """v(x, t) = c everywhere: Euler integrates it exactly, so forward and backward must
    round-trip to machine precision at any step count."""

    wav_frame_hop = 160
    latents_scale = 1.0
    t_eps = 0.02

    def __init__(self, c):
        self.c = c

    def velocity(self, x, t, *, cfg_strength=0.0, state=None):
        return self.c.expand_as(x)


def test_noise_trajectory_endpoints():
    torch.manual_seed(0)
    ref = torch.randn(1, 1600)
    t = torch.linspace(0, 1, 9)
    traj = ref_trajectory(None, ref, t, mode="noise", seed=0)

    assert traj.shape == (9, 1, 1600)
    torch.testing.assert_close(traj[-1], ref)  # data end is the reference itself
    # every intermediate state is the interpolant of the same two endpoints
    for i, tv in enumerate(t):
        torch.testing.assert_close(traj[i], tv * ref + (1 - tv) * traj[0])


def test_invert_integrate_roundtrip_on_an_exact_field():
    torch.manual_seed(0)
    ref = torch.randn(1, 1600)
    model = _ConstFlow(torch.tensor([[0.3]]))
    t = torch.linspace(0, 1, 17)

    traj = invert(model, ref, t)
    torch.testing.assert_close(integrate(model, traj[0], t), ref)


def test_replacement_pins_the_reference_at_every_step():
    ref_len, gen_len, steps = 320, 480, 8
    torch.manual_seed(0)
    model = _MeanOfRefFlow(ref_len)
    t = torch.linspace(0, 1, steps + 1)
    traj = ref_trajectory(None, torch.randn(1, ref_len), t, mode="noise", seed=0)

    out = generate(model, traj, gen_len, t, seed=1)

    # the field is read from the ref half at t[i], which the previous step pinned to
    # traj[i]; the generated half is therefore its noise plus a known sum
    noise = torch.randn(1, gen_len, generator=torch.Generator().manual_seed(1))
    expected = noise + sum((t[i + 1] - t[i]) * traj[i].mean() for i in range(steps))
    torch.testing.assert_close(out, expected)


def test_renoise_restores_the_marginal():
    torch.manual_seed(0)
    x1 = torch.randn(1, 200_000)
    t_b, t_a = 0.6, 0.3
    x_b = t_b * x1 + (1 - t_b) * torch.randn_like(x1)

    x_a = _renoise(x_b, t_b, t_a)

    # x_a must look like t_a·x1 + (1-t_a)·N(0,I): the data part scaled, the noise topped
    # back up. Check both moments against the interpolant that produced x_b.
    torch.testing.assert_close((x_a - t_a * x1).std(), torch.tensor(1 - t_a), rtol=0.02, atol=0.0)
    torch.testing.assert_close(
        torch.dot((x_a - t_a * x1).flatten(), x1.flatten()) / x1.numel(),
        torch.tensor(0.0),
        rtol=0.0,
        atol=0.02,
    )


def test_prepare_ref_aligns_to_the_hop_and_normalizes_loudness(tmp_path):
    import torchaudio

    path = tmp_path / "ref.wav"
    torchaudio.save(str(path), torch.randn(1, 5000) * 0.01 + 0.2, 16000)

    wav = prepare_ref(str(path), sample_rate=16000, target_rms=1.0, hop=160)

    assert wav.shape[1] == 4960 and wav.shape[1] % 160 == 0  # whole hops only
    assert abs(float(wav.mean())) < 1e-5  # DC removed, as the loader does
    assert math.isclose(float(wav.pow(2).mean().sqrt()), 1.0, rel_tol=1e-4)


class _SplitFlow:
    """v = a on a full [ref | gen] sequence, b on the generated half alone — so the two
    guidance branches are told apart by input length and the combination is checkable."""

    wav_frame_hop = 160
    latents_scale = 1.0
    t_eps = 0.02

    def __init__(self, full_len, a, b):
        self.full_len, self.a, self.b = full_len, a, b

    def velocity(self, x, t, *, cfg_strength=0.0, state=None):
        return torch.full_like(x, self.a if x.shape[1] == self.full_len else self.b)


def test_replacement_guidance_extrapolates_away_from_the_free_branch():
    ref_len, gen_len, steps, s = 320, 480, 4, 1.5
    torch.manual_seed(0)
    model = _SplitFlow(ref_len + gen_len, a=0.4, b=0.1)
    t = torch.linspace(0, 1, steps + 1)
    traj = ref_trajectory(None, torch.randn(1, ref_len), t, mode="noise", seed=0)
    noise = torch.randn(1, gen_len, generator=torch.Generator().manual_seed(1))

    plain = generate(model, traj, gen_len, t, seed=1)
    # rescale off: the extrapolation on its own is affine, so the outcome is in closed form
    guided = generate(model, traj, gen_len, t, seed=1, ccg=CCG(w=s, rescale=0.0))

    torch.testing.assert_close(plain, noise + 0.4)  # only the full-sequence velocity
    torch.testing.assert_close(guided, noise + (0.4 + (0.4 - 0.1) * s))


def test_guidance_rescale_restores_the_conditional_scale():
    from wavtts.infer.spk_cond import _guided_velocity

    ref_len, gen_len, t = 320, 480, torch.tensor(0.4)
    torch.manual_seed(0)

    class _ScaleFlow:  # v = k·x, so the two branches disagree on scale, not just offset
        t_eps = 0.02

        def velocity(self, x, t, *, cfg_strength=0.0, state=None):
            return x * (2.0 if x.shape[1] == ref_len + gen_len else 0.5)

    model = _ScaleFlow()
    x = torch.randn(1, ref_len + gen_len)
    x_gen, denom = x[:, ref_len:], (1.0 - t).clamp_min(model.t_eps)

    def x_pred_of(v):
        return x_gen + denom * v[:, ref_len:]

    def gv(**kw):
        return x_pred_of(_guided_velocity(model, x, ref_len, t, cfg_strength=0.0, ccg=CCG(**kw)))

    cond = gv(w=0.0)
    loud = gv(w=3.0, rescale=0.0)
    fixed = gv(w=3.0, rescale=1.0)

    assert loud.std() > 2 * cond.std()  # extrapolation inflates the data prediction
    torch.testing.assert_close(fixed.std(), cond.std())  # and the rescale puts it back
    torch.testing.assert_close(fixed.mean(), cond.mean())


def test_ccg_window_gates_guidance_by_timestep():
    c = CCG(w=2.0, window=(0.1, 0.5))
    assert c.strength(0.0) == 2.0 and c.strength(0.1) == 2.0  # full strength at high noise
    assert c.strength(0.5) == 0.0 and c.strength(0.9) == 0.0  # off near the data end
    assert 0.0 < c.strength(0.3) < 2.0  # and monotone in between
    assert c.strength(0.2) > c.strength(0.4)
    assert CCG(w=2.0).strength(0.9) == 2.0  # no window = every step


def test_apg_drops_the_component_along_the_conditional_prediction():
    from wavtts.infer.spk_cond import _guided_velocity

    ref_len, gen_len, t = 160, 320, torch.tensor(0.4)
    torch.manual_seed(0)

    class _ScaleFlow:  # the two branches differ by a pure gain, so Δ is entirely parallel
        t_eps = 0.02

        def velocity(self, x, t, *, cfg_strength=0.0, state=None):
            return x * (2.0 if x.shape[1] == ref_len + gen_len else 0.5)

    model, x = _ScaleFlow(), torch.randn(1, ref_len + gen_len)
    x_cur, denom = x[:, ref_len:], (1.0 - t).clamp_min(model.t_eps)

    def x_pred_of(ccg):
        return x_cur + denom * _guided_velocity(model, x, ref_len, t, cfg_strength=0.0, ccg=ccg)[:, ref_len:]

    cond = x_pred_of(CCG(w=0.0))
    # eta=0 keeps only what is orthogonal to x_cond; here that is nothing, so guidance
    # of any strength is a no-op — which is the whole point of the projection
    torch.testing.assert_close(x_pred_of(CCG(w=5.0, eta_apg=0.0, rescale=0.0)), cond)
    assert not torch.allclose(x_pred_of(CCG(w=5.0, eta_apg=1.0, rescale=0.0)), cond)


def test_generate_long_rolls_context_and_returns_the_requested_length():
    seg, ctx, total = 480, 320, 1100
    calls = []

    class _RecordingModel:
        wav_frame_hop = 160
        latents_scale = 1.0
        t_eps = 0.02

        def velocity(self, x, t, *, cfg_strength=0.0, state=None):
            return torch.zeros_like(x)

        def sample(self, n, *, batch=1, steps=8, cfg_strength=0.0, seed=None):
            calls.append(("uncond", n))
            return torch.zeros(1, n), None

    model = _RecordingModel()
    out = generate_long(model, total, torch.linspace(0, 1, 5), seg_samples=seg, ctx_samples=ctx, seed=0)

    assert out.shape == (1, total)  # exact length, last segment truncated
    assert calls == [("uncond", seg)]  # only the first segment is context-free


def test_ola_velocity_is_a_partition_of_unity():
    from wavtts.infer.spk_cond import _ola_velocity, _window_starts

    class _Const:
        def velocity(self, x, t, *, cfg_strength=0.0, state=None):
            return torch.full_like(x, 0.7)

    x = torch.randn(1, 4800)
    for hop in (800, 1600, 3200):
        v = _ola_velocity(_Const(), x, torch.tensor(0.5), win=3200, hop=hop)
        # every sample is covered exactly once on average, so a constant survives
        torch.testing.assert_close(v, torch.full_like(x, 0.7))

    # windows must cover the whole clip, with the last one snapped to the end
    starts = _window_starts(4800, 3200, 1000)
    assert starts[0] == 0 and starts[-1] + 3200 == 4800


def test_ola_short_branch_batches_the_windows():
    from wavtts.infer.spk_cond import _ola_velocity

    seen = []

    class _Recorder:
        def velocity(self, x, t, *, cfg_strength=0.0, state=None):
            seen.append(tuple(x.shape))
            return torch.zeros_like(x)

    _ola_velocity(_Recorder(), torch.randn(1, 9600), torch.tensor(0.5), win=3200, hop=1600)
    # one batched forward, and every window is the same length — the long branch's extra
    # context is the only thing that separates the two branches
    assert seen == [(5, 3200)]


def test_guidance_base_is_what_w0_returns():
    from wavtts.infer.spk_cond import _apply_guidance

    class _M:
        t_eps = 0.02

    x, t = torch.randn(1, 320), torch.tensor(0.4)
    v_base, delta = torch.randn(1, 320), torch.randn(1, 320)
    # w=0 must hand back the sampler's own baseline untouched, bit for bit (plan §5.7-1)
    out = _apply_guidance(_M(), x, t, v_base, delta, CCG(w=0.0, lp_k=8, eta_apg=0.0))
    assert out is v_base
