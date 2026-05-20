"""Reverse-diffusion samplers for joint (M1, M2) score-based generation.

Three samplers with a common interface:

    M1_out, M2_out = sampler(M1_T, M2_T, s_theta, s_phi, vpsde, **kwargs)

Classes / functions:
    s4_sampler: Symmetric Splitting Score Sampler (S4) — the primary contribution.
    pc_sampler: Alternating predictor-corrector baseline (O(δt²) splitting error).
    em_sampler: Naive Euler-Maruyama baseline (O(δt) error, no corrector).
    SAMPLERS: Dict mapping name strings to sampler callables.

The S4 sampler generalises GDSS (arXiv:2202.02514, Jo et al. 2022) from the
(graph node features, adjacency matrix) pair to arbitrary modality pairs (M1, M2).
The symmetric Strang-style arrangement around the M2 predictor step reduces
the local splitting error from O(δt²) to O(δt³); see GDSS Appendix B.
"""

from __future__ import annotations

import typing

import torch

import sde as sde_module

ScoreFn = typing.Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def _clip_score(score: torch.Tensor, x: torch.Tensor, max_ratio: float = 5.0) -> torch.Tensor:
    """Clip score so its per-sample norm ≤ max_ratio · ‖x‖ / √d.

    At the DSM optimum the score norm is ~‖x‖ (one noise vector); this cap
    allows max_ratio× overshoot before truncating, guarding against blow-up
    from a freshly fine-tuned (or pretrained-then-finetuned) score network
    that hasn't fully calibrated output magnitudes.

    Args:
        score: Score tensor of the same shape as x.
        x: Current noisy sample.
        max_ratio: Maximum allowed ‖score‖ / (‖x‖ / √d) ratio.

    Returns:
        Score tensor with norms clipped.
    """
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    B = x.shape[0]
    x_norm = x.view(B, -1).norm(dim=1)            # (B,)
    s_norm = score.view(B, -1).norm(dim=1)         # (B,)
    # At the DSM optimum: s*(x_t,t) = -ε/σ(t), so ‖s*‖ ≈ ‖ε‖ ≈ ‖x_t‖.
    # Allow max_ratio× overshoot before clipping.
    limit = max_ratio * x_norm
    ratio = (limit / s_norm.clamp(min=1e-8)).clamp(max=1.0)
    ratio = ratio.view((-1,) + (1,) * (score.dim() - 1))
    return score * ratio


def _reverse_sde_step(
    x: torch.Tensor,
    score: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    vpsde: sde_module.VPSDE,
) -> torch.Tensor:
    """Single Euler-Maruyama step of the reverse SDE.

    Args:
        x: Current sample.
        score: Score estimate at (x, t).
        t: Current timestep of shape (B,).
        dt: Step size (positive scalar).
        vpsde: VP-SDE instance.

    Returns:
        Sample at time t − dt.
    """
    score = _clip_score(score, x)
    drift = vpsde.reverse_drift(x, score, t)
    g = vpsde.diffusion_coeff(t)
    shape = (-1,) + (1,) * (x.dim() - 1)
    return x + drift * dt + g.view(shape) * (dt**0.5) * torch.randn_like(x)


def _langevin_step(
    x: torch.Tensor,
    x_cross: torch.Tensor,
    t: torch.Tensor,
    score_fn: ScoreFn,
    snr: float,
) -> torch.Tensor:
    """Single Langevin MCMC corrector step with adaptive step size.

    Step size rule from GDSS Appendix C:
        ε = (snr · ‖x‖ / ‖score‖)²

    Args:
        x: Current sample of modality M1.
        x_cross: Current sample of the conditioning modality.
        t: Current timestep of shape (B,).
        score_fn: Score function s(x, x_cross, t).
        snr: Signal-to-noise ratio hyperparameter (≈ 0.16 per GDSS).

    Returns:
        Updated sample after one Langevin step.
    """
    score = _clip_score(score_fn(x, x_cross, t), x)
    B = x.shape[0]
    x_norm = x.view(B, -1).norm(dim=1)
    s_norm = score.view(B, -1).norm(dim=1).clamp(min=1e-8)
    # Clamp at 2×snr² ≈ 0.05 to prevent accumulation over many corrector steps.
    # At ideal score s_norm ≈ x_norm, eps = snr² = 0.0256; this cap allows
    # 2× overshoot before cutting off, preventing GroupNorm NaN on long runs.
    eps = ((snr * x_norm / s_norm) ** 2).clamp(max=2 * snr ** 2)
    eps = eps.view((-1,) + (1,) * (x.dim() - 1))
    return x + eps * score + (2 * eps).sqrt() * torch.randn_like(x)


def _s4_step(
    m1: torch.Tensor,
    m2: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    s_theta: ScoreFn,
    s_phi: ScoreFn,
    vpsde: sde_module.VPSDE,
    snr: float,
    n_corrector: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single S4 step from time t to t − dt.

    Symmetric splitting (Strang-style):
        1. Half-corrector on M1 (Langevin, n_corrector steps)
        2. Full predictor on M2 (reverse SDE Euler-Maruyama)
        3. Half-corrector on M1 (symmetric counterpart of step 1)

    Args:
        m1: Current M1 sample.
        m2: Current M2 sample.
        t: Current timestep of shape (B,).
        dt: Step size.
        s_theta: Score function for M1.
        s_phi: Score function for M2.
        vpsde: VP-SDE instance.
        snr: Langevin corrector SNR hyperparameter.
        n_corrector: Number of Langevin steps per half-corrector.

    Returns:
        Updated (M1, M2) at time t − dt.
    """
    for _ in range(n_corrector):
        m1 = _langevin_step(m1, m2, t, s_theta, snr)

    score_m2 = s_phi(m2, m1, t)
    m2 = _reverse_sde_step(m2, score_m2, t, dt, vpsde)

    for _ in range(n_corrector):
        m1 = _langevin_step(m1, m2, t, s_theta, snr)

    return m1, m2


def s4_sampler(
    m1_T: torch.Tensor,
    m2_T: torch.Tensor,
    s_theta: ScoreFn,
    s_phi: ScoreFn,
    vpsde: sde_module.VPSDE,
    n_steps: int = 1000,
    snr: float = 0.16,
    n_corrector: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full S4 reverse sampling from T → ε.

    Args:
        m1_T: M1 sample at t=T (pure noise).
        m2_T: M2 sample at t=T (pure noise).
        s_theta: ECG score function s_θ(M1_t, M2_t, t).
        s_phi: Text score function s_φ(M2_t, M1_t, t).
        vpsde: VP-SDE instance.
        n_steps: Number of discretisation steps (NFE ≈ n_steps).
        snr: Langevin corrector SNR hyperparameter.
        n_corrector: Langevin steps per S4 half-corrector.

    Returns:
        Generated (M1, M2) samples.
    """
    ts = torch.linspace(vpsde.T, vpsde.eps, n_steps + 1, device=m1_T.device)
    dt = float((vpsde.T - vpsde.eps) / n_steps)
    m1, m2 = m1_T, m2_T
    for i in range(n_steps):
        t = ts[i].expand(m1.shape[0])
        m1, m2 = _s4_step(m1, m2, t, dt, s_theta, s_phi, vpsde, snr, n_corrector)
    return m1, m2


def pc_sampler(
    m1_T: torch.Tensor,
    m2_T: torch.Tensor,
    s_theta: ScoreFn,
    s_phi: ScoreFn,
    vpsde: sde_module.VPSDE,
    n_steps: int = 1000,
    snr: float = 0.16,
    n_corrector: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternating predictor-corrector sampler (ablation baseline).

    O(δt²) splitting error vs O(δt³) for S4.
    Order per step: predict M1 → correct M1 → predict M2 → correct M2.

    Args:
        m1_T: M1 sample at t=T.
        m2_T: M2 sample at t=T.
        s_theta: Score function for M1.
        s_phi: Score function for M2.
        vpsde: VP-SDE instance.
        n_steps: Discretisation steps.
        snr: Corrector SNR.
        n_corrector: Langevin corrector steps.

    Returns:
        Generated (M1, M2) samples.
    """
    ts = torch.linspace(vpsde.T, vpsde.eps, n_steps + 1, device=m1_T.device)
    dt = float((vpsde.T - vpsde.eps) / n_steps)
    m1, m2 = m1_T, m2_T
    for i in range(n_steps):
        t = ts[i].expand(m1.shape[0])
        score_m1 = s_theta(m1, m2, t)
        m1 = _reverse_sde_step(m1, score_m1, t, dt, vpsde)
        for _ in range(n_corrector):
            m1 = _langevin_step(m1, m2, t, s_theta, snr)
        score_m2 = s_phi(m2, m1, t)
        m2 = _reverse_sde_step(m2, score_m2, t, dt, vpsde)
        for _ in range(n_corrector):
            m2 = _langevin_step(m2, m1, t, s_phi, snr)
    return m1, m2


def em_sampler(
    m1_T: torch.Tensor,
    m2_T: torch.Tensor,
    s_theta: ScoreFn,
    s_phi: ScoreFn,
    vpsde: sde_module.VPSDE,
    n_steps: int = 1000,
    snr: float = 0.16,
    n_corrector: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive Euler-Maruyama sampler (ablation baseline).

    Both modalities are stepped simultaneously with no corrector.
    O(δt) local truncation error.

    Args:
        m1_T: M1 sample at t=T.
        m2_T: M2 sample at t=T.
        s_theta: Score function for M1.
        s_phi: Score function for M2.
        vpsde: VP-SDE instance.
        n_steps: Discretisation steps.
        snr: Unused; kept for interface parity with other samplers.
        n_corrector: Unused; kept for interface parity.

    Returns:
        Generated (M1, M2) samples.
    """
    ts = torch.linspace(vpsde.T, vpsde.eps, n_steps + 1, device=m1_T.device)
    dt = float((vpsde.T - vpsde.eps) / n_steps)
    m1, m2 = m1_T, m2_T
    for i in range(n_steps):
        t = ts[i].expand(m1.shape[0])
        score_m1 = s_theta(m1, m2, t)
        score_m2 = s_phi(m2, m1, t)
        m1 = _reverse_sde_step(m1, score_m1, t, dt, vpsde)
        m2 = _reverse_sde_step(m2, score_m2, t, dt, vpsde)
    return m1, m2


SAMPLERS: dict[str, typing.Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {
    "s4": s4_sampler,
    "pc": pc_sampler,
    "em": em_sampler,
}
