#!/usr/bin/env python3
"""
test_sfvs.py
============
Unit and integration tests for sfvs.py

Tests are organised in increasing complexity:
  1. Window mask generation
  2. Integrated spectral weight
  3. Michelson contrast formulas
  4. Spectral contrast (full pipeline)
  5. ζ-penalty factor
  6. Combined SFVS — worked example from SFVS_metric.md §6
  7. Edge cases and boundary conditions
  8. Score properties (bounds, label-swap symmetry, null baseline)

Run with:
    pytest test_sfvs.py -v
  or stand-alone:
    python test_sfvs.py
"""

import sys
import os
import math
import numpy as np

# ── import the module under test ──────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sfvs import (
    # Variant B (legacy 1D)
    spectral_masks,
    integrated_weight,
    michelson_contrast_lfts,
    michelson_contrast_dnls,
    spectral_contrast,
    zeta_penalty,
    compute_sfvs,
    W_T_LO, W_T_HI, W_D_LO, W_D_HI,
    Z_T_LO, Z_T_HI, Z_D_LO, Z_D_HI,
    R_OO_NM,
    # Variant A (3D volume)
    volume_integral_2d,
    compute_sfvs_3d,
    LFTS_K_LO, LFTS_K_HI, LFTS_Z_LO, LFTS_Z_HI,
    DNLS_K_LO, DNLS_K_HI, DNLS_Z_LO, DNLS_Z_HI,
)

TOL = 1e-4   # absolute tolerance for floating-point comparisons


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_k_values(n=1001, k_lo=0.0, k_hi=2.5):
    """
    Build a k_values array where k_norm = k_values directly
    (achieved by setting r_oo = 2π so k * r_oo / 2π = k).
    """
    return np.linspace(k_lo, k_hi, n)


def make_sk_from_weights(k_values, r_oo_eff,
                         val_T, val_D, val_other=1.0):
    """
    Build a piecewise-constant S(k) that equals val_T in W_T,
    val_D in W_D, and val_other elsewhere.
    """
    k_norm = k_values * r_oo_eff / (2.0 * np.pi)
    S = np.full(len(k_values), val_other, dtype=float)
    S[(k_norm >= W_T_LO) & (k_norm <= W_T_HI)] = val_T
    S[(k_norm >= W_D_LO) & (k_norm <= W_D_HI)] = val_D
    return S


def make_labels_and_zeta(N0, N1, n_noise,
                         zeta0_center=0.8, zeta1_center=-0.2,
                         zeta_spread=0.1, seed=42):
    """
    Build synthetic label and zeta arrays with clean Gaussian distributions.
    Cluster 0 (LFTS) centred at zeta0_center, cluster 1 (DNLS) at zeta1_center.
    """
    rng = np.random.default_rng(seed)
    z0 = rng.normal(zeta0_center, zeta_spread, N0)
    z1 = rng.normal(zeta1_center, zeta_spread, N1)
    zn = rng.normal(0.3,          0.3,        n_noise)

    zeta   = np.concatenate([z0, z1, zn])
    labels = np.concatenate([
        np.zeros(N0, dtype=int),
        np.ones(N1, dtype=int),
        np.full(n_noise, -1, dtype=int),
    ])
    return zeta, labels


# ─────────────────────────────────────────────────────────────────────────────
# 1.  spectral_masks
# ─────────────────────────────────────────────────────────────────────────────

def test_spectral_masks_windows_are_non_overlapping():
    """W_T and W_D must not share any k-point."""
    k = make_k_values()
    r_oo_eff = 2.0 * np.pi   # k_norm = k_values
    mT, mD = spectral_masks(k, r_oo=r_oo_eff)
    assert not np.any(mT & mD), "Tetrahedral and disordered windows overlap"


def test_spectral_masks_contain_expected_centers():
    """k_norm=0.75 should be in W_T, k_norm=1.00 should be in W_D."""
    # Build a single-point array at the expected peak center
    r_oo_eff = 2.0 * np.pi

    k_at_fsdp = np.array([0.75])
    mT, mD = spectral_masks(k_at_fsdp, r_oo=r_oo_eff)
    assert mT[0],  "FSDP center (0.75) not inside W_T"
    assert not mD[0], "FSDP center (0.75) incorrectly inside W_D"

    k_at_liquid = np.array([1.00])
    mT, mD = spectral_masks(k_at_liquid, r_oo=r_oo_eff)
    assert mD[0],  "Liquid peak (1.00) not inside W_D"
    assert not mT[0], "Liquid peak (1.00) incorrectly inside W_T"


def test_spectral_masks_boundaries():
    """Boundary k_norm values should be inside (closed intervals)."""
    r_oo_eff = 2.0 * np.pi
    for boundary in [W_T_LO, W_T_HI]:
        k = np.array([boundary])
        mT, _ = spectral_masks(k, r_oo=r_oo_eff)
        assert mT[0], f"k_norm={boundary} should be inside W_T (closed)"
    for boundary in [W_D_LO, W_D_HI]:
        k = np.array([boundary])
        _, mD = spectral_masks(k, r_oo=r_oo_eff)
        assert mD[0], f"k_norm={boundary} should be inside W_D (closed)"


def test_spectral_masks_gap_between_windows():
    """k_norm=0.875 (midpoint of gap) should be in neither window."""
    r_oo_eff = 2.0 * np.pi
    k = np.array([0.875])
    mT, mD = spectral_masks(k, r_oo=r_oo_eff)
    assert not mT[0] and not mD[0], "Gap region should be outside both windows"


def test_spectral_masks_physical_r_oo():
    """Verify masks work correctly with the physical r_OO = 0.285 nm."""
    k_values = np.linspace(1.0, 80.0, 5000)   # nm⁻¹
    mT, mD = spectral_masks(k_values, r_oo=R_OO_NM)
    k_norm = k_values * R_OO_NM / (2.0 * np.pi)

    assert np.all(k_norm[mT] >= W_T_LO) and np.all(k_norm[mT] <= W_T_HI)
    assert np.all(k_norm[mD] >= W_D_LO) and np.all(k_norm[mD] <= W_D_HI)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  integrated_weight
# ─────────────────────────────────────────────────────────────────────────────

def test_integrated_weight_constant_function():
    """∫₀.₂ 1 dk ≈ 0.20 for constant S(k)=1 over width-0.20 window."""
    r_oo_eff = 2.0 * np.pi
    n = 2001
    k = np.linspace(0.0, 2.5, n)
    dk = k[1] - k[0]
    mT, _ = spectral_masks(k, r_oo=r_oo_eff)
    S = np.ones(n)
    result = integrated_weight(S, mT, k)
    expected = np.sum(mT) * dk          # exact for rectangular rule
    assert abs(result - expected) < TOL


def test_integrated_weight_zero_for_empty_mask():
    """Mask with no True entries should return 0.0."""
    k = np.linspace(0.0, 0.5, 100)   # entirely below W_T
    r_oo_eff = 2.0 * np.pi
    mT, _ = spectral_masks(k, r_oo=r_oo_eff)
    assert not np.any(mT), "Expected empty mask for this k range"
    result = integrated_weight(np.ones(100), mT, k)
    assert result == 0.0


def test_integrated_weight_scales_with_amplitude():
    """Doubling S(k) should double the integral."""
    r_oo_eff = 2.0 * np.pi
    k = make_k_values()
    mT, _ = spectral_masks(k, r_oo=r_oo_eff)
    S1 = np.ones(len(k))
    S2 = 2.0 * S1
    I1 = integrated_weight(S1, mT, k)
    I2 = integrated_weight(S2, mT, k)
    assert abs(I2 - 2.0 * I1) < TOL


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Michelson contrast
# ─────────────────────────────────────────────────────────────────────────────

def test_michelson_lfts_perfect():
    """C₀ = +1 when all weight is in W_T."""
    assert abs(michelson_contrast_lfts(1.0, 0.0) - 1.0) < TOL


def test_michelson_lfts_inverted():
    """C₀ = −1 when all weight is in W_D."""
    assert abs(michelson_contrast_lfts(0.0, 1.0) + 1.0) < TOL


def test_michelson_lfts_equal():
    """C₀ = 0 when weight is equally split."""
    assert abs(michelson_contrast_lfts(1.0, 1.0)) < TOL


def test_michelson_dnls_perfect():
    """C₁ = +1 when all weight is in W_D."""
    assert abs(michelson_contrast_dnls(0.0, 1.0) - 1.0) < TOL


def test_michelson_dnls_inverted():
    """C₁ = −1 when all weight is in W_T."""
    assert abs(michelson_contrast_dnls(1.0, 0.0) + 1.0) < TOL


def test_michelson_dnls_equal():
    """C₁ = 0 when weight is equally split."""
    assert abs(michelson_contrast_dnls(1.0, 1.0)) < TOL


def test_michelson_zero_denominator():
    """Both functions should return 0.0 when both integrals are zero."""
    assert michelson_contrast_lfts(0.0, 0.0) == 0.0
    assert michelson_contrast_dnls(0.0, 0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  spectral_contrast
# ─────────────────────────────────────────────────────────────────────────────

def test_spectral_contrast_perfect_separation():
    """
    S(k) perfectly matches the expected state for each cluster:
      S0: high in W_T, zero in W_D  →  C0 = +1
      S1: zero in W_T, high in W_D  →  C1 = +1
      → 𝒞 = +1
    """
    r_oo_eff = 2.0 * np.pi
    k = make_k_values()
    S0 = make_sk_from_weights(k, r_oo_eff, val_T=2.0, val_D=0.0)
    S1 = make_sk_from_weights(k, r_oo_eff, val_T=0.0, val_D=2.0)
    C, info = spectral_contrast(S0, S1, k, r_oo=r_oo_eff, f0=0.5, f1=0.5)
    assert abs(C - 1.0) < TOL, f"Expected C=1.0, got {C:.6f}"
    assert abs(info["C0"] - 1.0) < TOL
    assert abs(info["C1"] - 1.0) < TOL


def test_spectral_contrast_inverted_labels():
    """Swapping S0 and S1 should flip the sign of 𝒞."""
    r_oo_eff = 2.0 * np.pi
    k = make_k_values()
    S0 = make_sk_from_weights(k, r_oo_eff, val_T=2.0, val_D=0.5)
    S1 = make_sk_from_weights(k, r_oo_eff, val_T=0.5, val_D=2.0)
    C_correct, _  = spectral_contrast(S0, S1, k, r_oo=r_oo_eff)
    C_inverted, _ = spectral_contrast(S1, S0, k, r_oo=r_oo_eff)
    assert abs(C_correct + C_inverted) < TOL, \
        f"C + C_inverted should be 0; got {C_correct:.4f} + {C_inverted:.4f}"


def test_spectral_contrast_identical_clusters():
    """When both clusters have identical S(k), 𝒞 = 0."""
    r_oo_eff = 2.0 * np.pi
    k = make_k_values()
    S = make_sk_from_weights(k, r_oo_eff, val_T=1.5, val_D=1.5)
    C, _ = spectral_contrast(S, S, k, r_oo=r_oo_eff)
    assert abs(C) < TOL, f"Identical S(k) should give C=0, got {C:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# 5.  zeta_penalty
# ─────────────────────────────────────────────────────────────────────────────

def test_zeta_penalty_perfect():
    """
    All cluster-0 molecules at zeta=+1.0 (well inside Z_T, outside Z_D),
    all cluster-1 molecules at zeta=-0.5 (inside Z_D, outside Z_T).
    → mis0=0, mis1=0  → Λ=1.0
    """
    N0, N1 = 100, 100
    zeta   = np.concatenate([np.full(N0, 1.0), np.full(N1, -0.5)])
    labels = np.concatenate([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])
    Lambda, info = zeta_penalty(zeta, labels, f0=0.5, f1=0.5)
    assert abs(Lambda - 1.0) < TOL, f"Expected Λ=1.0, got {Lambda:.6f}"
    assert info["mis0"] == 0.0
    assert info["mis1"] == 0.0


def test_zeta_penalty_complete_mismatch():
    """
    All cluster-0 molecules in Z_D (zeta=-0.5) and
    all cluster-1 molecules in Z_T (zeta=+1.0).
    → mis0=1, mis1=1  → Λ = 1 − (0.5×1 + 0.5×1) = 0.0
    """
    N0, N1 = 100, 100
    zeta   = np.concatenate([np.full(N0, -0.5), np.full(N1, 1.0)])
    labels = np.concatenate([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])
    Lambda, info = zeta_penalty(zeta, labels, f0=0.5, f1=0.5)
    assert abs(Lambda - 0.0) < TOL, f"Expected Λ=0.0, got {Lambda:.6f}"


def test_zeta_penalty_empty_cluster_returns_nan():
    """If either cluster has no molecules, Λ should be NaN."""
    zeta   = np.array([1.0, 1.0, -0.5])
    labels = np.array([0, 0, -1])   # cluster 1 is missing
    Lambda, _ = zeta_penalty(zeta, labels, f0=1.0, f1=0.0)
    assert math.isnan(Lambda), "Expected NaN when cluster 1 is empty"


def test_zeta_penalty_only_boundary_values():
    """Molecules exactly at boundary values Z_T_LO/HI and Z_D_LO/HI."""
    N = 50
    # Cluster 0: exactly at Z_D boundaries (both endpoints inclusive)
    zeta   = np.concatenate([
        np.full(N, Z_D_LO),    # cluster 0 at lower Z_D boundary
        np.full(N, Z_T_LO),    # cluster 1 at lower Z_T boundary
    ])
    labels = np.concatenate([np.zeros(N, dtype=int), np.ones(N, dtype=int)])
    Lambda, info = zeta_penalty(zeta, labels, f0=0.5, f1=0.5)
    assert info["mis0"] == 1.0, "All cluster-0 at Z_D_LO should give mis0=1"
    assert info["mis1"] == 1.0, "All cluster-1 at Z_T_LO should give mis1=1"


def test_zeta_penalty_noise_excluded():
    """Noise molecules (label=−1) must not affect the penalty."""
    N0, N1, Nn = 100, 100, 500
    # Noise in Z_D — should not increase mis0
    zeta = np.concatenate([
        np.full(N0, 1.0),     # cluster 0: LFTS zone
        np.full(N1, -0.5),    # cluster 1: DNLS zone
        np.full(Nn, -0.5),    # noise in Z_D
    ])
    labels = np.concatenate([
        np.zeros(N0, dtype=int),
        np.ones(N1, dtype=int),
        np.full(Nn, -1, dtype=int),
    ])
    Lambda, info = zeta_penalty(zeta, labels, f0=0.5, f1=0.5)
    assert info["mis0"] == 0.0, "Noise should not affect mis0"
    assert abs(Lambda - 1.0) < TOL


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Worked example from SFVS_metric.md §6
# ─────────────────────────────────────────────────────────────────────────────

def _build_worked_example():
    """
    Reproduce exactly the numbers from §6 of SFVS_metric.md.

    N0=11,000  N1=8,500  noise=970
    I0_T=0.42  I0_D=0.18  I1_T=0.15  I1_D=0.38
    mis0=820/11000  mis1=510/8500
    Expected SFVS ≈ 0.387
    """
    # Use r_oo = 2π so that k_norm = k_values directly
    r_oo_eff = 2.0 * np.pi
    # 10001 points over [0, 2.5] → dk = 2.5/10000 = 0.00025
    k  = np.linspace(0.0, 2.5, 10_001)
    dk = k[1] - k[0]

    # Compute window widths in k-norm (= k here) to back-calculate S-values
    mT_width = np.sum((k >= W_T_LO) & (k <= W_T_HI)) * dk
    mD_width = np.sum((k >= W_D_LO) & (k <= W_D_HI)) * dk

    # Desired integrated weights
    I0_T_target, I0_D_target = 0.42, 0.18
    I1_T_target, I1_D_target = 0.15, 0.38

    # Constant S(k) in each window that reproduces the target integrals
    s0_T = I0_T_target / mT_width
    s0_D = I0_D_target / mD_width
    s1_T = I1_T_target / mT_width
    s1_D = I1_D_target / mD_width

    S0 = make_sk_from_weights(k, r_oo_eff, val_T=s0_T, val_D=s0_D)
    S1 = make_sk_from_weights(k, r_oo_eff, val_T=s1_T, val_D=s1_D)

    # Synthetic ζ arrays matching mis0=820/11000, mis1=510/8500
    N0, N1, Nn = 11_000, 8_500, 970
    rng = np.random.default_rng(0)

    # Cluster 0: 820 molecules in Z_D, rest in Z_T
    n_mis0 = 820
    z0 = np.concatenate([
        rng.uniform(Z_D_LO, Z_D_HI, n_mis0),           # "wrong" ζ
        rng.uniform(Z_T_LO, Z_T_HI, N0 - n_mis0),      # "correct" ζ
    ])

    # Cluster 1: 510 molecules in Z_T, rest in Z_D
    n_mis1 = 510
    z1 = np.concatenate([
        rng.uniform(Z_T_LO, Z_T_HI, n_mis1),           # "wrong" ζ
        rng.uniform(Z_D_LO, Z_D_HI, N1 - n_mis1),      # "correct" ζ
    ])

    zn     = rng.uniform(-0.5, 0.5, Nn)
    zeta   = np.concatenate([z0, z1, zn])
    labels = np.concatenate([
        np.zeros(N0, dtype=int),
        np.ones(N1, dtype=int),
        np.full(Nn, -1, dtype=int),
    ])

    return S0, S1, k, zeta, labels, r_oo_eff


def test_worked_example_population_fractions():
    """f0 and f1 must match §6 values."""
    N0, N1, Nn = 11_000, 8_500, 970
    f0_exp = N0 / (N0 + N1)   # 0.5641...
    f1_exp = N1 / (N0 + N1)   # 0.4359...
    assert abs(f0_exp - 0.564) < 0.001
    assert abs(f1_exp - 0.436) < 0.001


def test_worked_example_spectral_contrasts():
    """C0, C1, and 𝒞 must match the values in §6 (tolerance 0.005)."""
    S0, S1, k, zeta, labels, r_oo_eff = _build_worked_example()
    N0, N1 = 11_000, 8_500
    f0 = N0 / (N0 + N1)
    f1 = N1 / (N0 + N1)

    C, info = spectral_contrast(S0, S1, k, r_oo=r_oo_eff, f0=f0, f1=f1)

    assert abs(info["C0"] - 0.400) < 0.005, \
        f"C0 expected ≈0.400, got {info['C0']:.4f}"
    assert abs(info["C1"] - 0.434) < 0.005, \
        f"C1 expected ≈0.434, got {info['C1']:.4f}"
    assert abs(C - 0.415) < 0.005, \
        f"𝒞 expected ≈0.415, got {C:.4f}"


def test_worked_example_zeta_penalty():
    """mis0, mis1, and Λ must match §6 (tolerance 0.005)."""
    _, _, _, zeta, labels, _ = _build_worked_example()
    N0, N1 = 11_000, 8_500
    f0, f1 = N0 / (N0 + N1), N1 / (N0 + N1)

    Lambda, info = zeta_penalty(zeta, labels, f0=f0, f1=f1)

    assert abs(info["mis0"] - 820 / 11_000) < 0.001, \
        f"mis0 expected ≈0.075, got {info['mis0']:.4f}"
    assert abs(info["mis1"] - 510 / 8_500) < 0.001, \
        f"mis1 expected ≈0.060, got {info['mis1']:.4f}"
    assert abs(Lambda - 0.932) < 0.005, \
        f"Λ expected ≈0.932, got {Lambda:.4f}"


def test_worked_example_final_sfvs():
    """Full SFVS must reproduce the §6 result: SFVS ≈ 0.387 (tolerance 0.005)."""
    S0, S1, k, zeta, labels, r_oo_eff = _build_worked_example()
    sfvs, info = compute_sfvs(S0, S1, k, zeta, labels, r_oo=r_oo_eff)

    assert not math.isnan(sfvs), "SFVS should not be NaN for the worked example"
    assert abs(sfvs - 0.387) < 0.005, \
        f"SFVS expected ≈0.387, got {sfvs:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_sfvs_all_noise_returns_nan():
    """All molecules labelled noise (-1) → SFVS = NaN."""
    k = make_k_values()
    r_oo_eff = 2.0 * np.pi
    S = np.ones(len(k))
    zeta   = np.array([0.5, -0.5, 0.2])
    labels = np.array([-1, -1, -1])
    sfvs, _ = compute_sfvs(S, S, k, zeta, labels, r_oo=r_oo_eff)
    assert math.isnan(sfvs), "Expected NaN when all labels are noise"


def test_sfvs_missing_cluster_returns_nan():
    """Only one physical cluster present → SFVS = NaN."""
    k = make_k_values()
    r_oo_eff = 2.0 * np.pi
    S = np.ones(len(k))
    zeta   = np.array([1.0, 1.0, -0.5])
    labels = np.array([0, 0, -1])   # cluster 1 absent
    sfvs, _ = compute_sfvs(S, S, k, zeta, labels, r_oo=r_oo_eff)
    assert math.isnan(sfvs), "Expected NaN when cluster 1 is missing"


def test_sfvs_no_noise_molecules():
    """No noise molecules should still compute a valid SFVS."""
    r_oo_eff = 2.0 * np.pi
    k = make_k_values()
    S0 = make_sk_from_weights(k, r_oo_eff, val_T=2.0, val_D=0.5)
    S1 = make_sk_from_weights(k, r_oo_eff, val_T=0.5, val_D=2.0)
    zeta, labels = make_labels_and_zeta(N0=500, N1=400, n_noise=0)
    sfvs, _ = compute_sfvs(S0, S1, k, zeta, labels, r_oo=r_oo_eff)
    assert not math.isnan(sfvs), "SFVS should be defined even with zero noise"


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Score properties
# ─────────────────────────────────────────────────────────────────────────────

def test_sfvs_bounds():
    """SFVS must lie within [−1, +1]."""
    r_oo_eff = 2.0 * np.pi
    k = make_k_values()
    # Try several extreme configurations
    configs = [
        (2.0, 0.0, 0.0, 2.0, 0.8, -0.2),   # perfect
        (0.0, 2.0, 2.0, 0.0, 0.8, -0.2),   # inverted
        (1.5, 0.5, 0.5, 1.5, 0.8, -0.2),   # moderate
        (1.0, 1.0, 1.0, 1.0, 0.3,  0.3),   # no contrast
    ]
    for (s0T, s0D, s1T, s1D, z0c, z1c) in configs:
        S0 = make_sk_from_weights(k, r_oo_eff, val_T=s0T, val_D=s0D)
        S1 = make_sk_from_weights(k, r_oo_eff, val_T=s1T, val_D=s1D)
        zeta, labels = make_labels_and_zeta(N0=200, N1=200, n_noise=20,
                                             zeta0_center=z0c,
                                             zeta1_center=z1c)
        sfvs, _ = compute_sfvs(S0, S1, k, zeta, labels, r_oo=r_oo_eff)
        if not math.isnan(sfvs):
            assert -1.0 <= sfvs <= 1.0, \
                f"SFVS={sfvs:.4f} out of [−1,+1] for config {s0T,s0D,s1T,s1D}"


def test_sfvs_label_swap_antisymmetry():
    """
    Swapping cluster labels 0↔1 should negate the spectral contrast.
    (Λ also changes; this test verifies the sign-flip of 𝒞.)
    Property from §5.2: label swap → SFVS → −SFVS (with same |Λ|).
    """
    r_oo_eff = 2.0 * np.pi
    k = make_k_values()
    S0 = make_sk_from_weights(k, r_oo_eff, val_T=2.0, val_D=0.5)
    S1 = make_sk_from_weights(k, r_oo_eff, val_T=0.5, val_D=2.0)
    N0, N1 = 300, 200
    zeta, labels = make_labels_and_zeta(N0=N0, N1=N1, n_noise=50)

    # Compute original 𝒞 and swapped 𝒞
    f0 = N0 / (N0 + N1)
    f1 = N1 / (N0 + N1)
    C_orig,    _ = spectral_contrast(S0, S1, k, r_oo=r_oo_eff, f0=f0, f1=f1)
    C_swapped, _ = spectral_contrast(S1, S0, k, r_oo=r_oo_eff, f0=f1, f1=f0)
    assert abs(C_orig + C_swapped) < TOL, \
        f"Spectral contrast should negate on label swap: " \
        f"{C_orig:.4f} + {C_swapped:.4f} ≠ 0"


def test_sfvs_null_baseline_near_zero():
    """
    When labels are randomised, the per-cluster S(k) curves both converge
    to the mean S(k) — i.e., S0 ≈ S1.  Passing identical S(k) for both
    clusters simulates this null scenario: 𝒞 must equal 0 exactly.
    """
    r_oo_eff = 2.0 * np.pi
    k = make_k_values()
    # Both clusters see the same mixed S(k) → no spectral contrast
    S_mean = make_sk_from_weights(k, r_oo_eff, val_T=1.2, val_D=1.2)
    zeta, labels = make_labels_and_zeta(N0=500, N1=500, n_noise=50)

    sfvs, info = compute_sfvs(S_mean, S_mean, k, zeta, labels, r_oo=r_oo_eff)
    assert not math.isnan(sfvs), "Null baseline should still compute (not NaN)"
    assert abs(info["C"]) < TOL, \
        f"Identical S(k) clusters should give 𝒞=0, got {info['C']:.6f}"


def test_sfvs_monotone_with_separation():
    """
    As the spectral separation increases (S0 more peaked in W_T),
    the SFVS should increase monotonically.
    """
    r_oo_eff = 2.0 * np.pi
    k = make_k_values()
    zeta, labels = make_labels_and_zeta(N0=500, N1=400, n_noise=50)

    sfvs_values = []
    for ratio in [1.0, 1.5, 2.0, 3.0, 5.0]:   # increasing S0_T/S0_D ratio
        S0 = make_sk_from_weights(k, r_oo_eff, val_T=ratio, val_D=1.0)
        S1 = make_sk_from_weights(k, r_oo_eff, val_T=1.0,   val_D=ratio)
        sfvs, _ = compute_sfvs(S0, S1, k, zeta, labels, r_oo=r_oo_eff)
        sfvs_values.append(sfvs)

    for i in range(len(sfvs_values) - 1):
        assert sfvs_values[i] < sfvs_values[i + 1], \
            f"SFVS not monotonically increasing at step {i}: " \
            f"{sfvs_values[i]:.4f} >= {sfvs_values[i+1]:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_all():
    """Execute all test functions and report results."""
    import traceback

    all_fns = [(name, obj) for name, obj in sorted(globals().items())
               if name.startswith("test_") and callable(obj)]

    passed, failed = 0, []
    for name, fn in all_fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}")
            print(f"        {exc}")
            traceback.print_exc()
            failed.append(name)

    total = len(all_fns)
    print(f"\n{'='*60}")
    print(f"  {passed}/{total} tests passed")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print('='*60)
    return len(failed) == 0


# =============================================================================
# ─── Variant A: 3D Volume Score tests ────────────────────────────────────────
# =============================================================================

def _make_sk_zeta(n_zeta=50, n_k=200,
                  peak_k_norm=0.77, peak_z=1.2,
                  r_oo=R_OO_NM, zeta_lo=-2.0, zeta_hi=3.0):
    """Synthetic S(k,ζ) with a Gaussian peak at (k_norm, zeta)."""
    k_values = np.linspace(0.1, 50.0, n_k)
    k_norm   = k_values * r_oo / (2.0 * np.pi)
    zeta_centers = np.linspace(zeta_lo, zeta_hi, n_zeta)
    KK, ZZ = np.meshgrid(k_norm, zeta_centers)
    S = 1.0 + 4.0 * np.exp(-((KK - peak_k_norm)**2 / 0.005 +
                               (ZZ - peak_z)**2 / 0.3))
    return S, k_values, zeta_centers


def test_volume_integral_zero_outside_window():
    """volume_integral_2d returns 0 when no grid points fall in the window."""
    S, k_vals, z_cents = _make_sk_zeta()
    # Use a window completely outside the grid range
    vol = volume_integral_2d(S, k_vals, z_cents, 10.0, 11.0, 10.0, 11.0)
    assert vol == 0.0, f"Expected 0, got {vol}"


def test_volume_integral_positive_in_peak():
    """volume_integral_2d returns a positive value when peak is in the window."""
    S, k_vals, z_cents = _make_sk_zeta(peak_k_norm=0.77, peak_z=1.2)
    vol = volume_integral_2d(S, k_vals, z_cents,
                              LFTS_K_LO, LFTS_K_HI,
                              LFTS_Z_LO, LFTS_Z_HI)
    assert vol > 0.0, "Expected positive volume in LFTS window"


def test_volume_integral_less_when_peak_outside():
    """Volume is larger when peak is inside window than when outside."""
    # S0: peak in LFTS window
    S_lfts, k_vals, z_cents = _make_sk_zeta(peak_k_norm=0.77, peak_z=1.2)
    # S1: peak in DNLS window
    S_dnls, _, _ = _make_sk_zeta(peak_k_norm=0.97, peak_z=-0.5)

    vol_lfts_in_lfts = volume_integral_2d(S_lfts, k_vals, z_cents,
                                           LFTS_K_LO, LFTS_K_HI,
                                           LFTS_Z_LO, LFTS_Z_HI)
    vol_dnls_in_lfts = volume_integral_2d(S_dnls, k_vals, z_cents,
                                           LFTS_K_LO, LFTS_K_HI,
                                           LFTS_Z_LO, LFTS_Z_HI)
    assert vol_lfts_in_lfts > vol_dnls_in_lfts, \
        "LFTS surface should have higher volume in LFTS window"


def test_compute_sfvs_3d_perfect_separation():
    """Perfect clusters → SFVS-3D near +1."""
    # Cluster 0: big peak at LFTS box
    S0, k_vals, z_cents = _make_sk_zeta(peak_k_norm=0.77, peak_z=1.2)
    # Cluster 1: big peak at DNLS box
    S1, _, _ = _make_sk_zeta(peak_k_norm=0.97, peak_z=-0.5,
                              n_k=len(k_vals), n_zeta=len(z_cents))
    n = 1000
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    sfvs, info = compute_sfvs_3d(S0, S1, k_vals, z_cents, labels, verbose=False)
    assert np.isfinite(sfvs), "Expected finite SFVS-3D for perfect clusters"
    assert sfvs > 0.5, f"Expected SFVS-3D > 0.5, got {sfvs:.4f}"
    assert info["C0"] > 0, f"Expected C0 > 0 for LFTS cluster, got {info['C0']:.4f}"
    assert info["C1"] > 0, f"Expected C1 > 0 for DNLS cluster, got {info['C1']:.4f}"


def test_compute_sfvs_3d_inverted_labels():
    """Swapping cluster assignments should invert the score."""
    S0, k_vals, z_cents = _make_sk_zeta(peak_k_norm=0.77, peak_z=1.2)
    S1, _, _ = _make_sk_zeta(peak_k_norm=0.97, peak_z=-0.5,
                              n_k=len(k_vals), n_zeta=len(z_cents))
    n = 1000
    labels = np.array([0] * (n // 2) + [1] * (n // 2))

    sfvs_correct, _ = compute_sfvs_3d(S0, S1, k_vals, z_cents, labels, verbose=False)
    # Swap S0/S1 to simulate inverted labels
    sfvs_inverted, _ = compute_sfvs_3d(S1, S0, k_vals, z_cents, labels, verbose=False)

    assert sfvs_correct > 0, "Correct labeling should give positive SFVS-3D"
    assert sfvs_inverted < 0, "Inverted labeling should give negative SFVS-3D"


def test_compute_sfvs_3d_identical_clusters():
    """Identical S(k,ζ) surfaces → SFVS-3D near 0."""
    S, k_vals, z_cents = _make_sk_zeta(peak_k_norm=0.87, peak_z=0.5)
    n = 1000
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    sfvs, info = compute_sfvs_3d(S, S, k_vals, z_cents, labels, verbose=False)
    assert np.isfinite(sfvs), "Expected finite SFVS-3D for identical surfaces"
    assert abs(sfvs) < 0.05, f"Expected SFVS-3D ≈ 0, got {sfvs:.4f}"


def test_compute_sfvs_3d_empty_cluster_returns_nan():
    """If one cluster is empty, SFVS-3D should be NaN."""
    S0, k_vals, z_cents = _make_sk_zeta()
    S1, _, _ = _make_sk_zeta(n_k=len(k_vals), n_zeta=len(z_cents))
    labels = np.zeros(100, dtype=int)  # all cluster 0, no cluster 1
    sfvs, info = compute_sfvs_3d(S0, S1, k_vals, z_cents, labels, verbose=False)
    assert np.isnan(sfvs), f"Expected NaN for empty cluster, got {sfvs}"


def test_compute_sfvs_3d_all_noise_returns_nan():
    """All-noise labels → NaN."""
    S0, k_vals, z_cents = _make_sk_zeta()
    S1, _, _ = _make_sk_zeta(n_k=len(k_vals), n_zeta=len(z_cents))
    labels = np.full(100, -1, dtype=int)
    sfvs, info = compute_sfvs_3d(S0, S1, k_vals, z_cents, labels, verbose=False)
    assert np.isnan(sfvs), f"Expected NaN for all-noise labels, got {sfvs}"


def test_compute_sfvs_3d_bounds():
    """SFVS-3D must be in [−1, +1]."""
    S0, k_vals, z_cents = _make_sk_zeta(peak_k_norm=0.77, peak_z=1.2)
    S1, _, _ = _make_sk_zeta(peak_k_norm=0.97, peak_z=-0.5,
                              n_k=len(k_vals), n_zeta=len(z_cents))
    n = 1000
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    sfvs, _ = compute_sfvs_3d(S0, S1, k_vals, z_cents, labels, verbose=False)
    assert -1.0 <= sfvs <= 1.0, f"SFVS-3D out of bounds: {sfvs:.4f}"


def test_compute_sfvs_3d_nan_surface_handled():
    """NaN bins in S(k,ζ) surface are treated as zero (no crash)."""
    S0, k_vals, z_cents = _make_sk_zeta(peak_k_norm=0.77, peak_z=1.2)
    S1, _, _ = _make_sk_zeta(peak_k_norm=0.97, peak_z=-0.5,
                              n_k=len(k_vals), n_zeta=len(z_cents))
    # Introduce NaN into S0
    S0_nan = S0.copy()
    S0_nan[:10, :] = np.nan
    n = 1000
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    sfvs, info = compute_sfvs_3d(S0_nan, S1, k_vals, z_cents, labels, verbose=False)
    assert np.isfinite(sfvs), f"Expected finite SFVS-3D with NaN in surface, got {sfvs}"


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
