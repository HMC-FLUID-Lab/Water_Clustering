#!/usr/bin/env python3
"""
sfvs.py
=======
Structure Factor Validation Score (SFVS)

Two variants are provided:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Variant A — 3D Volume Score  (recommended, simpler)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Uses the 2D surface S(k, ζ) per cluster.
  Integrates over rectangular (k, ζ) boxes and computes a
  Michelson contrast between the "correct" and "wrong" box:

  For Cluster 0 (LFTS):
    V₀_lfts = ∬_{LFTS box} S₀(k,ζ) dk dζ     ← should be HIGH
    V₀_dnls = ∬_{DNLS box} S₀(k,ζ) dk dζ     ← penalty if non-zero
    C₀ = (V₀_lfts − V₀_dnls) / (V₀_lfts + V₀_dnls)

  For Cluster 1 (DNLS):
    V₁_dnls = ∬_{DNLS box} S₁(k,ζ) dk dζ     ← should be HIGH
    V₁_lfts = ∬_{LFTS box} S₁(k,ζ) dk dζ     ← penalty if non-zero
    C₁ = (V₁_dnls − V₁_lfts) / (V₁_dnls + V₁_lfts)

  SFVS_3D = f₀·C₀ + f₁·C₁

  Windows (k·r_OO/2π, ζ in Å):
    LFTS box:  k_norm ∈ [0.70, 0.85],  ζ ∈ [1.0, 1.5]
    DNLS box:  k_norm ∈ [0.90, 1.05],  ζ ∈ [−1.0, 0.0]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Variant B — Original 1D Score  (legacy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Uses S(k) per cluster (1D) + separate ζ fraction check.
  SFVS = C × Λ  (spectral contrast × ζ-penalty)

Reference
---------
  Shi & Tanaka, JACS 2020 — two-state model for water.

Score range and interpretation
-------------------------------
  > 0.7        Excellent — strong two-state separation in both S(k) and ζ
  0.4 – 0.7    Good      — clear separation with some transition-region overlap
  0.1 – 0.4    Weak      — marginal; clusters partially overlap
  < 0.1        Failed    — clustering does not recover physically distinct states
  Negative     Inverted labels — swap 0 ↔ 1

Convention
----------
  Cluster 0 = LFTS-like (tetrahedral, high ζ, FSDP peak)
  Cluster 1 = DNLS-like (disordered, low ζ, main liquid peak)
  Label  −1 = noise — excluded from all scoring
"""

import numpy as np
import warnings
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants and window boundaries
# ─────────────────────────────────────────────────────────────────────────────

R_OO_NM       = 0.285          # mean O–O distance (nm)

# ── Variant A: 3D (k, ζ) box windows ────────────────────────────────────────
# k axis  : normalised  k·r_OO/2π  (dimensionless)
# ζ axis  : Å  (sk_zeta_3d.py auto-converts nm → Å)
LFTS_K_LO,  LFTS_K_HI  = 0.70,  0.85   # FSDP / tetrahedral k-window
LFTS_Z_LO,  LFTS_Z_HI  = 1.0,   1.5    # LFTS ζ-window (Å)

DNLS_K_LO,  DNLS_K_HI  = 0.90,  1.05   # main liquid k-window
DNLS_Z_LO,  DNLS_Z_HI  = -1.0,  0.0    # DNLS ζ-window (Å)

# ── Variant B: 1D spectral windows (legacy) ──────────────────────────────────
W_T_LO, W_T_HI = 0.65, 0.85   # tetrahedral / FSDP window
W_D_LO, W_D_HI = 0.90, 1.10   # disordered / liquid window

# ζ-diagnostic windows for Variant B penalty
Z_T_LO, Z_T_HI =  0.5,  1.5   # LFTS-expected ζ range
Z_D_LO, Z_D_HI = -1.0,  0.0   # DNLS-expected ζ range


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Spectral window masks
# ─────────────────────────────────────────────────────────────────────────────

def spectral_masks(k_values: np.ndarray,
                   r_oo: float = R_OO_NM) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return boolean masks for the tetrahedral (W_T) and disordered (W_D)
    spectral windows.

    Parameters
    ----------
    k_values : 1-D array of wave-vector magnitudes (nm⁻¹)
    r_oo     : O–O reference distance (nm, default 0.285)

    Returns
    -------
    mask_T, mask_D : boolean arrays of length len(k_values)
    """
    k_norm = k_values * r_oo / (2.0 * np.pi)
    mask_T = (k_norm >= W_T_LO) & (k_norm <= W_T_HI)
    mask_D = (k_norm >= W_D_LO) & (k_norm <= W_D_HI)
    return mask_T, mask_D


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Integrated spectral weights
# ─────────────────────────────────────────────────────────────────────────────

def integrated_weight(S_k: np.ndarray,
                      mask: np.ndarray,
                      k_values: np.ndarray) -> float:
    """
    Numerical integration  Iₐᵂ = Σ_{k∈W} Sₐ(k) Δk  (rectangular rule).

    Parameters
    ----------
    S_k      : structure factor array (same length as k_values)
    mask     : boolean window mask from spectral_masks()
    k_values : wave-vector magnitudes (nm⁻¹)

    Returns
    -------
    float   — integrated spectral weight; 0.0 if no k-points fall in window
    """
    if not np.any(mask):
        return 0.0
    dk = float(k_values[1] - k_values[0]) if len(k_values) > 1 else 1.0
    return float(np.sum(S_k[mask]) * dk)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-cluster Michelson contrast  (§3.3)
# ─────────────────────────────────────────────────────────────────────────────

def michelson_contrast_lfts(I_T: float, I_D: float) -> float:
    """
    C₀ = (I₀ᵀ − I₀ᴰ) / (I₀ᵀ + I₀ᴰ)

    +1 → all weight in tetrahedral window (correct for LFTS)
     0 → equal weight
    −1 → all weight in disordered window (inverted)
    """
    denom = I_T + I_D
    if denom == 0.0:
        return 0.0
    return (I_T - I_D) / denom


def michelson_contrast_dnls(I_T: float, I_D: float) -> float:
    """
    C₁ = (I₁ᴰ − I₁ᵀ) / (I₁ᴰ + I₁ᵀ)

    +1 → all weight in disordered window (correct for DNLS)
     0 → equal weight
    −1 → all weight in tetrahedral window (inverted)
    """
    denom = I_T + I_D
    if denom == 0.0:
        return 0.0
    return (I_D - I_T) / denom


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Population-weighted spectral contrast  𝒞  (§3.4)
# ─────────────────────────────────────────────────────────────────────────────

def spectral_contrast(S0_k: np.ndarray,
                      S1_k: np.ndarray,
                      k_values: np.ndarray,
                      r_oo: float = R_OO_NM,
                      f0: Optional[float] = None,
                      f1: Optional[float] = None) -> Tuple[float, dict]:
    """
    Compute population-weighted spectral contrast  𝒞 = f₀·C₀ + f₁·C₁.

    If f0/f1 are not supplied, equal weighting (0.5 / 0.5) is used.

    Returns
    -------
    C     : float — spectral contrast in [−1, +1]
    info  : dict  — intermediate quantities for reporting / debugging
    """
    if f0 is None or f1 is None:
        f0 = f1 = 0.5

    mask_T, mask_D = spectral_masks(k_values, r_oo)

    I0_T = integrated_weight(S0_k, mask_T, k_values)
    I0_D = integrated_weight(S0_k, mask_D, k_values)
    I1_T = integrated_weight(S1_k, mask_T, k_values)
    I1_D = integrated_weight(S1_k, mask_D, k_values)

    C0 = michelson_contrast_lfts(I0_T, I0_D)
    C1 = michelson_contrast_dnls(I1_T, I1_D)
    C  = f0 * C0 + f1 * C1

    info = dict(I0_T=I0_T, I0_D=I0_D, I1_T=I1_T, I1_D=I1_D,
                C0=C0, C1=C1, C=C, f0=f0, f1=f1)
    return C, info


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ζ-penalty factor  Λ  (§4)
# ─────────────────────────────────────────────────────────────────────────────

def zeta_penalty(zeta: np.ndarray,
                 labels: np.ndarray,
                 f0: float,
                 f1: float) -> Tuple[float, dict]:
    """
    Λ = 1 − (f₀·mis₀ + f₁·mis₁)

    mis₀ = fraction of LFTS molecules (label 0) with ζ ∈ Z_D = [−1.0, 0.0]
    mis₁ = fraction of DNLS molecules (label 1) with ζ ∈ Z_T = [+0.5, 1.5]

    Returns
    -------
    Lambda : float in [0, 1]
    info   : dict of intermediate values
    """
    C0_mask = labels == 0
    C1_mask = labels == 1

    n0 = int(C0_mask.sum())
    n1 = int(C1_mask.sum())

    if n0 == 0 or n1 == 0:
        return np.nan, {}

    zeta0 = zeta[C0_mask]
    zeta1 = zeta[C1_mask]

    mis0 = float(np.sum((zeta0 >= Z_D_LO) & (zeta0 <= Z_D_HI)) / n0)
    mis1 = float(np.sum((zeta1 >= Z_T_LO) & (zeta1 <= Z_T_HI)) / n1)

    Lambda = 1.0 - (f0 * mis0 + f1 * mis1)

    info = dict(mis0=mis0, mis1=mis1, n0=n0, n1=n1, Lambda=Lambda)
    return Lambda, info


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Combined SFVS  (§5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sfvs(S0_k: np.ndarray,
                 S1_k: np.ndarray,
                 k_values: np.ndarray,
                 zeta: np.ndarray,
                 labels: np.ndarray,
                 r_oo: float = R_OO_NM,
                 verbose: bool = False) -> Tuple[float, dict]:
    """
    Compute the Structure Factor Validation Score.

        SFVS = 𝒞 × Λ

    Parameters
    ----------
    S0_k     : per-cluster S(k) array for cluster 0 (LFTS)  shape (n_k,)
    S1_k     : per-cluster S(k) array for cluster 1 (DNLS)  shape (n_k,)
    k_values : wave-vector magnitudes in nm⁻¹                shape (n_k,)
    zeta     : ζ order parameter per molecule                 shape (n_mol,)
    labels   : cluster labels per molecule (−1 / 0 / 1)      shape (n_mol,)
    r_oo     : O–O reference distance (nm, default 0.285)
    verbose  : print a detailed breakdown

    Returns
    -------
    sfvs  : float — the final score (NaN if computation is undefined)
    info  : dict  — all intermediate quantities

    Notes
    -----
    - Noise molecules (label == −1) are excluded from f₀, f₁, and the ζ-penalty.
    - Returns NaN if either cluster is empty.
    """
    # Population fractions (noise excluded)
    clean_mask = labels != -1
    N_clean    = int(clean_mask.sum())
    N0         = int((labels == 0).sum())
    N1         = int((labels == 1).sum())

    if N0 == 0 or N1 == 0 or N_clean == 0:
        warnings.warn("SFVS undefined: one or both clusters are empty.")
        return np.nan, {}

    f0 = N0 / N_clean
    f1 = N1 / N_clean

    # Spectral contrast
    C, sc_info = spectral_contrast(S0_k, S1_k, k_values, r_oo, f0, f1)

    # ζ-penalty
    Lambda, zp_info = zeta_penalty(zeta, labels, f0, f1)
    if np.isnan(Lambda):
        warnings.warn("SFVS undefined: ζ-penalty could not be computed.")
        return np.nan, {}

    sfvs = C * Lambda

    info = {
        "N0": N0, "N1": N1, "N_noise": int((labels == -1).sum()),
        "f0": f0, "f1": f1,
        **sc_info,
        **zp_info,
        "C": C, "Lambda": Lambda, "sfvs": sfvs,
    }

    if verbose:
        _print_report(info)

    return sfvs, info


def _print_report(info: dict) -> None:
    print("\n" + "─" * 55)
    print("  SFVS — Structure Factor Validation Score")
    print("─" * 55)
    print(f"  Populations  N0={info['N0']:,}  N1={info['N1']:,}  "
          f"noise={info['N_noise']:,}")
    print(f"  Fractions    f0={info['f0']:.4f}  f1={info['f1']:.4f}")
    print(f"\n  Spectral weights")
    print(f"    LFTS cluster:  I_T={info['I0_T']:.4f}  I_D={info['I0_D']:.4f}"
          f"  →  C0={info['C0']:.4f}")
    print(f"    DNLS cluster:  I_T={info['I1_T']:.4f}  I_D={info['I1_D']:.4f}"
          f"  →  C1={info['C1']:.4f}")
    print(f"  Spectral contrast  𝒞 = {info['C']:.4f}")
    print(f"\n  ζ misassignment")
    print(f"    mis0 (LFTS with DNLS-ζ) = {info['mis0']:.4f}")
    print(f"    mis1 (DNLS with LFTS-ζ) = {info['mis1']:.4f}")
    print(f"  ζ-penalty  Λ = {info['Lambda']:.4f}")
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  SFVS = {info['sfvs']:+.4f}  "
          f"({_quality(info['sfvs'])}){'':10}│")
    print(f"  └─────────────────────────────────────┘")
    print("─" * 55)


def _quality(sfvs: float) -> str:
    if np.isnan(sfvs):
        return "undefined"
    if sfvs > 0.7:
        return "Excellent"
    if sfvs > 0.4:
        return "Good"
    if sfvs > 0.1:
        return "Weak"
    return "Failed"


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Variant A — 3D Volume Score  compute_sfvs_3d
# ─────────────────────────────────────────────────────────────────────────────

def volume_integral_2d(S_k_zeta: np.ndarray,
                        k_values: np.ndarray,
                        zeta_centers: np.ndarray,
                        k_norm_lo: float, k_norm_hi: float,
                        zeta_lo: float,   zeta_hi: float,
                        r_oo: float = R_OO_NM) -> float:
    """
    Integrate S(k, ζ) over a rectangular (k, ζ) box.

        V = Σ_{k∈[k_lo,k_hi]} Σ_{ζ∈[ζ_lo,ζ_hi]} S(k,ζ) · Δk · Δζ

    NaN bins are treated as zero (excluded from the sum).

    Parameters
    ----------
    S_k_zeta     : 2D array (n_zeta, n_k) — S(k,ζ) surface
    k_values     : 1D array (n_k,)  in nm⁻¹
    zeta_centers : 1D array (n_zeta,) in Å
    k_norm_lo/hi : normalised k·r_OO/2π window boundaries
    zeta_lo/hi   : ζ window boundaries in Å
    r_oo         : O–O distance in nm (default 0.285)

    Returns
    -------
    float — integrated volume (0.0 if no grid points fall in the box)
    """
    k_norm = k_values * r_oo / (2.0 * np.pi)
    k_mask = (k_norm >= k_norm_lo) & (k_norm <= k_norm_hi)
    z_mask = (zeta_centers >= zeta_lo) & (zeta_centers <= zeta_hi)

    if not np.any(k_mask) or not np.any(z_mask):
        return 0.0

    dk = float(k_values[1] - k_values[0]) if len(k_values) > 1 else 1.0
    dz = float(zeta_centers[1] - zeta_centers[0]) if len(zeta_centers) > 1 else 1.0

    patch = S_k_zeta[np.ix_(z_mask, k_mask)]   # (n_z_in, n_k_in)
    patch = np.where(np.isnan(patch), 0.0, patch)
    return float(np.sum(patch) * dk * dz)


def compute_sfvs_3d(S0_k_zeta: np.ndarray,
                    S1_k_zeta: np.ndarray,
                    k_values: np.ndarray,
                    zeta_centers: np.ndarray,
                    labels: np.ndarray,
                    r_oo: float = R_OO_NM,
                    verbose: bool = False) -> Tuple[float, dict]:
    """
    Compute SFVS using the 3D volume approach (Variant A).

    For each cluster, compute the integral of S(k,ζ) over its *correct* box
    and its *wrong* box, then score with Michelson contrast.

      Cluster 0 (LFTS):
        V₀_lfts = ∬_{LFTS box} S₀(k,ζ) dk dζ   ← correct
        V₀_dnls = ∬_{DNLS box} S₀(k,ζ) dk dζ   ← penalty
        C₀ = (V₀_lfts − V₀_dnls) / (V₀_lfts + V₀_dnls)

      Cluster 1 (DNLS):
        V₁_dnls = ∬_{DNLS box} S₁(k,ζ) dk dζ   ← correct
        V₁_lfts = ∬_{LFTS box} S₁(k,ζ) dk dζ   ← penalty
        C₁ = (V₁_dnls − V₁_lfts) / (V₁_dnls + V₁_lfts)

      SFVS_3D = f₀·C₀ + f₁·C₁

    Parameters
    ----------
    S0_k_zeta    : S(k,ζ) for cluster 0  shape (n_zeta, n_k)
    S1_k_zeta    : S(k,ζ) for cluster 1  shape (n_zeta, n_k)
    k_values     : nm⁻¹,  shape (n_k,)
    zeta_centers : Å,     shape (n_zeta,)
    labels       : per-molecule cluster labels (−1 / 0 / 1)
    r_oo         : O–O reference distance in nm
    verbose      : print breakdown table

    Returns
    -------
    sfvs  : float
    info  : dict with all intermediate volumes and contrasts
    """
    # Population fractions (noise excluded)
    N0 = int((labels == 0).sum())
    N1 = int((labels == 1).sum())
    N_clean = N0 + N1
    if N0 == 0 or N1 == 0 or N_clean == 0:
        warnings.warn("SFVS_3D undefined: one or both clusters are empty.")
        return np.nan, {}

    f0 = N0 / N_clean
    f1 = N1 / N_clean

    def _vol(S, k_lo, k_hi, z_lo, z_hi):
        return volume_integral_2d(S, k_values, zeta_centers,
                                  k_lo, k_hi, z_lo, z_hi, r_oo)

    # Cluster 0 volumes
    V0_lfts = _vol(S0_k_zeta, LFTS_K_LO, LFTS_K_HI, LFTS_Z_LO, LFTS_Z_HI)
    V0_dnls = _vol(S0_k_zeta, DNLS_K_LO, DNLS_K_HI, DNLS_Z_LO, DNLS_Z_HI)

    # Cluster 1 volumes
    V1_lfts = _vol(S1_k_zeta, LFTS_K_LO, LFTS_K_HI, LFTS_Z_LO, LFTS_Z_HI)
    V1_dnls = _vol(S1_k_zeta, DNLS_K_LO, DNLS_K_HI, DNLS_Z_LO, DNLS_Z_HI)

    # Michelson contrasts
    denom0 = V0_lfts + V0_dnls
    denom1 = V1_dnls + V1_lfts
    C0 = (V0_lfts - V0_dnls) / denom0 if denom0 > 0 else 0.0
    C1 = (V1_dnls - V1_lfts) / denom1 if denom1 > 0 else 0.0

    sfvs = f0 * C0 + f1 * C1

    info = dict(
        N0=N0, N1=N1, N_noise=int((labels == -1).sum()),
        f0=f0,  f1=f1,
        V0_lfts=V0_lfts, V0_dnls=V0_dnls,
        V1_lfts=V1_lfts, V1_dnls=V1_dnls,
        C0=C0,  C1=C1,  sfvs=sfvs,
    )

    if verbose:
        _print_report_3d(info)

    return sfvs, info


def _print_report_3d(info: dict) -> None:
    sfvs = info.get("sfvs", float("nan"))
    print("\n" + "─" * 60)
    print("  SFVS-3D — Volume Score")
    print("─" * 60)
    print(f"  Populations  N0={info['N0']:,}  N1={info['N1']:,}  "
          f"noise={info['N_noise']:,}")
    print(f"  Fractions    f0={info['f0']:.4f}  f1={info['f1']:.4f}")
    print()
    print(f"  Windows:")
    print(f"    LFTS box : k·r_OO/2π ∈ [{LFTS_K_LO},{LFTS_K_HI}]"
          f"  ζ ∈ [{LFTS_Z_LO},{LFTS_Z_HI}] Å")
    print(f"    DNLS box : k·r_OO/2π ∈ [{DNLS_K_LO},{DNLS_K_HI}]"
          f"  ζ ∈ [{DNLS_Z_LO},{DNLS_Z_HI}] Å")
    print()
    print(f"  Cluster 0 (LFTS)")
    print(f"    V_correct (LFTS box) = {info['V0_lfts']:.6f}")
    print(f"    V_penalty (DNLS box) = {info['V0_dnls']:.6f}")
    print(f"    C0 = {info['C0']:+.4f}")
    print()
    print(f"  Cluster 1 (DNLS)")
    print(f"    V_correct (DNLS box) = {info['V1_dnls']:.6f}")
    print(f"    V_penalty (LFTS box) = {info['V1_lfts']:.6f}")
    print(f"    C1 = {info['C1']:+.4f}")
    print()
    print(f"  ┌──────────────────────────────────────────┐")
    print(f"  │  SFVS-3D = {sfvs:+.4f}  ({_quality(sfvs)}){'':12}│")
    print(f"  └──────────────────────────────────────────┘")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Convenience: score a list of (method, labels) results
# ─────────────────────────────────────────────────────────────────────────────

def rank_methods(method_results: list,
                 S_k_func,
                 k_values: np.ndarray,
                 zeta: np.ndarray,
                 r_oo: float = R_OO_NM) -> "pd.DataFrame":
    """
    Rank multiple clustering results by SFVS.

    Parameters
    ----------
    method_results : list of (method_name: str, labels: np.ndarray)
    S_k_func       : callable(labels, cluster_id) → S_k array
                     (wraps compute_per_cluster_structure_factor)
    k_values       : wave-vector magnitudes (nm⁻¹)
    zeta           : ζ order parameter per molecule
    r_oo           : O–O reference distance (nm)

    Returns
    -------
    pd.DataFrame sorted by SFVS descending
    """
    import pandas as pd
    rows = []
    for name, labels in method_results:
        S0 = S_k_func(labels, 0)
        S1 = S_k_func(labels, 1)
        sfvs, info = compute_sfvs(S0, S1, k_values, zeta, labels, r_oo)
        rows.append({
            "method"  : name,
            "sfvs"    : sfvs,
            "quality" : _quality(sfvs),
            "C"       : info.get("C",      np.nan),
            "Lambda"  : info.get("Lambda", np.nan),
            "C0"      : info.get("C0",     np.nan),
            "C1"      : info.get("C1",     np.nan),
            "mis0"    : info.get("mis0",   np.nan),
            "mis1"    : info.get("mis1",   np.nan),
            "f0"      : info.get("f0",     np.nan),
            "f1"      : info.get("f1",     np.nan),
        })
    return pd.DataFrame(rows).sort_values("sfvs", ascending=False).reset_index(drop=True)
