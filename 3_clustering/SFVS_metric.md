# Structure Factor Validation Score (SFVS)

A quantitative metric for evaluating whether unsupervised ML clustering of water molecules recovers physically distinct structural states, validated against independent reciprocal-space observables.

---

## 1. Motivation

After clustering water molecules into two populations (LFTS-like and DNLS-like), we need a **non-circular, quantitative** measure of whether the clusters correspond to physically distinct structural states. Visual inspection of S(k) curves is subjective. The SFVS provides a single scalar score that:

- Rewards clusters whose S(k) peaks at the theoretically expected wavenumber
- Penalizes clusters that show spectral weight where they shouldn't
- Incorporates an independent ζ-based consistency check
- Enables systematic comparison across methods, feature sets, temperatures, and water models

---

## 2. Definitions and Setup

### 2.1 Cluster Populations

After clustering, each molecule is assigned to one of two clusters:

- **Cluster 0 (LFTS-like):** expected to exhibit tetrahedral ordering
- **Cluster 1 (DNLS-like):** expected to exhibit disordered liquid structure
- **Noise:** molecules not assigned to either cluster (excluded from scoring)

Let $N_0$ and $N_1$ be the number of molecules in each cluster, and define the population fractions:

$$f_0 = \frac{N_0}{N_0 + N_1}, \qquad f_1 = \frac{N_1}{N_0 + N_1}$$

### 2.2 Per-Cluster Structure Factor

For each cluster $C_\alpha$, the oxygen-oxygen partial structure factor is computed via the Debye scattering equation restricted to molecules in that cluster:

$$S_\alpha(k) = 1 + \frac{1}{N_\alpha} \sum_{i \in C_\alpha} \sum_{\substack{j \in C_\alpha \\ j \neq i}} \frac{\sin(k r_{ij})}{k r_{ij}} W(r_{ij})$$

where $r_{ij}$ is the O–O distance and $W(r_{ij}) = \frac{\sin(\pi r_{ij}/r_c)}{\pi r_{ij}/r_c}$ is the window function with cutoff $r_c = 1.5$ nm.

### 2.3 Theoretical Peak Positions

From Shi & Tanaka (JACS 2020), the two structural states produce distinct signatures in S(k):

| State | Peak Label | Position ($k r_\text{OO}/2\pi$) | Physical Origin |
|-------|-----------|-------------------------------|-----------------|
| LFTS  | $k_{T1}$ (FSDP) | $\approx 3/4 = 0.75$ | Height of tetrahedral unit |
| DNLS  | $k_{D1}$        | $\approx 1.0$        | Nearest-neighbor O–O distance (ordinary liquid) |

---

## 3. Spectral Contrast Score

### 3.1 Diagnostic Wavenumber Windows

Define two integration windows centered on the expected peak positions:

$$W_T = \left\{ k : \frac{k r_\text{OO}}{2\pi} \in [0.65, \; 0.85] \right\} \qquad \text{(tetrahedral window)}$$

$$W_D = \left\{ k : \frac{k r_\text{OO}}{2\pi} \in [0.90, \; 1.10] \right\} \qquad \text{(disordered window)}$$

These windows are chosen to be symmetric about the expected peak centers with width $\Delta = 0.20$, wide enough to capture the peak but narrow enough to avoid overlap. The gap between windows ($0.85$ to $0.90$) prevents double-counting.

### 3.2 Integrated Spectral Weight

For cluster $\alpha$ and window $W$, compute the integrated spectral weight:

$$I_\alpha^W = \int_W S_\alpha(k) \, dk$$

In practice, with discrete $k$-points spaced by $\Delta k$:

$$I_\alpha^W = \sum_{k_i \in W} S_\alpha(k_i) \, \Delta k$$

This yields four quantities:

| Quantity | Cluster | Window | Expected |
|----------|---------|--------|----------|
| $I_0^T$ | 0 (LFTS) | $W_T$ | **High** (correct peak) |
| $I_0^D$ | 0 (LFTS) | $W_D$ | **Low** (wrong peak) |
| $I_1^T$ | 1 (DNLS) | $W_T$ | **Low** (wrong peak) |
| $I_1^D$ | 1 (DNLS) | $W_D$ | **High** (correct peak) |

### 3.3 Per-Cluster Contrast

The **Michelson contrast** for each cluster measures how strongly the correct peak dominates over the wrong one:

$$C_0 = \frac{I_0^T - I_0^D}{I_0^T + I_0^D}$$

$$C_1 = \frac{I_1^D - I_1^T}{I_1^D + I_1^T}$$

Each score ranges from $-1$ to $+1$:

| Value | Interpretation |
|-------|---------------|
| $+1$  | All spectral weight in the correct window |
| $0$   | Equal weight in both windows (no discrimination) |
| $-1$  | All spectral weight in the wrong window (inverted labels) |

### 3.4 Population-Weighted Spectral Contrast

Combine both clusters, weighted by their population fractions:

$$\mathcal{C} = f_0 \cdot C_0 + f_1 \cdot C_1$$

This ensures that a small minority cluster does not disproportionately influence the overall score.

---

## 4. ζ-Based Penalty

### 4.1 Motivation

The spectral contrast score operates in reciprocal space. To add a complementary real-space consistency check, we examine the ζ distribution within each cluster. If the clustering is correct:

- LFTS molecules (Cluster 0) should predominantly have **high ζ** (well-separated shells)
- DNLS molecules (Cluster 1) should predominantly have **low or negative ζ** (interpenetrating shells)

### 4.2 ζ Diagnostic Windows

Define two ζ ranges corresponding to the expected structural states:

$$Z_T = \{ \zeta \in [0.5, \; 1.5] \} \qquad \text{(LFTS-expected range)}$$

$$Z_D = \{ \zeta \in [-1.0, \; 0.0] \} \qquad \text{(DNLS-expected range)}$$

These ranges are chosen based on the bimodal ζ distribution observed in Tanaka's work, where the two Gaussian components are centered at approximately $\zeta \approx 0.8$ (LFTS) and $\zeta \approx -0.2$ (DNLS).

### 4.3 Misassignment Fractions

Count the fraction of molecules in each cluster that fall in the "wrong" ζ range:

$$\text{mis}_0 = \frac{|\{i \in C_0 : \zeta_i \in Z_D\}|}{|C_0|}$$

$$\text{mis}_1 = \frac{|\{i \in C_1 : \zeta_i \in Z_T\}|}{|C_1|}$$

- $\text{mis}_0$: fraction of supposed-LFTS molecules with DNLS-like ζ values
- $\text{mis}_1$: fraction of supposed-DNLS molecules with LFTS-like ζ values

### 4.4 ζ-Penalty Factor

$$\Lambda = 1 - \left( f_0 \cdot \text{mis}_0 + f_1 \cdot \text{mis}_1 \right)$$

| Value | Interpretation |
|-------|---------------|
| $1.0$ | No molecules in the wrong ζ range (perfect consistency) |
| $0.5$ | Significant cross-contamination |
| $0.0$ | All molecules in the wrong ζ range (complete mismatch) |

---

## 5. Combined SFVS

The final **Structure Factor Validation Score** is the product of the spectral contrast and the ζ-penalty:

$$\boxed{\text{SFVS} = \mathcal{C} \times \Lambda = \left( f_0 \cdot C_0 + f_1 \cdot C_1 \right) \times \left( 1 - f_0 \cdot \text{mis}_0 - f_1 \cdot \text{mis}_1 \right)}$$

The multiplicative form ensures that **both** reciprocal-space and real-space validation must be satisfied simultaneously. A high spectral contrast with poor ζ consistency (or vice versa) will produce a mediocre score.

### 5.1 Score Range and Interpretation

| SFVS | Quality | Description |
|------|---------|-------------|
| $> 0.7$ | **Excellent** | Strong two-state separation confirmed in both S(k) and ζ |
| $0.4 - 0.7$ | **Good** | Clear separation with some overlap in transition region |
| $0.1 - 0.4$ | **Weak** | Marginal separation; clusters partially overlap |
| $< 0.1$ | **Failed** | Clustering does not recover physically distinct states |

### 5.2 Properties

- **Non-circular:** S(k) is computed from atomic coordinates; clustering uses order parameters. ζ-penalty uses a descriptor but evaluates distribution overlap, not cluster definition.
- **Bounded:** SFVS ∈ [−1, +1], with negative values indicating inverted labels.
- **Symmetric under label swap:** If labels are swapped, $C_0 \leftrightarrow -C_1$ and $\text{mis}_0 \leftrightarrow \text{mis}_1$, yielding SFVS → −SFVS. Negative scores simply mean the labels should be flipped.
- **Population-weighted:** Robust when cluster sizes are unequal (e.g., at temperatures far from $T_{s=1/2}$).

---

## 6. Worked Example

Suppose at TIP4P/2005, T = −35°C, using UMAP + HDBSCAN + GMM:

**Cluster populations:** $N_0 = 11{,}000$, $N_1 = 8{,}500$, noise $= 970$

$$f_0 = \frac{11{,}000}{19{,}500} = 0.564, \qquad f_1 = \frac{8{,}500}{19{,}500} = 0.436$$

**Integrated spectral weights** (arbitrary units):

| | $W_T$ | $W_D$ |
|---|---|---|
| Cluster 0 (LFTS) | $I_0^T = 0.42$ | $I_0^D = 0.18$ |
| Cluster 1 (DNLS) | $I_1^T = 0.15$ | $I_1^D = 0.38$ |

**Spectral contrasts:**

$$C_0 = \frac{0.42 - 0.18}{0.42 + 0.18} = \frac{0.24}{0.60} = 0.400$$

$$C_1 = \frac{0.38 - 0.15}{0.38 + 0.15} = \frac{0.23}{0.53} = 0.434$$

**Population-weighted spectral contrast:**

$$\mathcal{C} = 0.564 \times 0.400 + 0.436 \times 0.434 = 0.226 + 0.189 = 0.415$$

**ζ misassignment:**

$$\text{mis}_0 = \frac{820}{11{,}000} = 0.075, \qquad \text{mis}_1 = \frac{510}{8{,}500} = 0.060$$

**ζ-penalty:**

$$\Lambda = 1 - (0.564 \times 0.075 + 0.436 \times 0.060) = 1 - (0.042 + 0.026) = 0.932$$

**Final SFVS:**

$$\text{SFVS} = 0.415 \times 0.932 = 0.387$$

**Interpretation:** Good separation — the clusters show clear spectral differentiation with low ζ cross-contamination, though there is moderate overlap in the S(k) windows, which is physically expected for supercooled (not deeply supercooled) water where the two populations have similar abundance.

---

## 7. Applications

The SFVS enables quantitative comparisons across:

| Comparison | Use Case |
|------------|----------|
| **Clustering methods** | Rank K-Means vs. GMM vs. HDBSCAN+GMM by SFVS |
| **Feature subsets** | Identify which order parameter combinations maximize SFVS |
| **Temperatures** | Track how separation quality changes from 0°C to −30°C |
| **Water models** | Compare TIP4P/2005 vs. TIP5P vs. SWM4-NDP clustering quality |
| **UMAP parameters** | Optimize n_neighbors and min_dist by maximizing SFVS |

---

## 8. Caveats and Considerations

1. **Window choice sensitivity:** The boundaries of $W_T$, $W_D$, $Z_T$, and $Z_D$ are guided by Tanaka's results but involve some discretion. A sensitivity analysis varying window widths by ±0.05 is recommended.

2. **ζ circularity concern:** If ζ is included as a clustering feature, the ζ-penalty introduces partial circularity. In this case, report the spectral contrast $\mathcal{C}$ alone as the primary non-circular metric, and the full SFVS as a supplementary consistency check. Alternatively, run the SFVS on feature subsets that exclude ζ.

3. **Baseline comparison:** Compute the SFVS for randomly shuffled cluster labels to establish a null distribution. A meaningful clustering should produce SFVS significantly above this baseline (expected $\mathcal{C} \approx 0$ for random labels).

4. **Temperature dependence:** At temperatures far above $T_{s=1/2}$, the LFTS population is small and the FSDP is weak, so even correct clustering will yield lower SFVS. Interpret scores relative to the expected signal strength at each temperature.

5. **Normalization of S(k):** Ensure consistent normalization of $S_\alpha(k)$ across clusters. The Debye equation normalization (dividing by $N_\alpha$) already accounts for different cluster sizes.
