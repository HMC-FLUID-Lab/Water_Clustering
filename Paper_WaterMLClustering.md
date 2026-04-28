# Is Pure Water a Mixture: Perspectives from Machine-Learning Classification Methods

**Yiqi Yao,¹ Diya Sanghi,¹ Akanksha D. Chokshi,² Minwoo Choi,² Haiqin Wang,¹ and Bilin Zhuang¹˒²˒\***

¹Department of Chemistry, Harvey Mudd College, 301 Platt Boulevard, Claremont, California 91711  
²Yale-NUS College, Singapore 138527, Singapore

*(Dated: March 9, 2026)*

\* bzhuang@hmc.edu

---

By unraveling the mysteries of water's anomalies, we not only deepen our fundamental knowledge of this vital substance but also pave the way for potential applications in fields ranging from materials science to biological systems, where water plays a pivotal role.

## I. Introduction

The local structure of liquid water has been a subject of sustained scientific debate for over a century. Two fundamentally different pictures have competed to explain water's thermodynamic and dynamic anomalies – its density maximum at 4°C, the rapid increase in isothermal compressibility upon supercooling, and the fragile-to-strong dynamic transition[1]. Continuum models describe water through a broad unimodal distribution of local environments[2,3]. Mixture models, dating back to Röntgen's 1892 hypothesis of coexisting "icebergs" and a fluid "sea"[4], instead posit two or more distinct structural populations whose relative fractions shift with temperature and pressure. Despite a clear difference in physical picture, the lack of unambiguous experimental evidence has prevented convergence of this debate[2,3,5].

Recent computational work by Shi and Tanaka has provided substantial support for a two-state description of liquid water[6–8]. In their framework, water is treated as a dynamic mixture of two local structural motifs: locally favored tetrahedral structures (LFTS), stabilized by four hydrogen bonds with low density and high local symmetry, and disordered normal-liquid structures (DNLS), characterized by broken tetrahedral symmetry, higher density, and greater entropy. Using the microscopic structural descriptor ζ—which quantifies the translational order in the second coordination shell[7]—they demonstrated that the distribution P(ζ) is bimodal in several classical water models (TIP4P/2005, TIP5P, ST2), with each component well described by a Gaussian corresponding to one of the two states. Crucially, they showed that the oxygen–oxygen partial structure factor S(k) contains two overlapping peaks hidden within the apparent first diffraction peak: a first sharp diffraction peak (FSDP) at k_T1 = kr_OO/2π ≈ 3/4, arising from the tetrahedral geometry of LFTS, and an ordinary liquid peak at k_D1 = kr_OO/2π ≈ 1, characteristic of DNLS[6,9].

However, a methodological limitation persists in prior work: the same structural descriptor ζ that is used to define and classify the two populations is also used to validate the classification through the ζ-resolved structure factor S(k, ζ). This circularity—where the input to and output of the analysis share a common variable—leaves open the possibility that the observed two-state signatures are artifacts of the descriptor construction rather than intrinsic features of the liquid structure. An independent classification method that does not rely on ζ-based thresholds to assign structural states would provide a more rigorous test of the two-state hypothesis.

Unsupervised machine learning offers a natural route to address this limitation. Clustering algorithms can identify structure in high-dimensional data without being provided predefined labels or classification rules, making them well suited for discovering latent structural populations in molecular simulation data. In this work, we develop a systematic unsupervised clustering pipeline that operates on a five-dimensional order-parameter feature vector (q, Q₆, LSI, Sₖ, ζ) computed from molecular dynamics trajectories of TIP4P/2005 and TIP5P water. We benchmark four clustering strategies—K-Means, Gaussian Mixture Models (GMM), and two hybrid pipelines combining density-based denoising (DBSCAN, HDBSCAN) with GMM—and validate the resulting cluster assignments through an independent physical observable: the per-cluster oxygen–oxygen partial structure factor S(k), computed via the Debye scattering equation. Because the clustering is performed in the order-parameter space while the validation relies on a reciprocal-space quantity derived directly from atomic coordinates, the two stages share no common descriptor, ensuring a non-circular assessment of cluster quality.

## II. Method

### A. Molecular Dynamics Simulations

Classical molecular dynamics simulations are carried out for a system of N = 1024 water molecules in the canonical (NVT) ensemble at ambient pressure using the OpenMM simulation package[10]. Two rigid, non-polarizable water models are employed: TIP4P/2005[11] and TIP5P[12]. We select these models because Shi and Tanaka[6] demonstrated that they produce strong tetrahedral ordering signatures in the structure factor, with well-separated locally favored tetrahedral structures (LFTS) and disordered normal-liquid structures (DNLS). Earlier exploratory work with the polarizable SWM4-NDP model showed substantially weaker structural bimodality, consistent with the expectation that flexible charge distributions reduce the sharpness of tetrahedral ordering, which will be further explored and compared.

For each water model, simulations are conducted at seven temperatures ranging from 0°C to −30°C in 5°C increments, spanning conditions where the LFTS population grows from a minority to a substantial fraction of all molecules. This temperature range includes the vicinity of the Schottky temperature T_{s=1/2}, at which the two structural populations are approximately equal (T_{s=1/2} ≈ 237.8 K for TIP4P/2005 and T_{s=1/2} ≈ 255.5 K for TIP5P)[6]. At each condition, we perform 20 independent NVT production runs following equilibration, collecting trajectory snapshots at regular intervals. The resulting data set comprises approximately 20,480 molecular configurations per temperature per water model. Trajectories and topology files are stored in .dcd and .pdb formats.

### B. Structural Order Parameter

For each trajectory frame, five structural order parameters are computed using the local oxygen-oxygen coordination environment of each molecule. These parameters, previously identified in the literature as sensitive probes of water's local structure (q, Q₆, LSI, Sₖ, ζ), serve as the feature vector for our clustering analysis. All parameters are calculated using the MDTraj library[13] for neighbor identification and custom implementations following the definitions below.

#### 1. Orientational Order Parameter, q

The orientational order parameter, q, measures the extent to which a molecule and its four neighbours form a tetrahedral arrangement[14,15]. It computes the angle ψ_{jk} between the oxygen atom of the designated water molecule and its nearest neighbour oxygen atoms indicated as j and k below:

$$q = 1 - \frac{3}{8} \sum_{j=1}^{3} \sum_{k=j+1}^{4} \left( \cos\psi_{jk} + \frac{1}{3} \right)^2$$

The value of q ranges from −3 to 1, where q = 1 represents a perfect tetrahedron.

#### 2. Bond-Orientational Order Parameter, Q₆

The Steinhardt bond-orientational order parameter[16] quantifies the degree of crystalline-like order in the arrangement of a molecule's 12 nearest neighbors by averaging the l = 6 spherical harmonics of the bond vectors:

$$Q_6 = \left( \frac{4\pi}{13} \sum_{m=-6}^{6} \left| \bar{Y}_{6m} \right|^2 \right)^{1/2}$$

where Ȳ₆ₘ is the average of Y₆ₘ(θ, ϕ) over all bonds to the 12 nearest neighbors. Higher Q₆ values indicate a higher long-range orientation order.

#### 3. Local Structure Index, LSI

The Local Structure Index quantifies the gap between the first and second hydration shells surrounding each molecule[17]:

$$\text{LSI} = \frac{1}{n} \sum_{i=1}^{n} \left( \Delta_i - \bar{\Delta} \right)^2$$

where Δᵢ = rᵢ₊₁ − rᵢ is the distance gap between consecutively ordered oxygen neighbors (r₁ < r₂ < ⋯ < rₙ) within a cutoff of 3.7 Å, and Δ̄ is the mean gap. A large LSI indicates a well-defined separation between shells, characteristic of tetrahedral ordering; a small LSI indicates a more uniform radial distribution of neighbors.

#### 4. Translational Tetrahedral Parameter, Sₖ

This parameter is much like the orientational order parameter, except instead of measuring the angle between the oxygen atom and its four nearest neighbors, it measures the change in radial distance between the designated oxygen atom and its four nearest neighboring oxygen atoms. It is algebraically defined as:

$$S_k = 1 - \frac{1}{3} \sum_{k=1}^{4} \frac{(r_k - \bar{r})^2}{4\bar{r}^2}$$

where rₖ is the radial distance to the nearest neighbor kᵗʰ and r̄ is the average of the four radial distances. Sₖ also has a maximum value of 1 which is seen when a tetrahedron is achieved[18].

#### 5. Local Translational Order Parameter, ζ

The ζ parameter, introduced by Russo and Tanaka[7], characterizes translational order in the second coordination shell:

$$\zeta = r_{\text{nearest non-HB}} - r_{\text{farthest HB}}$$

where r_{farthest HB} is the distance from the central molecule to its most distant hydrogen-bonded neighbor and r_{nearest non-HB} is the distance to the closest non-hydrogen-bonded neighbor. A large positive ζ indicates a clear separation between bonded and non-bonded shells (LFTS), while ζ ≈ 0 or negative indicates shell interpenetration (DNLS). This parameter serves as the primary discriminator in Tanaka's two-state framework[7,8].

All five order parameters are computed for each of the 1024 molecules in every trajectory frame and stored in .mat format. The resulting feature matrix has dimensions N_samples × 5, where N_samples = N_frames × N_molecules × N_runs.

### C. Clustering Algorithm

Six unsupervised clustering methods are applied to classify each water molecule into a structural state based on its five-dimensional order-parameter feature vector. All features are standardized to zero mean and unit variance prior to clustering to ensure equal weighting across parameters with different natural scales. The algorithms we employ fall into three categories: partition-based, density-based, and distribution-based, as well as hybrid combinations thereof[19].

#### 1. Partition-based and distribution-based methods

K-Means clustering[20] serves as a hard-assignment baseline. The algorithm partitions the standardized feature space into k disjoint regions by iteratively assigning each molecule to its nearest centroid in the five-dimensional order-parameter space and recomputing cluster centroids until convergence. Centroids are initialized using the KMeans++ scheme to mitigate sensitivity to initial placement, with n_init = 20 independent restarts and a maximum of 300 iterations per run. The number of clusters is fixed at k = 2, corresponding to the two-state hypothesis.

The Gaussian Mixture Model (GMM)[21] provides a probabilistic alternative that is physically well-motivated: Tanaka's two-state framework predicts that the order-parameter distributions of LFTS and DNLS are each approximately Gaussian, making a two-component mixture a natural model[6]. We fit GMMs with k = 2 components using the expectation-maximization algorithm[22] with full covariance matrices, and we evaluate models with k = 2, 3, 4 components using the Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC) to test for the presence of additional structural states beyond the two-state model.

#### 2. Density-based methods

Density-Based Spatial Clustering of Applications with Noise (DBSCAN)[23] identifies clusters as contiguous regions of high point density in feature space, classifying molecules in low-density regions as noise. The algorithm requires two parameters: the neighborhood radius ε (Eps) and the minimum number of points MinPts. Because Eps and MinPts are not known a priori, a systematic parameter sweep is performed, with cluster quality assessed at each combination using the silhouette score[24].

Hierarchical DBSCAN extends DBSCAN by constructing a cluster hierarchy across all density scales and extracting the most persistent clusters, eliminating the need to specify ε[25]. The primary parameter is min_cluster_size, which is set to approximately 1% of the total sample size to ensure that only statistically significant populations are identified as clusters.

#### 3. Main Method, Hybrid methods: DBSCAN+GMM and HDBSCAN+GMM

Two-stage hybrid pipelines are constructed by combining density-based denoising with distribution-based classification. In the first stage, DBSCAN or HDBSCAN is applied not as a primary clustering method but solely as a denoising step: molecules residing in low-density transition regions between the two structural states are identified and removed. While density-based methods alone often fail to achieve clean separation between the two populations, their capacity to isolate ambiguous boundary points proves effective as a preprocessing operation, which provides a cleaner subset for further clustering separation. In the second stage, GMM is fitted to the denoised subset, yielding probabilistic cluster assignments with markedly higher silhouette scores than those obtained without prior noise removal. This hybrid approach leverages the complementary strengths of both paradigms—the noise-identification capability of density-based algorithms and the physically motivated soft clustering of GMM.

#### 4. Feature ablation

To determine which order parameters are most informative for structural classification, we perform systematic feature ablation studies. We run the clustering pipeline on all C(5,k) subsets of features for k = 1, …, 5 and evaluate each combination using the silhouette score and, independently, the quality of the resulting structure-factor validation (Sec. III B). This analysis reveals the minimal feature set required for robust two-state identification and quantifies the contribution of each parameter to cluster separation.

### D. Non-Circular Validation via Structure Factor Analysis

The central methodological contribution of this work is a validation strategy that avoids the circular reasoning inherent in prior studies, where the same structural descriptor ζ is used both to define clusters and to validate them. In our approach, cluster labels assigned by the unsupervised learning pipeline are mapped back onto the original molecular dynamics trajectories, and the local structural characteristic in the wavenumber space, S(k), is computed independently for each labeled subpopulation using the Debye scattering equation[26]. Because the clustering is performed in the five-dimensional order-parameter space while the validation relies on a reciprocal-space observable derived directly from atomic coordinates, the two stages share no common descriptor. If the machine-learning-assigned clusters correspond to physically distinct structural states, their S(k) curves should exhibit the characteristic peak signatures identified by Shi and Tanaka[6]: a first sharp diffraction peak (FSDP) at k_T1 = kr_OO/2π ≈ 3/4 for the LFTS-like cluster, and an ordinary liquid peak at k_D1 = kr_OO/2π ≈ 1 for the DNLS-like cluster. Higher-order peaks such as k_T2 and k_T3, while present in the structure factors of tetrahedral materials[9], are not considered in the present analysis.

We compute the oxygen-oxygen partial structure factor using the Debye scattering equation[26]:

$$S(k) = 1 + \frac{1}{N} \sum_{i=1}^{N} \sum_{\substack{j=1 \\ j \neq i}}^{N} \frac{\sin(kr_{ij})}{kr_{ij}} W(r_{ij})$$

where rᵢⱼ is the distance between oxygen atoms i and j, and the window function is defined as:

$$W(r_{ij}) = \frac{\sin(\pi r_{ij} / r_c)}{\pi r_{ij} / r_c}$$

with cutoff distance rₒ. The window function suppresses Gibbs ringing artifacts from the hard distance cutoff and confines the calculation to the local structural environment around each central molecule. We use rₒ = 1.5 nm, which includes approximately five coordination shells and provides a balance between local structural resolution and statistical convergence. The wavenumber axis is normalized as kr_OO/2π, where r_OO ≈ 2.8 Å is the nearest-neighbor oxygen-oxygen distance, following the convention of Shi and Tanaka. For each cluster identified by the ML pipeline, we compute S(k) by restricting the Debye summation to molecules assigned to that cluster. Specifically, for a cluster Cα, the per-cluster structure factor is:

$$S_\alpha(k) = 1 + \frac{1}{N_\alpha} \sum_{i \in C_\alpha} \sum_{\substack{j \in C_\alpha \\ j \neq i}} \frac{\sin(kr_{ij})}{kr_{ij}} W(r_{ij})$$

where Nα = |Cα| is the number of molecules in cluster α. This per-cluster decomposition allows direct comparison with the theoretical predictions for LFTS and DNLS structure factors.

## III. Results and Discussions

This work establishes a systematic unsupervised clustering pipeline capable of resolving two distinct local structural states in liquid water. Following the framework of Shi and Tanaka[6], we focus our initial algorithm benchmarking on TIP4P/2005 water at T = 253.15 K, a temperature in the vicinity of the Schottky temperature T_{s=1/2} ≈ 237.8 K where the LFTS and DNLS populations coexist in appreciable proportions, providing a stringent test of each method's discriminative power.

### A. Clustering Algorithm

The development of our clustering pipeline proceeds in two stages. We first verify that basic unsupervised methods—K-Means and Gaussian Mixture Models—can recover a physically meaningful two-state separation from the five-dimensional order-parameter space. Having established this baseline, we then extend the pipeline by incorporating density-based denoising and dimensionality reduction techniques to sharpen the cluster boundaries and remove ambiguous transition-state molecules, yielding progressively cleaner structural classifications.

#### 1. Baseline: K-Means and GMM clustering

We first apply K-Means (k = 2) and GMM (k = 2, full covariance) to the standardized five-dimensional order-parameter vectors of TIP4P/2005 water at T = 253.15 K.

The dataset comprises N = 20,480 molecular configurations (1024 molecules, 20 frames). K-Means partitions the data into Cluster 0 (11,807 molecules, 57.7%) and Cluster 1 (8,673 molecules, 42.3%). Based on the mean ζ values of each cluster, we identify Cluster 0 as the LFTS population and Cluster 1 as DNLS. The per-cluster distributions of all five order parameters are shown in Fig. 1. The ζ distribution exhibits a clear bimodal structure, with well-resolved peaks near ζ ≈ 0.02 Å (DNLS) and ζ ≈ 0.06 Å (LFTS), consistent with the two-Gaussian decomposition reported by Shi and Tanaka[6]. The orientational order parameter q also shows appreciable separation, with the LFTS cluster enriched at q ≳ 0.8, reflecting enhanced tetrahedral coordination. In contrast, Q₆, LSI, and Sₖ display substantial overlap between the two clusters.

**Fig. 1.** K-Means distributions of the five structural order parameters (q, Q₆, LSI, Sₖ, ζ)

The pairwise correlations between scaled order parameters (Fig. 2) further elucidate the multivariate structure exploited by the algorithm. The q–ζ projection provides the most informative two-state diagnostic: the LFTS cluster occupies the high-q, high-ζ region, while the DNLS cluster extends toward lower values of both parameters. This correlated behavior reflects the physical coupling between tetrahedral angular order and second-shell translational order[6].

**Fig. 2.** K-Means scatter plot of the five structural order parameters (q, Q₆, LSI, Sₖ, ζ)

The silhouette score[24] for K-Means is 0.2418, slightly below the conventional moderate-separation threshold of 0.25. This is physically expected: at temperatures near T_{s=1/2}, the LFTS and DNLS populations overlap substantially due to continuous structural fluctuations, and no denoising has been applied to remove ambiguous transition-state molecules.

We repeat the analysis with a two-component GMM using full covariance matrices. GMM is a natural probabilistic counterpart to Tanaka's two-state framework, which predicts that the order-parameter distributions of LFTS and DNLS are each approximately Gaussian[6]. The resulting cluster populations (11,661 and 8,819 molecules) and per-cluster distributions (Fig. 3) are qualitatively similar to K-Means, confirming that the two-state separation is robust to the choice of baseline algorithm.

**Fig. 3.** GMM distributions of the five structural order parameters (q, Q₆, LSI, Sₖ, ζ)

The GMM silhouette score of 0.2144 is modestly lower than that of K-Means, reflecting the softer probabilistic boundaries inherent to mixture-model classification.

**Fig. 4.** GMM scatter plot of the five structural order parameters (q, Q₆, LSI, Sₖ, ζ)

Taken together, the baseline results demonstrate that even the simplest unsupervised methods recover a physically meaningful two-state partition from the five-dimensional order-parameter space, with cluster populations and ζ distributions consistent with the predictions of Tanaka's two-state model.

#### 2. Combination: DBSCAN, HDBSCAN, and GMM

In the five-dimensional order-parameter space, LFTS and DNLS molecules occupy distinct dense regions separated by a sparsely populated transition zone. DBSCAN flags molecules in this low-density boundary as noise—points with fewer than MinPts neighbors within radius ε—while HDBSCAN achieves the same goal without a fixed ε by extracting only clusters that persist across multiple density scales. Neither method reliably separates the two structural populations on its own, but both effectively isolate ambiguous transition-state molecules for removal prior to GMM classification.

**a. Parameter Selection.** The choice of density-based parameters involves a trade-off: stricter thresholds remove more boundary molecules and yield higher silhouette scores but risk discarding weakly structured yet physically genuine configurations. This trade-off is mapped in Fig. 5.

**Fig. 5.** Parameter selection heatmaps (DBSCAN, left; HDBSCAN, right)

Based on this analysis, we select ε = 0.05 and MinPts = 20 for DBSCAN, which removes approximately 18.9% of molecules as noise, and MinSamples = 7, MinCluster = 8 for HDBSCAN. The retained molecules—those not labeled as noise—are then passed to GMM for probabilistic two-state classification.

**b. Implementation.** In the hybrid pipeline, DBSCAN or HDBSCAN first assigns each molecule a preliminary label: 0 for molecules residing in dense regions and −1 for noise points in the low-density transition zone. GMM is then fitted exclusively to the retained molecules (label = 0), yielding a two-component probabilistic classification on a denoised subset.

For the DBSCAN–GMM pipeline, the resulting silhouette score of 0.2841 represents a substantial improvement over both K-Means (0.2418) and standalone GMM (0.2144), confirming that density-based denoising effectively removes ambiguous boundary molecules and sharpens the two-state separation.

**Fig. 6.** GMM distributions of the five structural order parameters (q, Q₆, LSI, Sₖ, ζ)

Approximately 22% of molecules are excluded as noise—slightly above the 18.8% estimated from the parameter sweep, a discrepancy attributable to frame-to-frame variation across the 20 independent simulation trajectories.

**Fig. 7.** DBSCAN-GMM scatter plot of the five structural order parameters (q, Q₆, LSI, Sₖ, ζ)

The analogous HDBSCAN–GMM pipeline filters out 15.2% of molecules as noise and achieves a silhouette score of 0.2572, comparable to DBSCAN–GMM. The corresponding distributions and scatter plots are provided in Appendix.

### B. Structure Factor Validation

To validate the physical meaningfulness of the ML-assigned clusters through an independent observable, we compute the oxygen–oxygen partial structure factor S(k) separately for each cluster identified by the DBSCAN–GMM pipeline, which achieved the highest silhouette score among the methods tested. Each molecule is assigned a binary cluster label (0 or 1), and the per-cluster structure factor Sα(k) is evaluated using the Debye scattering equation (Eq. 6), restricting the summation to molecules belonging to cluster Cα. Crucially, because the clustering is performed in the five-dimensional order-parameter space while the validation relies on a reciprocal-space observable derived directly from atomic coordinates, the two stages share no common descriptor, ensuring a non-circular assessment of cluster quality.

**Fig. 8.** Per-cluster S(k) for DBSCAN–GMM, TIP4P/2005 at T = 253.15 K.

The resulting per-cluster S(k) curves are shown in Fig. 8. The two clusters exhibit a clear shift in the position of the first diffraction peak. The cluster identified as LFTS (Cluster 0) displays its primary peak at k_T1 = kr_OO/2π ≈ 0.81, consistent with the first sharp diffraction peak (FSDP) characteristic of tetrahedral materials[9], while the DNLS cluster (Cluster 1) peaks at k_D1 = kr_OO/2π ≈ 1.05, matching the ordinary liquid peak position observed in simple disordered systems. These values are in close quantitative agreement with the theoretical predictions of k_T1 ≈ 3/4 and k_D1 ≈ 1 reported by Shi and Tanaka[6]. Notably, the total (unclustered) structure factor exhibits a single broad first peak whose position lies between k_T1 and k_D1—confirming that the apparent first diffraction peak of liquid water is a superposition of two overlapping contributions from structurally distinct subpopulations, as proposed in the two-state framework.

The ζ-resolved structure factor S(k, ζ) provides further confirmation. Fig. 9 shows the two-dimensional contour map of S(k, ζ), computed following the methodology of Shi and Tanaka[6]. Two distinct ridges are visible in different ζ domains: the FSDP at k_T1 ≈ 3/4 is concentrated in the high-ζ region, corresponding to molecules with well-separated bonded and non-bonded coordination shells, while the ordinary peak at k_D1 ≈ 1 dominates the low-ζ region, where shell interpenetration reflects disordered local environments. The continuous evolution of peak position with ζ and the clear localization of each peak within a distinct ζ domain are in quantitative agreement with the S(k, ζ) surfaces reported for TIP4P/2005 water at comparable temperatures[6].

**Fig. 9.** S(k, ζ) contour map, TIP4P/2005 at T = 253.15 K.

The three-dimensional S(k, ζ) surfaces separated by cluster assignment (Fig. 10) further illustrate the complementary nature of the two structural populations. The LFTS surface (Fig. 10a) exhibits a pronounced ridge at the FSDP position k_T1 with suppressed intensity near k_D1, whereas the DNLS surface (Fig. 10b) displays the inverse pattern: a dominant ridge at k_D1 and diminished scattering at k_T1. This complementary peak structure—where each cluster's S(k) maximum corresponds to the other's minimum—provides independent, non-circular confirmation that the unsupervised clustering pipeline has resolved two physically distinct structural states whose reciprocal-space signatures match the predictions of Tanaka's two-state model.

**Fig. 10.** 3D S(k, ζ) surfaces for Cluster 0 (LFTS, left) and Cluster 1 (DNLS, right).

### C. Robustness and Generality of the Two-State Classification

#### 1. Removal of ζ

To assess the contribution of individual order parameters to the clustering outcome—and to strengthen the non-circularity of the validation—we repeat the DBSCAN–GMM pipeline with ζ excluded from the feature vector, clustering on the remaining four parameters (q, Q₆, LSI, Sₖ) alone. This test is particularly important because ζ is the primary descriptor in Tanaka's two-state framework[6]; demonstrating that the pipeline recovers physically meaningful clusters without ζ would further establish the independence of the clustering from Tanaka's original classification scheme. The DBSCAN parameters are held fixed at ε = 0.05 and MinPts = 20, resulting in approximately 14% of molecules removed as noise.

**Fig. 11.** S(k, ζ) contour map without ζ in the clustering feature set, TIP4P/2005 at T = 253.15 K.

The silhouette score drops substantially to 0.1599, compared with 0.2841 for the full five-feature pipeline, indicating that ζ is the single most informative parameter for cluster separation in the standardized feature space. Despite this reduction, the ζ-resolved structure factor S(k, ζ) (Fig. 11)—computed by mapping the cluster labels back onto the ζ values that were not used during clustering—still reveals recognizable two-state signatures.

The per-cluster S(k) curves (Fig. 12) confirm that the FSDP–ordinary peak separation persists even without ζ as a clustering feature. The DNLS cluster (Cluster 1) retains a well-defined peak near k_D1 ≈ 1, consistent with the full-feature result. The LFTS cluster (Cluster 0), however, shows a weaker and broader peak near k_T1 ≈ 3/4, with its S(k) intensity partially elevated toward the k_D1 position, suggesting that the four-parameter clustering misassigns a fraction of DNLS molecules into the LFTS cluster. This contamination is consistent with the reduced silhouette score and indicates that while q, Q₆, LSI, and Sₖ collectively carry sufficient structural information to partially resolve the two states, ζ provides critical discriminative power for clean separation.

**Fig. 12.** Per-cluster S(k) without ζ in the clustering feature set, TIP4P/2005 at T = 253.15 K.

#### 2. Minimal feature set: q and ζ

*(Section to be completed)*

#### 3. Temperature dependence of cluster populations

*(Section to be completed)*

#### 4. Optimal cluster number via information criteria and silhouette analysis

*(Section to be completed)*

#### 5. Model comparison: TIP5P and SWM4-NDP

*(Section to be completed)*

## IV. Conclusion

This work demonstrates that unsupervised machine learning can independently recover two structurally distinct local populations in supercooled water, corresponding to the locally favored tetrahedral structures (LFTS) and disordered normal-liquid structures (DNLS) of Tanaka's two-state framework. A systematic comparison of clustering algorithms—K-Means, GMM, and hybrid density-GMM pipelines—applied to TIP4P/2005 water at T = 253.15 K reveals that even basic methods recover a physically meaningful two-state partition, while DBSCAN–GMM denoising substantially sharpens the cluster boundaries, raising the silhouette score from 0.21–0.24 (baseline) to 0.28 (hybrid).

The central result is the non-circular validation of the ML-assigned clusters via the oxygen–oxygen partial structure factor. The LFTS cluster exhibits a first sharp diffraction peak at k_T1 = kr_OO/2π ≈ 0.81, while the DNLS cluster peaks at k_D1 ≈ 1.05, in quantitative agreement with the theoretical predictions of Shi and Tanaka[6]. Because the clustering operates in the five-dimensional order-parameter space and the validation uses a reciprocal-space observable derived independently from atomic coordinates, this agreement constitutes model-independent confirmation that the two-state signature in the structure factor is not an artifact of the ζ-based classification scheme.

## References

1. P. Gallo, K. Amann-Winkel, C. A. Angell, M. A. Anisimov, F. Caupin, C. Chakravarty, E. Lascaris, T. Loerting, A. Z. Panagiotopoulos, J. Russo, J. A. Sellberg, H. E. Stanley, H. Tanaka, C. Vega, L. Xu, and L. G. M. Pettersson, **116**, 7463.
2. A. H. Narten and H. A. Levy, **165**, 447.
3. D. Eisenberg and W. Kauzmann, *The Structure and Properties of Water* (Oxford University Press).
4. W. C. Röntgen, *Annalen der Physik* **281**, 91 (1892).
5. P. H. Handle, T. Loerting, and F. Sciortino, **114**, 13336.
6. R. Shi and H. Tanaka, **142**, 2868.
7. J. Russo and H. Tanaka, **5**, 3556.
8. R. Shi and H. Tanaka, **148**, 124503.
9. R. Shi and H. Tanaka, *Science Advances* **5**, eaav3194 (2019).
10. P. Eastman and V. S. Pande, **12**, 34.
11. J. L. F. Abascal and C. Vega, **123**, 234505.
12. M. W. Mahoney and W. L. Jorgensen, **112**, 8910.
13. R. McGibbon, K. Beauchamp, M. Harrigan, C. Klein, J. Swails, C. Hernández, C. Schwantes, L.-P. Wang, T. Lane, and V. Pande, **109**, 1528.
14. P.-L. Chau and A. J. Hardwick, **93**, 511.
15. J. R. Errington and P. G. Debenedetti, **409**, 318.
16. P. J. Steinhardt, D. R. Nelson, and M. Ronchetti, **28**, 784.
17. E. Shiratani and M. Sasai, **104**, 7671.
18. E. Dubouë-Dijon and D. Laage, **119**, 8406.
19. S. Zheng and J. Zhao, *Computer-aided chemical engineering* (2018) pp. 2239–2244.
20. A. M. Ikotun, A. E. Ezugwu, L. Abualigah, B. Abuhaija, and J. Heming, **622**, 178.
21. M. Deprez and E. C. Robinson, *Elsevier eBooks* (2024) pp. 139–151.
22. A. P. Dempster, N. M. Laird, and D. B. Rubin, **39**, 1.
23. M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, pp. 226–331.
24. P. J. Rousseeuw, **20**, 53.
25. L. McInnes, J. Healy, and S. Astels, **2**, 205.
26. P. Debye, **351**, 809.
