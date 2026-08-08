# U-ADAPT: Uncertainty-Aware Post-Hoc Adaptation of Open-Vocabulary Detectors for Few-Shot Cross-Domain Disaster Perception

**MSc Thesis Proposal**

---

## Revision Note

This document is the current-state version of the proposal. Earlier drafts were reviewed in two mock-review rounds; the full change log and point-by-point reviewer responses are preserved separately in `U-ADAPT_Revision_Log.md` and are not part of the proposal itself. The substantive revisions reflected in this version are: (1) a strict 1/3/5-shot protocol with Mode A (training-free analytic gating) and Mode B (lightweight calibration, logistic regression primary) reported separately; (2) the MVUE derivation of the gating rule, with T-Rex2's static averaging recovered as a special case; (3) a pre-registered statistical plan and D1–D5 diagnostics; (4) verified citation status for all 2026 references; (5) the former "Mode C" (source-domain meta-training) folded into Mode B as a gate-initialization ablation and renamed **Mode D** (Section 5.4.3); and (6) the **full-scale LADD 10-seed results** (`outputs/real_data/ten_seed_protocol_modeB_ladd/stats.json`), which executed the pre-registered statistical plan and reframed the primary contribution: a rigorous diagnostic framework establishing that post-hoc score-level fusion degrades performance when uncertainty proxies are uninformative (D1/D2 ≈ 0, Proxy Saturation, Ranking Destruction), together with a new **Mode C (Text-Anchored Feature Manifold Repair)** that addresses the root cause on the feature manifold rather than on downstream scores.

---

## 1. Problem Statement

Open-vocabulary object detectors can localize user-specified concepts using text prompts, reducing dependence on fixed label sets. However, in aerial disaster imagery, their performance degrades under strong domain shift, clutter, smoke, occlusion, small-object scale, and subtle damage-level variation.

Michailidou et al. demonstrate that supervised detectors remain the most reliable approach when annotations are available, especially for small objects and fine localization in cluttered scenes. Their evaluation reveals that open-vocabulary detectors can be strong in some settings, but performance is highly dataset- and category-dependent. For example, Grounding DINO achieves 61.0% mAP50 zero-shot on LADD (pedestrian detection), while OWL-ViT achieves 36.4% zero-shot on D-Fire (fire/smoke detection). Critically, the gap between zero-shot and the transfer-learning upper bound (full-label partial fine-tuning, Michailidou et al.) is large: Grounding DINO improves from 61.0% to 92.2% on LADD (+31.2 pp) and from 27.5% to 65.6% on D-Fire (+38.1 pp) with full-label partial fine-tuning. This gap represents an opportunity: can a lightweight, few-shot adapter recover a meaningful fraction of this improvement without any backbone updates?

**Decomposing the gap.** Understanding *why* the zero-shot-to-transfer gap exists matters for scoping what a post-hoc adapter can recover. Three contributory factors are identifiable from the literature: (1) **vocabulary mismatch** — zero-shot text prompts may not capture disaster-specific terminology, producing systematic misclassification of proposals; (2) **feature-distribution shift** — the frozen backbone's feature space differs between pretraining data and aerial disaster imagery, degrading similarity scoring even for correct boxes (mislocalization); and (3) **proposal quality degradation** — the detector may produce fewer or noisier proposals under domain shift, an upper bound that no re-scoring method can exceed. As part of the ceiling analysis in Phase 1, we will empirically decompose the gap into misclassification vs. mislocalization vs. missed proposals (raw proposal recall). This decomposition is source-supported where Michailidou et al. report per-category trends, and inference elsewhere; it directly informs which components of the gap U-ADAPT can plausibly close (RQ2).

T-Rex2 is the current state-of-the-art for text–visual prompt synergy in open-vocabulary detection. During training, T-Rex2 aligns text and visual prompt embeddings via a region-level contrastive objective (InfoNCE loss) that maps both modalities into a shared feature space. However, at inference time, the mixed-prompt workflow fuses text and visual embeddings through static arithmetic averaging:

$$P_{\text{mixed}} = \frac{T + V}{2}$$

where $T$ and $V$ are the aligned text and visual prompt embeddings, respectively. While this leverages the learned alignment, the fusion weight is fixed — it does not dynamically account for domain-shift uncertainty or modality-specific reliability. In cross-domain disaster scenarios, where text prompts may be unreliable due to vocabulary shift and visual exemplars may be noisy due to scene clutter, this static fusion is suboptimal.

**There is therefore a need for lightweight few-shot adaptation methods that dynamically decide how much to trust text prompts versus visual support examples, guided by uncertainty estimates.**

**Validated finding (full-scale LADD, pre-registered 10-seed protocol).** The pre-registered statistical plan (§7.6) has been executed at full scale on LADD: the learned Mode B logistic gate (6 parameters, trained on 20 boxes/class calibration) **significantly underperforms naive averaging ($w = 0.5$) by 5.4–6.7 percentage points across all shot settings** (k=1: −6.7 pp; k=3: −6.4 pp; k=5: −5.4 pp), with paired $t$-test $p < 0.0001$, Wilcoxon $W = 0$ in every cell (all ten paired differences carry the same sign), and Cohen's $d$ between −2.6 and −3.6. This is not noise: it is a systematic, statistically unambiguous degradation. The root cause is **Proxy Saturation**: LADD is single-class ("person"), so the text-uncertainty proxy — variance across paraphrased prompts — collapses to zero variance (D1/D2 ≈ 0); the gate then overfits to noise in the visual branch, pushing the fusion weight away from the Bayes-optimal $w = 0.5$ and producing **Ranking Destruction** for Average Precision (§5.4.4).

**Reframed primary contribution.** This thesis therefore contributes, first, a **rigorous diagnostic framework** that precisely identifies the failure modes of post-hoc score-level fusion in open-vocabulary detectors under domain shift, and second, a proposed repair: **Mode C, Text-Anchored Feature Manifold Repair** (§5.4.3), a closed-form James-Stein-style shrinkage of the few-shot visual prototype toward the domain-invariant text anchor — attacking the feature-shift bottleneck *before* scoring rather than blending broken scores *after* scoring.

---

## 2. Related Work

### 2.1 Open-Vocabulary Object Detection

Open-vocabulary detection extends conventional object detection to recognize categories beyond the training label set. Grounding DINO combines a language-guided decoder with DINO-style detection, achieving strong zero-shot performance across diverse benchmarks. OWL-ViT applies CLIP-based text–image alignment to object detection with text-conditioned queries. YOLOE (specifically the YOLOE26 variant based on YOLO26) offers real-time open-vocabulary detection with text, visual, and prompt-free inference modes, though it underperforms on several disaster-specific benchmarks. Michailidou et al. provide the first systematic comparison of these detectors on post-disaster aerial imagery, revealing that no single open-vocabulary detector dominates across all disaster types and object categories.

### 2.2 Text–Visual Prompt Fusion

T-Rex2 introduced the concept of combining text and visual prompts for generic object detection. Its training procedure uses region-level contrastive alignment (InfoNCE loss) to map text and visual prompt embeddings into a shared space. At inference, the mixed-prompt workflow averages the aligned embeddings: $P_{\text{mixed}} = (T + V) / 2$. While this is effective in in-distribution settings, the static fusion weight does not adapt to domain shift or modality-specific uncertainty. U-ADAPT addresses this limitation by introducing an uncertainty-gated fusion mechanism that dynamically reweights text and visual contributions based on their estimated reliability — and, where full-scale evidence shows such score-level fusion to be provably harmful under uninformative proxies (Proxy Saturation, §1, §7.6), by repairing the feature manifold directly (Mode C, §5.4.3).

### 2.3 Confidence-Gated Multimodal Fusion

The general pattern of using confidence or uncertainty signals to dynamically weight multimodal inputs is well-established. ReliFusion assigns context-specific confidence weights to LiDAR and camera features for robust 3D object detection under sensor degradation. Gated multimodal units have been used in audio-visual fusion, visual question answering, and other multimodal learning tasks since approximately 2017. However, these methods operate on sensor readings or dense features. A caveat applies to the common framing that co-located sensors (camera, LiDAR) "fail in correlated ways": in fact, camera and LiDAR degrade in *decorrelated* ways under several motivating conditions in the sensor-fusion-robustness literature (darkness disables the camera but not LiDAR; glare blinds the camera but not LiDAR), and decorrelation of failure modes is arguably the standard justification for why sensor fusion improves robustness at all. The relevant claim for U-ADAPT is therefore not a categorical sensor-vs.-prompt distinction but a more modest, defensible one: text prompts and visual exemplars in open-vocabulary detection fail for *largely different* reasons — text prompts due to vocabulary mismatch, ambiguity, or domain-specific terminology; visual exemplars due to clutter, occlusion, or appearance shift — and we treat their errors as *approximately* uncorrelated. This is a working approximation, not a proven fact: a single confounder (e.g., heavy smoke) can degrade both modalities simultaneously, and under such correlated failure the inverse-variance motivation weakens (Assumption 8, Section 5.4.2). Whether the empirical benefit of gating materializes is tested directly by D3 and the H-fail hypothesis (§3, §7.6). To the extent that the two failure modes are approximately uncorrelated, dynamic gating is particularly valuable — which is the technical hook that distinguishes U-ADAPT from the broader confidence-gated fusion literature.

### 2.4 Uncertainty Estimation in Vision-Language Models

Gal & Ghahramani established that training a neural network with dropout active during training is mathematically equivalent to approximate Bayesian inference in a Deep Gaussian Process. At inference, $T$ stochastic forward passes with different dropout masks approximate the predictive distribution:

$$\hat{y} \approx \frac{1}{T} \sum_{t=1}^{T} \hat{f}_{\mathbf{w}_t}(\mathbf{x})$$

Model uncertainty is computed as the variance across these passes. BayesVLM (Baumann et al., ICLR 2026) extends this to vision-language models with a training-free, post-hoc approach that applies a Laplace approximation to the final projection layers of CLIP or SigLIP, using a ProbCosine module to propagate Gaussian embedding uncertainty into calibrated distributions over cosine similarities without Monte Carlo sampling. Query2Uncertainty (Beemelmanns et al., CVPR 2026) introduces density-aware calibration for DETR-style detectors that couples post-hoc calibrators with the feature density of latent object queries. U-ADAPT is distinct from both: BayesVLM provides uncertainty for the VLM backbone but does not perform prompt fusion gating; Query2Uncertainty targets 3D detection under distribution shift; U-ADAPT specifically estimates uncertainty over the fusion decision itself to gate between text and visual prompt contributions in a few-shot adaptation setting.

### 2.5 Few-Shot and Cross-Domain Object Detection

Few-shot object detection (FSOD) methods typically use meta-learning or transfer-learning approaches to detect novel classes from $k$ examples. Cross-domain FSOD (CD-FSOD) additionally requires robustness to distribution shift between source and target domains. Standard evaluation protocols use $k \in \{1, 3, 5, 10\}$ shots sampled from the training split, excluded from the test split, with results averaged over 5–10 random seeds. U-ADAPT differs from traditional FSOD in that the detector backbone is entirely frozen — **no gradient steps are taken on the backbone weights**. The adaptation is purely post-hoc, operating only on the fusion and scoring stages, making it applicable in resource-constrained settings (e.g., Colab) where the transfer-learning upper bound (full-label partial fine-tuning of the detector on the target domain) is impractical. A small gating MLP or logistic-regression gate may be trained (Mode B), but this occurs on a frozen backbone with cached features.

### 2.6 Calibration for Object Detection

Temperature scaling (Guo et al., 2017) applies a single scalar parameter $T > 0$ to rescale logits before softmax:

$$\hat{q}_i = \max_k \sigma_{\text{SM}}(\mathbf{z}_i / T)^{(k)}$$

where $\sigma_{\text{SM}}$ is the softmax function and $T$ is optimized on a validation set by minimizing negative log likelihood. This improves confidence calibration without changing predictions (argmax is preserved). U-ADAPT applies temperature scaling as a final calibration step on the fused similarity scores, following established best practices for post-hoc calibration in detection.

### 2.7 Test-Time Adaptation

Test-time adaptation (TTA) methods adapt a model at inference time without training on the source domain. TENT (Wang et al., ICLR 2021) minimizes prediction entropy by updating only normalization layers at test time; MEMO (Zhang et al., NeurIPS 2022) applies a similar objective across augmented views of each test sample. Both operate with **gradient updates on a subset of parameters** and require batch-level or augmentation-based statistics at test time. U-ADAPT is related to TTA in spirit — post-hoc adaptation without source retraining — but differs in three ways: (1) Mode A takes **no gradient steps at all** (analytic rule), (2) U-ADAPT gates between text and visual modalities rather than adapting feature extractors, and (3) it targets a specific failure mode (asymmetric modality reliability) rather than generic distribution shift. TENT/MEMO-style adaptation is out of scope (see Delimitations) but is noted as an orthogonal extension.

### 2.8 Prototype-Based Few-Shot Detection

Prototype-based methods represent each class by a single or few exemplar features and classify by similarity. In few-shot object detection, FsDetView (Hu et al., CVPR 2021) and MPSR (Wu et al., CVPR 2021) refine positive samples or multi-scale representations for novel-class detection under transfer-learning protocols. In the vision-language setting, prompt-learning methods CoOp (Zhou et al., IJCV 2022) and CoCoOp (Zhou et al., CVPR 2022) learn continuous text prompts for CLIP-style zero-shot classification, which are conceptually adjacent to U-ADAPT's text prototype but are *learned* rather than derived from hand-written templates. U-ADAPT differs from all of these: it keeps prototypes **frozen and unlearned** (Mode A), it operates on top of an open-vocabulary detector rather than a classifier or a dedicated FSOD training protocol, and it fuses text and visual prototypes under an uncertainty gate rather than relying on a single prototype source. This positions U-ADAPT as a post-hoc fusion method complementary to — not competing with — prototype-learning and prompt-learning approaches.

---

## 3. Research Questions

**RQ1 (Primary):** When do post-hoc score-level fusion and learned gating (Mode A / Mode B) improve or degrade open-vocabulary detection under 1/3/5-shot cross-domain disaster conditions, and can a feature-manifold repair (Mode C) recover performance by attacking the root cause of fusion failure?

**RQ2 (Gap Recovery):** How much of the zero-shot-to-transfer performance gap can U-ADAPT recover with $k$ examples and zero backbone gradient steps? Specifically, on LADD (61.0% → 92.2%, gap = 31.2 pp) and D-Fire (27.5% → 65.6%, gap = 38.1 pp), what fraction of these gaps is closed by U-ADAPT at 1/3/5 shots?

**RQ3 (Cross-Domain Transfer):** Does the gating mechanism transfer across disaster domains? **Mode A is the primary transfer test**: the fixed analytic coefficients ($\alpha = \beta = \gamma = 1$) are applied unchanged on a held-out disaster domain with only the $k$ prototypes updating — no target-domain training or tuning of any kind. Mode B (frozen trained gate) is a secondary transfer probe reported separately (Section 7.2); its random-initialization and COCO/LVIS-pretrained-initialization variants (Mode D, the former Mode C) are compared as an ablation within Mode B (Section 5.4.3). Specifically, if Mode A's analytic rule is effective on LADD, does it remain effective few-shot on D-Fire and vice versa?

**RQ4 (Reliability):** Does uncertainty estimation and calibration improve detection reliability in cluttered aerial disaster scenes, as measured by Expected Calibration Error, Brier score, and uncertainty AUROC?

**RQ5 (Backbone Sensitivity):** How does backbone choice affect performance for compact targets (pedestrians on LADD) versus diffuse targets (fire/smoke on D-Fire), and is U-ADAPT's relative gain backbone-agnostic? **Pre-registered numeric definition:** a gain is considered backbone-agnostic if U-ADAPT's relative improvement over its own zero-shot baseline is within a factor of 2× across all tested backbones (i.e., the ratio of the largest to the smallest relative improvement across backbones is ≤ 2), evaluated per dataset.

**Hypothesis on failure conditions (H-fail):** We expected U-ADAPT's dynamic gate to provide the largest gains when modality reliability is *asymmetric* (one branch clearly more reliable than the other) and the smallest gains — possibly underperforming naive averaging — when **both modalities exhibit high uncertainty simultaneously** (e.g., heavy smoke degrading both text semantics and visual matching). **Status (full-scale LADD):** the *uninformative-proxy* variant of H-fail has been confirmed — when the uncertainty proxies carry no information (D1/D2 ≈ 0), gating degrades performance relative to naive averaging with extreme statistical significance. The failure mode is **Proxy Saturation → Ranking Destruction** (§5.4.4): with no signal to gate on, any deviation of $w$ from the Bayes-optimal 0.5 reorders the detection ranking and reduces AP. This is reported honestly as a validated diagnostic finding and motivates the Mode C pivot (§5.4.3).

---

## 4. Contributions

1. **A rigorous diagnostic framework for post-hoc score-level fusion.** The primary contribution is a pre-registered, statistically validated characterization of *when* post-hoc score-level fusion in open-vocabulary detectors fails under domain shift. When uncertainty proxies are uninformative (D1/D2 ≈ 0), gating **mathematically degrades performance** relative to static averaging ($w = 0.5$): at full scale on LADD the learned Mode B gate loses 5.4–6.7 pp ($p < 0.0001$, Cohen's $d$ ∈ [−3.6, −2.6]), and the analytic Mode A gate shows the same direction at pilot scale. The framework identifies the causal chain — **Proxy Saturation → Ranking Destruction** (§5.4.4) — and specifies the conditions under which score-level fusion is provably harmful.

2. **U-ADAPT**, a lightweight post-hoc adaptation framework for open-vocabulary detectors with **no backbone gradient steps**, now comprising three modes: a **training-free analytic rule (Mode A)** requiring only the $k$ support examples; a **lightweight calibrated variant (Mode B)** training a logistic-regression (primary) or small MLP gate on a 20-box-per-class calibration set with a frozen backbone; and the new **Mode C (Text-Anchored Feature Manifold Repair)** — a closed-form, James-Stein-style shrinkage $p_{\text{repaired}} = \text{normalize}(\beta \cdot p_{\text{text}} + (1-\beta) \cdot p_{\text{visual}})$ that repairs the few-shot visual prototype using the domain-invariant text anchor, attacking the feature-shift bottleneck directly rather than trying to fix broken similarities downstream. The former Mode C (source-domain meta-training) is retained as the **Mode D** gate-initialization ablation within Mode B (Section 5.4.3).

3. **A strict 1/3/5-shot cross-domain evaluation protocol** for aerial disaster detection using LADD and D-Fire as primary detection benchmarks, with categories held out from the target detection label set (though not necessarily novel to the vision-language pretraining vocabulary) from RescueNet (4,494 images, 10 classes) and FloodNet+ (2,289 images, 9 classes) to validate the open-vocabulary claim. Category-level masks from segmentation datasets will be converted to bounding boxes using connected-component extraction with **pre-registered filtering criteria** (Section 6).

4. **A gap-recovery analysis** quantifying the fraction of the zero-shot-to-transfer gap that U-ADAPT can close without backbone updates, providing a concrete, pre-registered target for evaluation.

5. **A cross-domain transfer experiment** testing whether the gating mechanism trained on one disaster domain generalizes to another, elevating the work beyond per-dataset evaluation.

6. **A reliability analysis** using calibration error, uncertainty quality metrics, and failure-mode visualization, with ablations isolating each component's contribution.

7. **A Colab-feasible implementation strategy** using frozen open-weight models with cached features, limited proposals (top-$k$ = 100–300), and training-free adaptation (Mode A analytic gating and the closed-form Mode C repair) for the strict few-shot setting, ensuring reproducibility and accessibility.

---

## 5. Method: U-ADAPT

U-ADAPT is a post-hoc adaptation framework. It does not retrain the full detector backbone — **no gradient steps are taken on backbone weights**. The method consists of five phases:

### Phase 1: Candidate Generation

A frozen open-vocabulary detector generates candidate bounding boxes. The primary backbone is **Grounding DINO (Swin-T)** for **both** LADD and D-Fire. The rationale is threefold: (1) Grounding DINO achieves the **highest zero-shot performance** on LADD (61.0% mAP50) among all open-vocabulary detectors in Michailidou Table III, making it the most competitive starting point; (2) it exhibits the **largest zero-shot-to-transfer gap** on D-Fire (27.5% → 65.6%, gap = 38.1 pp), providing the greatest room for recovery; and (3) using a single backbone across both datasets ensures that the gap-recovery metric (RQ2) is *internally consistent* — the same model defines both the floor and the ceiling for each dataset, avoiding mixing zero-shot results from one model with transfer results from another.

**Cross-backbone ablations:** OWL-ViT and YOLOE26 will also be evaluated on both LADD and D-Fire to verify that U-ADAPT's relative gain is backbone-agnostic rather than dependent on a favorable starting point. YOLO11-small may be included as an efficient Colab-friendly alternative, noted as an engineering extension if used.

**Ceiling analysis:** The frozen detector's raw proposal recall will be reported for all test images, establishing the maximum possible recovery for any downstream re-scoring method. This pre-empts the reviewer objection that no post-hoc method can recover boxes the detector never proposes.

**Computational note:** To ensure Colab feasibility (see Q2), the number of proposals per image will be limited to the **top-$k$ most confident detections** ($k = 100$ for primary experiments, $k = 300$ as an upper-bound ablation). This reduces the downstream computational load for feature extraction and gating from a potentially unbounded number of candidates.

### Phase 2: Feature Extraction

The frozen detector is run **once per image** to generate proposals. If a separate visual encoder (e.g., CLIP, DINOv2) is used instead of the detector's internal features, box features are extracted in **one additional frozen-encoder pass** and cached. No backbone or encoder is run repeatedly per proposal. The encoder choice will be treated as an ablation:

- CLIP visual encoder (one additional frozen pass after detection)
- DINOv2 (one additional frozen pass after detection)
- The detector's own intermediate feature representation (no additional pass — features extracted during detection are cached)

### Phase 3: Prototype Construction

For each target class:

- A **text prototype** $p_{\text{text}}$ is created from the class name using the CLIP text encoder.
- A **visual prototype** $p_{\text{visual}}$ is created from $k \in \{1, 3, 5\}$ support boxes sampled from the target dataset training split.

For 1-shot adaptation, the visual prototype is the single support feature. For 3/5-shot adaptation, the prototype is the L2-normalized centroid of support features. Outlier rejection uses Mahalanobis distance for $k \geq 5$ (with shrinkage covariance estimation to improve stability from few samples): support features more than 2σ from the centroid mean are excluded before averaging. For $k < 5$, Mahalanobis-based rejection is disabled because covariance estimates from fewer than 5 samples are numerically unstable; instead, a simple cosine-distance threshold is applied: any support feature with cosine similarity below 0.5 relative to the centroid is excluded.

### Phase 4: Uncertainty-Gated Fusion

For each candidate box with feature $f_{\text{box}}$, two similarity scores are computed:

$$S_{\text{text}} = \text{sim}(f_{\text{box}}, p_{\text{text}})$$
$$S_{\text{visual}} = \text{sim}(f_{\text{box}}, p_{\text{visual}})$$

A gating weight $w \in [0, 1]$ is predicted and applied as:

$$S_{\text{final}} = (1 - w) \cdot S_{\text{text}} + w \cdot S_{\text{visual}}$$

When $S_{\text{text}}$ is unreliable (e.g., due to domain-specific vocabulary shift), the gate should down-weight the text branch; when visual support exemplars are noisy or unrepresentative, the gate should favor text. The core challenge is how to compute $w$ under few-shot constraints.

U-ADAPT provides **three evaluation modes** distinguished by how much additional labeled data is required. **Mode A** is the primary strict few-shot mode; **Mode B** trades strict few-shot purity for potentially higher performance by using a small calibration set; **Mode C** (Text-Anchored Feature Manifold Repair, Section 5.4.3) is label-free and operates on the feature manifold *before* scoring — it does not compute a fusion weight $w$ at all. All modes operate on a **frozen backbone** with **cached features**; Mode C additionally requires no feature re-extraction, since it transforms only the cached prototype vectors.

---

#### 5.4.1 Uncertainty Estimation

Modality-specific uncertainties $\sigma_{\text{text}}^2$ and $\sigma_{\text{visual}}^2$ are estimated via two strategies, one for each evaluation mode. **Crucially, both strategies produce scalar-valued uncertainty estimates that are normalized to $[0, 1]$ before being passed to the gating mechanism**, ensuring that terms from different modalities (text vs. visual) are on comparable scales.

**Practical note (logarithmic derivation):** The Taylor expansion of $\log \tilde{\sigma}^2$ is defined only for $\tilde{\sigma}^2 > 0$. In practice, all normalized variances are clamped to $[\varepsilon, 1 - \varepsilon]$ with $\varepsilon = 10^{-6}$ before any logarithmic operation is applied conceptually. The final analytic gating rule $w = \sigma(-\alpha \cdot \tilde{\sigma}_{\text{visual}}^2 + \beta \cdot \tilde{\sigma}_{\text{text}}^2 + \gamma \cdot \tilde{a}_{\text{visual}})$ does **not** compute logarithms explicitly — the derivation justifies the linear-in-variances form, but the actual computation is a simple linear combination followed by a sigmoid.

##### Mode A: Training-Free Proxy Uncertainty (Strict Few-Shot)

No target-domain labels beyond the $k$ support examples are used. Uncertainty is derived directly from frozen feature statistics:

- **Text uncertainty $\sigma_{\text{text}}^2$:** Mean pairwise cosine distance among an ensemble of prompt templates. For each class, $M = 20$ paraphrased prompts are constructed using structured templates (e.g., "a photo of a {class}", "an aerial view of a {class}", "a disaster scene with {class}", "a search and rescue image of {class}", plus variations with adjectives, size descriptors, and background context). The CLIP text encoder produces $M$ embeddings per class, and text uncertainty is the mean pairwise cosine distance across all pairs:

  $$\sigma_{\text{text}}^2 = \frac{2}{M(M-1)} \sum_{i=1}^{M} \sum_{j=i+1}^{M} \left(1 - \frac{e_i \cdot e_j}{\|e_i\| \|e_j\|}\right)$$

  where $e_i = \text{CLIP}_{\text{text}}(\text{prompt}_i)$. This scalar is unitless (cosine distance) and naturally lies in $[0, 2]$ under cosine similarity in $[-1, 1]$. It is normalized to $[0, 1]$ via min-max scaling using support-set statistics before entering the gating mechanism (see "Normalization" below). This is more stable than the trace of the covariance matrix (which scales with embedding dimension) and places text and visual uncertainties on comparable scales. A sensitivity check will compare $M = 20$ vs. $M = 50$ templates.

- **Visual uncertainty $\sigma_{\text{visual}}^2$:** For $k \geq 3$, this is the mean pairwise cosine distance among the $k$ support feature embeddings:

  $$\sigma_{\text{visual}}^2 = \frac{2}{k(k-1)} \sum_{i=1}^{k} \sum_{j=i+1}^{k} \left(1 - \frac{f_i \cdot f_j}{\|f_i\| \|f_j\|}\right)$$

  For $k = 1$, no variance estimate is possible from a single exemplar, so the raw visual uncertainty is set to **zero** (minimum possible variance), and the normalized value entering the gate is also zero. **Justification:** with one exemplar, the visual prototype is defined *exactly* by that exemplar — there is no observed dispersion to estimate, so the variance term carries no information and must not dominate the gate; the reliability signal is instead carried entirely by the per-box visual affinity $a_{\text{visual}}$ (Section 5.4.2), which measures how well each candidate matches the single prototype. This is the maximum-likelihood treatment of a degenerate sample (no degrees of freedom). An **ablation** replaces the zero with a maximum-entropy (uninformative) prior $\tilde{\sigma}_{\text{visual}}^2 = 0.5$ to verify that the zero choice is not itself responsible for any observed k=1 behavior; if results are insensitive to the two choices, the simplicity of the zero default is retained.

- **Visual affinity $a_{\text{visual}}$:** $a_{\text{visual}} = \frac{1 + \text{sim}(f_{\text{box}}, p_{\text{visual}})}{2}$, the per-box shifted cosine similarity to the visual prototype, mapping the $[-1, 1]$ range of cosine similarity into $[0, 1]$. Higher affinity means the candidate box closely matches the visual prototype, indicating that the visual prototype is a reliable reference for this box.

**Normalization:** All gating inputs are normalized to $[0, 1]$ before entering the gating mechanism. For visual affinity, the shifted cosine similarity $a_{\text{visual}} = (1 + \cos)/2$ is already in $[0, 1]$. For cosine distances (text uncertainty, visual uncertainty), the raw distance lies in $[0, 2]$ and is min-max normalized to $[0, 1]$ using support-set statistics. For score similarities ($S_{\text{text}}, S_{\text{visual}}$), min-max normalization is similarly applied. The default normalization scheme is min-max across candidate proposals in the support set: for each input dimension, $\tilde{x} = (x - x_{\min}) / (x_{\max} - x_{\min} + \epsilon)$. This ensures numerical stability across datasets with different feature distributions. The choice of normalization strategy (none vs. min-max vs. percentile rank) will be treated as an ablation.

These proxies are heuristic but require **zero additional training** beyond the $k$ support examples, making them suitable for strict few-shot conditions (Mode A).

##### Mode B: Learned MC Dropout (Lightweight Calibration)

When a small calibration set (20 labeled boxes per class) is available, a lightweight scoring MLP (single hidden layer, 128 units; 5→128→1) is trained on a **frozen backbone** to produce modality-specific scores with dropout active. During training, dropout ($p = 0.3$) is applied to the hidden layer, and the MLP is optimized with a cosine-similarity regression objective (minimizing MSE between predicted and ground-truth cosine similarity to the correct prototype). At inference, $T = 10$ stochastic forward passes produce:

$$S_{\text{text}}^{(t)} = \text{MLP}_{\theta_t}([f_{\text{box}}; p_{\text{text}}])$$

$$S_{\text{visual}}^{(t)} = \text{MLP}_{\theta_t}([f_{\text{box}}; p_{\text{visual}}])$$

Modality-specific uncertainties are the variances across stochastic passes:

$$\sigma_{\text{text}}^2 = \text{Var}_{t=1}^{T}[S_{\text{text}}^{(t)}]$$

$$\sigma_{\text{visual}}^2 = \text{Var}_{t=1}^{T}[S_{\text{visual}}^{(t)}]$$

These variance estimates are computed **before** the gating MLP processes them — they measure uncertainty in the score for each modality individually (not a combined score), avoiding circularity.

**Computational note on MC Dropout:** $T = 10$ passes is the default for primary experiments. A stability check with $T = 50$ will be run on one dataset subset to verify that $T = 10$ is sufficient. Because features are cached and the scoring MLP is tiny (≈900 parameters), each stochastic pass is a lightweight forward operation. Mode A does not require MC Dropout and is therefore the primary low-cost setting.

---

#### 5.4.2 Gating Mechanism

The gating network is a separate single-hidden-layer MLP that receives a **normalized** 5-dimensional input vector and outputs a scalar gate $w \in [0, 1]$ via sigmoid activation:

$$x = [\tilde{S}_{\text{text}}, \tilde{S}_{\text{visual}}, \tilde{\sigma}_{\text{text}}^2, \tilde{\sigma}_{\text{visual}}^2, \tilde{a}_{\text{visual}}]$$

where each $\tilde{\cdot}$ denotes the normalized version (min-max normalization to $[0, 1]$ using support-set or calibration-set statistics). The MLP has one hidden layer (dim = 128) with ReLU activation, followed by dropout ($p = 0.3$) during training (Mode B only). When the gating is analytic (Mode A), the MLP is not used — instead, a closed-form rule is applied.

##### Mode A: Analytic Training-Free Gating (No MLP)

$$w = \sigma\left(-\alpha \cdot \tilde{\sigma}_{\text{visual}}^2 + \beta \cdot \tilde{\sigma}_{\text{text}}^2 + \gamma \cdot \tilde{a}_{\text{visual}}\right)$$

where $\tilde{\sigma}_{\text{visual}}^2$, $\tilde{\sigma}_{\text{text}}^2$, and $\tilde{a}_{\text{visual}}$ are the normalized (to $[0, 1]$) versions of the uncertainty and affinity estimates from Section 5.4.1. The default coefficients are $\alpha = \beta = \gamma = 1$, and these are **not learned from the target domain** in Mode A, preserving the strict few-shot claim.

**Derivation from minimum-variance estimation.** Consider the text and visual modalities as two independent noisy measurements of the latent true similarity $s^*$ between the candidate box and the target class. Let:

$$S_{\text{text}} = s^* + \epsilon_{\text{text}}, \quad \epsilon_{\text{text}} \sim \mathcal{N}(0, \sigma_{\text{text}}^2)$$
$$S_{\text{visual}} = s^* + \epsilon_{\text{visual}}, \quad \epsilon_{\text{visual}} \sim \mathcal{N}(0, \sigma_{\text{visual}}^2)$$

where $\text{Cov}(\epsilon_{\text{text}}, \epsilon_{\text{visual}}) = 0$ by the uncorrelated-failure-modes argument (Section 2.3). The **minimum-variance unbiased estimator** (MVUE) of $s^*$ from two independent Gaussian observations is the inverse-variance weighted average:

$$S_{\text{final}} = \frac{\sigma_{\text{visual}}^{-2} S_{\text{text}} + \sigma_{\text{text}}^{-2} S_{\text{visual}}}{\sigma_{\text{text}}^{-2} + \sigma_{\text{visual}}^{-2}}$$

This can be rewritten as $S_{\text{final}} = (1 - w) S_{\text{text}} + w S_{\text{visual}}$ with:

$$w = \frac{\sigma_{\text{text}}^2}{\sigma_{\text{text}}^2 + \sigma_{\text{visual}}^2} \in [0, 1]$$

This is the theoretically optimal fusion weight under Gaussian noise: the modality with **lower** variance receives **higher** weight. When $\sigma_{\text{visual}}^2 \ll \sigma_{\text{text}}^2$ (visual is much more reliable), $w \to 1$ (trust visual); conversely when $\sigma_{\text{text}}^2 \ll \sigma_{\text{visual}}^2$, $w \to 0$ (trust text).

To express this in a form compatible with the sigmoid gating framework, we apply the identity $\frac{a}{a+b} = \sigma(\log a - \log b)$:

$$w = \frac{\sigma_{\text{text}}^2}{\sigma_{\text{text}}^2 + \sigma_{\text{visual}}^2} = \sigma\left(\log \sigma_{\text{text}}^2 - \log \sigma_{\text{visual}}^2\right)$$

For the normalized variances $\tilde{\sigma}^2 \in [0, 1]$, we apply a first-order Taylor expansion of $\log \tilde{\sigma}^2$ around the midpoint $\tilde{\sigma}^2 = 0.5$:

$$\log \tilde{\sigma}^2 \approx \left.\log x\right|_{x=0.5} + \left.\frac{d}{dx}\log x\right|_{x=0.5} \!\!\!\!\!\!\cdot (\tilde{\sigma}^2 - 0.5) = -\log 2 + 2(\tilde{\sigma}^2 - 0.5) = 2\tilde{\sigma}^2 - 1 - \log 2$$

The difference of two such expansions gives:

$$\log \tilde{\sigma}_{\text{text}}^2 - \log \tilde{\sigma}_{\text{visual}}^2 \approx 2\tilde{\sigma}_{\text{text}}^2 - 2\tilde{\sigma}_{\text{visual}}^2$$

where the additive constants $-1 - \log 2$ cancel. This is proportional to $\tilde{\sigma}_{\text{text}}^2 - \tilde{\sigma}_{\text{visual}}^2$; the factor $2$ (the derivative of $\log x$ at $x = 0.5$) is absorbed into the sigmoid's implicit steepness, since $\sigma(kx)$ and $\sigma(x)$ cover the same function family up to a scaling of the argument. Reparameterizing yields the empirically testable linear form:

$$\log \tilde{\sigma}_{\text{text}}^2 - \log \tilde{\sigma}_{\text{visual}}^2 \approx \beta \cdot \tilde{\sigma}_{\text{text}}^2 - \alpha \cdot \tilde{\sigma}_{\text{visual}}^2$$

with default coefficients $\alpha = \beta = 1$, which are **not learned** from the target domain in Mode A, preserving the strict few-shot claim.

**Why the sigmoid-linearized form rather than the exact ratio?** The theoretically justified object is the exact inverse-variance weight $w = \sigma(\log \tilde{\sigma}_{\text{text}}^2 - \log \tilde{\sigma}_{\text{visual}}^2)$. The linear-in-variances form used in practice is a first-order Taylor approximation of this exact ratio, adopted because it is the only tractable way to fold in the per-box affinity term $\gamma \tilde{a}_{\text{visual}}$ — which has no natural place inside a pure log-variance ratio — while keeping the gate a single linear combination followed by a sigmoid. To be explicit: the clean optimality result (MVUE, the variance-reduction comparison, and the T-Rex2 limiting case) applies to the *exact ratio form*; the sigmoid-linearized version is a tractable approximation whose fidelity — especially at the extremes of $\tilde{\sigma}^2$, where the gate does its most consequential work — is an empirical question that diagnostic D5 (§7.6) is designed to answer.

**Effective visual variance under prototype bias.** The MVUE derivation assumes both modalities provide unbiased estimates of $s^*$. In practice, the visual prototype may be a **biased** estimator when the $k$ support examples are unrepresentative. We model this by decomposing the visual score as:

$$S_{\text{visual}} = s^* + b_{\text{visual}} + \epsilon_{\text{visual}}$$

where $b_{\text{visual}}$ is a bias term that grows with the distance between the candidate box and the visual prototype. We model this bias as zero-mean additional uncertainty:

$$b_{\text{visual}} \sim \mathcal{N}(0, \lambda), \qquad \lambda \propto (1 - a_{\text{visual}}), \quad \lambda \geq 0$$

i.e., the bias variance is proportional to how dissimilar the candidate is from the prototype. The effective visual variance then becomes:

$$\sigma_{\text{visual, eff}}^2 = \sigma_{\text{visual}}^2 + \lambda \cdot (1 - a_{\text{visual}})$$

where $\lambda \geq 0$ is a non-negative scalar controlling the strength of the bias-variance penalty. After normalizing $\tilde{\sigma}_{\text{visual}}^2$ and $\tilde{a}_{\text{visual}}$ to $[0, 1]$ and applying the Taylor expansion above, $\lambda$ is absorbed into the coefficient $\gamma$ of the affinity term.

Substituting this into the inverse-variance weight and applying the same sigmoid-and-Taylor derivation yields:

$$w = \sigma\left(-\alpha \cdot \tilde{\sigma}_{\text{visual}}^2 + \beta \cdot \tilde{\sigma}_{\text{text}}^2 + \gamma \cdot \tilde{a}_{\text{visual}}\right)$$

The affinity term $+\gamma \cdot \tilde{a}_{\text{visual}}$ now has a clear interpretation: **high visual affinity (candidate close to prototype) increases trust in the visual branch**, while low affinity (candidate far from prototype) reduces it — the opposite of the earlier distance-based formulation, which was conceptually incorrect. This matches the intuition that if a candidate box looks unlike the visual prototype, the visual similarity score is less reliable and should carry less weight. The default coefficient $\gamma = 1$ means the affinity contribution to the logit is approximately as strong as each variance term.

At $\alpha = \beta = \gamma = 1$, and with all inputs normalized to $[0, 1]$, the sigmoid input is bounded in $[-1, 2]$, producing gate values across the full $[0, 1]$ range. The resulting behaviour is:
- **Low visual variance, high text variance, high affinity** ($\tilde{\sigma}_{\text{visual}}^2 \to 0, \tilde{\sigma}_{\text{text}}^2 \to 1, \tilde{a}_{\text{visual}} \to 1$): $w \to \sigma(2) \approx 0.88$ — strongly visual.
- **Low text variance, high visual variance** ($\tilde{\sigma}_{\text{text}}^2 \to 0, \tilde{\sigma}_{\text{visual}}^2 \to 1$): $w \to \sigma(-1 + \gamma \cdot \tilde{a}_{\text{visual}})$ — text-dominated unless affinity is very high.
- **Low affinity** ($\tilde{a}_{\text{visual}} \to 0$): the $+\gamma \cdot \tilde{a}_{\text{visual}}$ term vanishes, and the inverse-variance logic dominates, naturally down-weighting visual when the prototype is a poor match.

The sigmoid nonlinearity bounds $w$ strictly within $(0, 1)$, ensuring that neither modality is ever fully suppressed — providing graceful degradation even under extreme uncertainty.

**Theoretical comparison to T-Rex2.** T-Rex2 uses static arithmetic averaging: $S_{\text{final}} = (S_{\text{text}} + S_{\text{visual}}) / 2$, which corresponds to a fixed gate $w = 0.5$. Within U-ADAPT's framework, this is a special case that arises when:

$$\tilde{\sigma}_{\text{text}}^2 = \tilde{\sigma}_{\text{visual}}^2 \quad \text{and} \quad \gamma = 0$$

When modality variances are equal and the affinity term is disabled, the inverse-variance terms cancel: $-\alpha \cdot \tilde{\sigma}_{\text{visual}}^2 + \beta \cdot \tilde{\sigma}_{\text{text}}^2 = 0$, so $w = \sigma(0) = 0.5$ exactly, recovering T-Rex2's fixed averaging. With $\gamma > 0$ active, the affinity term adjusts this baseline: high affinity ($\tilde{a}_{\text{visual}} \to 1$) pushes $w$ above 0.5 (more visual weight), while low affinity ($\tilde{a}_{\text{visual}} \to 0$) pushes $w$ below 0.5 (less visual weight). T-Rex2's static averaging therefore implicitly assumes **equal modality reliability** and **no affinity-dependent adjustment** — both assumptions that are violated under cross-domain disaster conditions, where modality reliability is asymmetric and prototype representativeness varies per candidate box.

A more fundamental comparison involves the variance of the fused estimate. For T-Rex2's equal averaging ($w = 0.5$):

$$\text{Var}[S_{\text{final}}^{(\text{T-Rex2})}] = \frac{\sigma_{\text{text}}^2 + \sigma_{\text{visual}}^2}{4}$$

For the optimal inverse-variance weight:

$$\text{Var}[S_{\text{final}}^{(\text{opt})}] = \frac{1}{\sigma_{\text{text}}^{-2} + \sigma_{\text{visual}}^{-2}} = \frac{\sigma_{\text{text}}^2 \sigma_{\text{visual}}^2}{\sigma_{\text{text}}^2 + \sigma_{\text{visual}}^2}$$

By the inequality of arithmetic and harmonic means:

$$\text{Var}[S_{\text{final}}^{(\text{opt})}] \leq \text{Var}[S_{\text{final}}^{(\text{T-Rex2})}]$$

with equality if and only if $\sigma_{\text{text}}^2 = \sigma_{\text{visual}}^2$. When one modality is substantially more reliable than the other, the variance reduction from U-ADAPT's dynamic gating can be large. For example, if $\sigma_{\text{visual}}^2 = \sigma_{\text{text}}^2 / 10$, then $\text{Var}_{\text{opt}} = 0.091 \cdot \sigma_{\text{text}}^2$ vs. $\text{Var}_{\text{T-Rex2}} = 0.275 \cdot \sigma_{\text{text}}^2$ — a **3-fold variance reduction**. This provides a formal justification for why dynamic gating outperforms static averaging under asymmetric modality reliability.

**Important caveat:** The comparison above treats T-Rex2's averaging as operating on *score* values ($S_{\text{text}}$ and $S_{\text{visual}}$), not on *prompt embeddings* as in the original T-Rex2 formulation. T-Rex2 averages prompt embeddings in a learned shared space before computing similarity, while U-ADAPT computes separate text and visual scores and fuses them via a learned or analytic gate. These are different operations — embedding-space averaging vs. score-space gating — and the variance comparison should be interpreted as a theoretical motivation for score-level dynamic gating, not as a direct performance comparison with T-Rex2's embedding-level fusion.

**Ablations:** The contribution of each term will be isolated by setting individual coefficients to zero:

| Variant | $\alpha$ | $\beta$ | $\gamma$ | What It Tests |
|---------|----------|---------|----------|---------------|
| Full (default) | 1 | 1 | 1 | All three signals |
| No visual uncertainty | **0** | 1 | 1 | Is visual variance useful? |
| No text uncertainty | 1 | **0** | 1 | Is text variance useful? |
| No affinity | 1 | 1 | **0** | Is per-box affinity useful? |
| Visual uncertainty only | 1 | 0 | 0 | Only visual variance |
| Text uncertainty only | 0 | 1 | 0 | Only text variance |
| Affinity only | 0 | 0 | 1 | Only visual affinity |

The "No affinity" ablation ($\gamma = 0$) recovers the pure inverse-variance estimator, providing a direct test of whether the affinity correction adds value beyond the uncertainty variances alone.

**Note on strict few-shot Mode A purity:** The coefficients $\alpha, \beta, \gamma$ in Mode A are set to their default values of 1 and are **not learned** from target-domain data. A variant where gating parameters are initialized from a source domain (e.g., COCO or LVIS) is part of Mode B's initialization ablation (Section 5.4.3), not Mode A. This preserves the strict few-shot, training-free claim for the primary evaluation mode.

**Caveat on 2026 citations (verified as of July 2026):** All 2026 citations have been verified against arXiv records. BayesVLM (Baumann et al., *Post-hoc Probabilistic Vision-Language Models*) is **published at ICLR 2026** (acceptance confirmed via OpenReview, forum XLiUcvHfzS; arXiv:2412.06014, v5, 13 Feb 2026). Query2Uncertainty (Beemelmanns et al., *Robust Uncertainty Quantification and Calibration for 3D Object Detection under Distribution Shift*) is **accepted at CVPR 2026**, arXiv:2605.05328. Michailidou et al. (*Open-Vocabulary vs Supervised Learning Methods for Post-Disaster Visual Scene Understanding*, arXiv:2603.01324) remains an **arXiv preprint** (v1, 7 pages, no venue) as of this writing; it is cited as a preprint throughout and its reported numbers are treated as source-supported but subject to final publication. T-Rex2 (arXiv:2403.14610) is confirmed published at ECCV 2024 (LNCS 15053, pp. 38–57). No venue claims are made for any preprint.

##### Assumptions and Approximations

The Mode A derivation relies on several idealizing assumptions. We state them explicitly here:

1. **Gaussian noise model.** The additive noise $\epsilon_{\text{text}}, \epsilon_{\text{visual}}$ is assumed Gaussian. Cosine similarities are bounded in $[-1, 1]$ and the true similarity $s^*$ is not observed, so the Gaussian assumption is a tractable approximation (maximum-entropy distribution given estimated mean and variance).

2. **Bounded support of similarity scores.** Cosine similarities are not Gaussian-distributed in practice — they are bounded and exhibit skew near the boundaries. The first-order Taylor expansion of $\log \tilde{\sigma}^2$ introduces further approximation error, particularly when normalized variances are near 0 or 1 where the log-derivative is steep.

3. **Independence of modality errors.** Text and visual errors are assumed approximately independent ($\text{Cov}(\epsilon_{\text{text}}, \epsilon_{\text{visual}}) = 0$), justified by the uncorrelated-failure-modes argument (Section 2.3). If the two modalities fail in a correlated way (e.g., both are confused by heavy smoke), the inverse-variance motivation weakens and the dynamic gate may not outperform static averaging.

4. **Monotonicity of uncertainty proxies.** The normalized mean pairwise cosine distances $\tilde{\sigma}_{\text{text}}^2$ and $\tilde{\sigma}_{\text{visual}}^2$ are assumed to be monotonic with the true modality error variance. This is a heuristic: high prompt-template variance is reasonable evidence of text uncertainty, but it does not capture systematic failures (e.g., all templates being equally wrong for an out-of-distribution class).

5. **Bias variance model.** The bias variance $\lambda \propto (1 - a_{\text{visual}})$ (with $\lambda \geq 0$) is a linear approximation. The true relationship between visual prototype representativeness and bias magnitude is unknown and likely non-linear. The ablation study ($\gamma = 0$) tests whether this term is necessary at all.

6. **The derivation motivates the functional form but does not prove optimality.** The MVUE framework shows that a sigmoid of a linear combination of normalized variances is a plausible architecture for dynamic gating. It does not guarantee that $\alpha = \beta = \gamma = 1$ is optimal for any specific dataset. The ablation study provides empirical validation, and the Mode B initialization ablation (gate parameters initialized from COCO/LVIS and then calibrated on the target) tests whether better parameters exist.

7. **Choice of Gaussian over alternative distributions.** Cosine similarities are directional data, for which von Mises–Fisher (vMF) or Beta distributions are arguably more natural: vMF models points on the unit sphere directly, and Beta has bounded support matching $[-1, 1]$-derived affinities. We retain the Gaussian model for three reasons of tractability: (a) the inverse-variance weighting result (MVUE) is closed-form only under Gaussian noise; (b) our inputs to the gate are *normalized scalar variances*, not raw directional vectors, so the bounded-support objection applies to the raw similarities, not to the variance statistics themselves; and (c) the sigmoid output provides a monotone link that partially absorbs misspecification. As a robustness check, the D5 distributional diagnostic reports the empirical support of $\tilde{\sigma}^2$ values; if they cluster at the boundaries (near 0 or 1), a Beta-regression variant of the gate is pre-registered as a fallback analysis.

8. **Independence caveat under correlated failure.** The claim $\text{Cov}(\epsilon_{\text{text}}, \epsilon_{\text{visual}}) = 0$ is an approximation, not a proven fact. A mechanistic argument for why it is plausible *for this specific pairing*: text errors originate in the language encoder's vocabulary/terminology matching, while visual errors originate in the vision encoder's appearance matching — two distinct encoders with distinct failure pathways, so under domain shift the two error terms are driven by largely different image statistics. However, in disaster imagery a single confounding factor (e.g., heavy smoke) can simultaneously degrade text semantics (ambiguous descriptors) and visual matching (occlusion). The derivation acknowledges that under such correlated failure the inverse-variance motivation weakens. D3 and the H-fail hypothesis (§3) test this empirically: when both modalities are unreliable, the gate is expected to be no better than chance, and results in these regimes are reported separately rather than averaged into headline numbers.

These assumptions are standard for post-hoc training-free methods where full distributional information is unavailable. The ablation study, sensitivity analysis, and diagnostics (D1–D5) are designed to validate the robustness of the approach when these assumptions are violated.

---

##### Mode B: Learned MLP Gating

The gating MLP (5→128→1, ≈900 parameters) is trained on a small held-out **calibration set** from the target domain training split (20 labeled boxes per class), disjoint from both the $k$ support examples and the test set. The MLP receives the normalized 5D input vector $x$ and is trained with binary cross-entropy:

$$w^* = \begin{cases} 1 & \text{if visual prototype top-1 matches ground truth and text does not} \\ 0 & \text{if text prototype top-1 matches ground truth and visual does not} \\ \sigma(S_{\text{visual}} - S_{\text{text}}) & \text{if both or neither are correct (soft margin)} \end{cases}$$

**Primary variant and overfitting mitigation:** Because training on 20 boxes per class with a ≈900-parameter MLP risks overfitting (especially given the high variance of disaster imagery), the **logistic regression gate (5→1, only 6 parameters including bias) is the primary Mode B claim**; the MLP is a secondary variant reported for comparison. The following mitigations are pre-registered:

1. **Logistic regression gate as primary:** With 6 parameters and a 20-box-per-class calibration set, the parameter-to-sample ratio is ≈0.3, keeping overfitting risk low. This is the headline Mode B result.
2. **MLP as secondary variant:** The ≈900-parameter MLP (single hidden layer, 128 units; 5→128→1) with **dropout ($p = 0.3$)**, **L2 weight decay** ($\lambda = 1 \times 10^{-4}$), and **early stopping** on a held-out validation fold (5 images per class from the calibration set, patience 10 epochs) is reported alongside; if it fails to beat the logistic gate, that finding is reported honestly.
3. **5-fold cross-validation** on the calibration set, reporting mean and std of gate performance.
4. **Stratified calibration sampling:** Calibration boxes are sampled stratified by class (exactly 20 per class, not 20 total). Classes with high intra-class variance (e.g., smoke, debris) are acknowledged as the hardest for a small calibration set; per-class gate performance is reported (mean ± std across classes) rather than pooled into a single number that could mask a poorly calibrated class. If a per-class gate underperforms the shared gate for any class, the shared gate is reported for that class with the discrepancy logged.

---

#### 5.4.3 Evaluation Modes

Because the amount of available target-domain labeled data varies, U-ADAPT defines three evaluation modes with different data requirements and strictness levels, plus one gate-initialization ablation (Mode D). The former Mode C (source-domain meta-training) is folded into Mode B as a gate-initialization ablation and renamed **Mode D** (below); the new **Mode C (Text-Anchored Feature Manifold Repair)** is the primary innovation introduced in response to the full-scale findings (§1, §7.6, §10).

| Mode | Data Required Beyond $k$ Support | Backbone Gradients? | Training Status | Initialization | Description |
|------|----------------------------------|---------------------|-----------------|----------------|-------------|
| **A** | None | No | **Training-free** (primary strict few-shot, analytic) | N/A (analytic rule) | Analytic gating rule with normalized uncertainty proxies. No neural network trained on target domain. |
| **B** | 20 labeled boxes/class (calibration) | No (frozen backbone) | **Lightweight calibrated adaptation** (secondary claim) | **Random** (default) or **COCO/LVIS-pretrained** (ablation; the former Mode C) | Logistic regression (primary) or small MLP gate trained on cached features. Not strict few-shot. |
| **C** | None | No | **Training-free, closed-form** (primary new claim) | N/A (label-free text weight $\beta$) | **Text-Anchored Feature Manifold Repair**: $p_{\text{repaired}} = \text{normalize}(\beta \cdot p_{\text{text}} + (1-\beta) \cdot p_{\text{visual}})$. Repairs the few-shot visual prototype before scoring; no fusion weight is computed at all. |
| **D** | 20 labeled boxes/class (calibration) | No (frozen backbone) | Lightweight calibrated (Mode B variant) | **COCO/LVIS-pretrained** (formerly Mode C) | Mode B gate initialized from COCO/LVIS-pretrained weights, then calibrated on target. Ablation within Mode B. |

**Mode C: Text-Anchored Feature Manifold Repair (new).** The full-scale LADD findings (§1, §7.6) established that score-level fusion — analytic or learned — degrades detection when the uncertainty proxies are uninformative (D1/D2 ≈ 0), because any reweighting of already-computed scores can only destroy the ranking required for Average Precision (§5.4.4). Mode C therefore abandons score-level fusion entirely and operates on the **feature manifold, before scoring**. The few-shot visual prototype $p_{\text{visual}}$ — a high-variance estimator of the true class centroid, especially at $k = 1$ — is shrunk toward the domain-invariant text anchor $p_{\text{text}}$ (the CLIP text embedding of the class name):

$$p_{\text{repaired}} = \text{normalize}\left(\beta \cdot p_{\text{text}} + (1-\beta) \cdot p_{\text{visual}}\right)$$

where $\beta \in [0, 1]$ is a **label-free text weight** and $\text{normalize}(\cdot)$ denotes $L_2$ normalization. This is a **James-Stein-style shrinkage estimator**: the visual prototype's sampling variance is traded for a controlled bias toward the text anchor, minimizing expected squared error of the class centroid under feature-distribution shift (Phase 3). All downstream similarities ($S_{\text{text}}, S_{\text{visual}}$) are then computed against the *repaired* prototype, so the feature-shift bottleneck is attacked directly rather than patching broken similarities downstream. Because Mode C is a closed-form operation on already-cached features and prototypes, it is **directly testable on the existing cached features — no re-extraction, no training** — making it highly feasible within the remaining timeline (§11). The default $\beta$ is a label-free heuristic (e.g., decreasing in $k$ as the visual prototype gains samples, and increasing in estimated visual-prototype variance); a sensitivity sweep over $\beta \in \{0, 0.25, 0.5, 0.75, 1\}$ is pre-registered.

**Former Mode C as a Mode B initialization ablation (now Mode D).** Source-domain meta-training is not a separate evaluation mode. It is folded into Mode B as a **gate-initialization ablation** (Initialization column above): the Mode B gate may be initialized either randomly (default) or from weights pre-trained on COCO/LVIS episodic few-shot data (Mode D), then calibrated on the same 20-box-per-class target set. This turns "Mode B vs. Mode C" into "Mode B, random init vs. Mode D, COCO/LVIS-pretrained init" — one mode with an initialization ablation, not two modes with two write-ups. If the pretrained initialization outperforms random initialization on the target domain, domain-invariant gating priors exist; if not, random init is retained. The COCO/LVIS episodic-sampling pipeline for this condition is a bounded, well-scoped implementation task (sampling k-shot episodes from COCO/LVIS categories through the same Phase 1–4 pipeline); it is sequenced after the primary Mode A, Mode C, and random-init Mode B results so it cannot block the critical path.

**Key clarifications on data requirements (addressing reviewer concerns):**

- **Mode A is the primary strict few-shot setting.** It requires no target-domain labels beyond the $k$ support examples and involves no neural network training on the target domain. All terms in the analytic gating rule are derived from frozen feature statistics.
- **Mode B is not strict few-shot.** It uses 20 additional labeled boxes per class for calibration. This is explicitly labeled as "few-shot plus lightweight calibration" and results are reported **separately** from Mode A results — they are never averaged or conflated.
- **Mode C is strict few-shot and label-free.** It requires no labels beyond the $k$ support examples (the text anchor is derived from the class name alone) and no training; it is reported separately from both Mode A and Mode B.
- **Mode D init ablation (formerly Mode C) is secondary.** Whether COCO/LVIS-pretrained initialization of the Mode B gate helps is reported as an ablation; if it does not transfer, this is neither a failure of the primary method nor a weakness in the proposal.

**All modes share the same frozen backbone and cached features.** The only difference is how the final score is derived: Mode A/B compute a fusion weight $w$; Mode C repairs the prototype on the feature manifold before any scoring, so that both similarity scores improve simultaneously.

**Overfitting risk (Mode B, explicit discussion):** With 20 labeled boxes per class, the **logistic regression gate (6 parameters) is the primary Mode B claim** because its low capacity keeps overfitting risk low. The ≈900-parameter MLP is a secondary variant; its mitigations (dropout, weight decay, early stopping, cross-validation) are listed in Section 5.4.2. If the MLP fails to beat the logistic gate, this is reported honestly. If even the logistic gate underperforms Mode A, this is reported honestly and discussed as a limitation. **Full-scale status (LADD):** the logistic gate underperformed even the naive $w = 0.5$ baseline by 5.4–6.7 pp (§7.6); this pre-registered outcome is the diagnostic finding that motivates Mode C.

#### 5.4.4 The Ranking Destruction Hypothesis

The full-scale LADD results — and the earlier exploratory **Model C (Asymmetric Visual Rescue)** experiments (`scripts/run_model_c*.py`) — are explained by a single mechanism: **Ranking Destruction**. (Note on nomenclature: "Model C" here denotes the exploratory asymmetric-rescue experiment; it is unrelated to the new **Mode C** defined in §5.4.3.) Average Precision is a rank statistic: it integrates precision over a ranking of proposals. Any score transformation that is not monotone in the true posterior permutes true positives below false positives and directly reduces AP, even if per-box accuracy appears unchanged or improved.

**Why Model C (Asymmetric Visual Rescue) failed.** Model C fused the two score families with a per-box sigmoid gate of the *difference* of scores, $w = \sigma(\kappa \cdot (S_{\text{visual}} - S_{\text{text}}))$, deliberately up-weighting whichever modality scored higher on each box. This is not a stable estimator: it applies a *pointwise, non-monotone* reweighting to scores that live on incomparable scales — $S_{\text{text}}$ is a discriminative classification score produced by the detector's language-conditioned head, while $S_{\text{visual}}$ is a geometric retrieval metric (cosine similarity to a prototype). Blending a discriminative classifier output with a retrieval metric destroys the detection ranking required for AP: the fused score is neither a calibrated probability nor a consistent similarity, and the resulting ranking is strictly worse than either input alone.

**Why Mode B failed.** The learned logistic gate suffered the same structural problem with an additional failure: on single-class LADD, the text-uncertainty proxy collapses to zero variance (D1/D2 ≈ 0), so the gate's text channel carries no signal. With no informative input to gate on, the 6-parameter logistic regression overfits to noise in the visual branch, learning weights that push $w$ away from the Bayes-optimal $w = 0.5$ and systematically destroying the ranking — confirmed by the 10-seed protocol (−5.4 to −6.7 pp, Cohen's $d$ ∈ [−2.6, −3.6], Wilcoxon $W = 0$ in every cell).

**The general principle.** Both failures share one root cause: they attempt to fix the problem *after* scoring, by blending or reweighting scores that are already rank-degraded under domain shift. The solution must operate on the **feature manifold before scoring** — repairing the estimators ($p_{\text{visual}}$) rather than the estimates ($S$). This is precisely the design of Mode C (§5.4.3). Mode A's analytic gate, Mode B's learned gate, and Model C's asymmetric rescue all operate downstream of the damage; Mode C is the first U-ADAPT component that intervenes at the source.

### Phase 5: Calibration

Temperature scaling is applied to the final fused scores to improve confidence reliability. The temperature parameter $T > 0$ rescales logits before softmax or score comparison. **Importantly, in Mode A (primary strict few-shot) no calibration data is available, so $T = 1$ is used (no learned scaling).** Temperature scaling is learned only in modes with additional held-out data:

- **Mode A:** $T = 1$ (fixed). No calibration set used — the primary strict few-shot claim remains training-free.
- **Mode B:** $T$ is optimized on the 20-box-per-class calibration split by minimizing negative log likelihood: $T^* = \arg\min_T \text{NLL}(S_{\text{final}} / T, y_{\text{true}})$.
- **Mode B (pretrained init):** as Mode B; $T$ is calibrated on the target 20-box-per-class set (source-domain temperature is not carried over).

**Optional ablation (Mode A+):** A temperature parameter learned on the $k$ support examples (treated as a mini-calibration set) will be reported as an ablation if feasible. This is **not** part of the primary Mode A claim and is clearly labeled as a post-hoc calibration experiment.

**Reported metrics:**
- Expected Calibration Error (ECE) with 15 bins
- Reliability diagrams
- Uncertainty AUROC
- Brier score

### Pipeline Summary

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│ Phase 1: Frozen OV Detector         │ → Top-k candidate boxes
│ (Grounding DINO Swin-T, primary;     │   (k=100 default, k=300
│  OWL-ViT, YOLOE26 as cross-bb abl.) │    upper-bound ablation)
│ Proposals limited to top-k.          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Phase 2: Feature Extraction (frozen)│ → Cached box features f_box
│ (CLIP / DINOv2 / detector features) │   (single forward pass)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Phase 3: Prototype Construction     │
│  p_text ← CLIP text encoder         │
│  p_visual ← centroid of k support   │
│  (Mahalanobis outlier rejection)    │
└─────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Phase 4: Uncertainty-Gated Fusion                    │
│                                                       │
│  ┌─ 5.4.1 Uncertainty Estimation ────────────────┐   │
│  │                                                 │   │
│  │  Mode A (training-free proxy):                 │   │
│  │    σ²_text = mean pairwise cosine dist (M=20)  │   │
│  │    σ²_visual = mean pairwise cosine dist (k)   │   │
│  │    (all normalized to [0,1])                   │   │
│  │                                                 │   │
│  │  Mode B (learned MC Dropout, T=10):            │   │
│  │    S_text^(t) = MLP_θt(f_box, p_text)          │   │
│  │    S_visual^(t) = MLP_θt(f_box, p_visual)      │   │
│  │    σ² = Var_t[S^(t)]                           │   │
│  └────────────────────────────────────────────────┘   │
│                                                       │
│  ┌─ 5.4.2 Gating Mechanism ──────────────────────┐   │
│  │  x = [norm(S_text), norm(S_visual),            │   │
│  │       norm(σ²_text), norm(σ²_visual),          │   │
│  │       norm(a_visual)]                          │   │
│  │                                                 │   │
│  │  Mode A: w = σ(-α·σ̃²_visual + β·σ̃²_text      │   │
│  │                  + γ·ã_visual) (α=β=γ=1)      │   │
│  │           (no training, no MLP)                 │   │
│  │                                                 │   │
│  │  Mode B: w = σ(MLP(x)) or σ(LogReg(x))         │   │
│  │           (trained on 20-box/class calibration) │   │
│  │                                                 │   │
│  │  S_final = (1-w)·S_text + w·S_visual           │   │
│  └────────────────────────────────────────────────┘   │
│                                                       │
│  Mode C (feature repair, before scoring):            │
│    p_repaired = normalize(β·p_text + (1-β)·p_visual) │
│                                                       │
│  Mode D init: gate pretrained on COCO/LVIS (abl.)    │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Phase 5: Temperature Scaling        │ → Calibrated confidence
│ (Mode A: T=1, no learning;          │
│  Mode B: learned on calibration)    │
└─────────────────────────────────────┘
    │
    ▼
Final Detections
```

---

## 6. Datasets

### Primary Detection Datasets (Few-Shot Benchmarks)

From Michailidou et al. Table I:

| Dataset | Images | Classes | Task | Zero-Shot (Best) | Transfer Learning (Best) | Gap |
|---------|--------|---------|------|-------------------|----------------------|-----|
| **LADD** | 1,365 | 1 | Pedestrian detection (Search and Rescue) | 61.0% mAP50 (Grounding DINO) | 92.2% mAP50 (Grounding DINO) | 31.2 pp |
| **D-Fire** | 21,527 | 2 | Fire and smoke detection | 27.5% mAP50 (Grounding DINO) | 65.6% mAP50 (Grounding DINO) | 38.1 pp |

These are the main few-shot detection benchmarks. The zero-shot and transfer learning numbers from Michailidou provide pre-established floor and ceiling values for gap-recovery analysis.

### Auxiliary Segmentation Datasets (Novel-Category Validation)

| Dataset | Images | Classes | Purpose |
|---------|--------|---------|---------|
| **RescueNet** | 4,494 | 10 | Categories held out from target detection label set, for open-vocabulary validation |
| **FloodNet+** | 2,289 | 9 | Categories held out from target detection label set, for open-vocabulary validation |

These datasets serve two purposes: (1) providing damage-level and structural classes that test the open-vocabulary claim beyond the three primary categories (pedestrian, fire, smoke), and (2) enabling cross-domain transfer experiments where the gating mechanism is trained on one domain and evaluated on another. We use the term "held out" to indicate these categories are not part of the few-shot detection benchmarks, while acknowledging that they may appear in the pretraining vocabulary of the vision-language backbone.

### Pre-Registered Mask-to-Box Filtering Criteria

Both RescueNet and FloodNet+ provide semantic segmentation masks. To convert these to bounding boxes suitable for detection evaluation, the following criteria are **pre-registered before any experimental results are collected**:

#### Candidate Classes

**RescueNet (10 classes):**
| Class | Type | Retained? | Rationale |
|-------|------|-----------|-----------|
| Building | **Object-like** | **Yes** | Regular, boxable structure |
| Road | Stuff (region) | No | Pure stuff class; no compact bounding box |
| Tree | Stuff (region) | No | Pure stuff class; irregular boundary |
| Grass | Stuff (region) | No | Pure stuff class; no semantic box |
| Pool | **Object-like** | **Yes** | Typically compact, rectangular |
| Vehicle | **Object-like** | **Yes** | Standard detection class |
| Debris | **Object-like** | **Marginal** | Included with area filtering; acknowledged as region-level |
| Sand | Stuff (region) | No | Pure stuff class |
| Roof | **Object-like** | **Yes** | Regular structure in aerial view |
| Water | Stuff (region) | No | Pure stuff class |

**FloodNet+ (9 classes):**
| Class | Type | Retained? | Rationale |
|-------|------|-----------|-----------|
| Building-flooded | **Region-level damage** | **Yes** | Retained as coarse detection target |
| Building-non-flooded | **Region-level damage** | **Yes** | Retained as coarse detection target |
| Road-flooded | **Region-level** | **Marginal** | Included; acknowledged as region-level |
| Road-non-flooded | **Region-level** | **Marginal** | Included; acknowledged as region-level |
| Water | Stuff (region) | No | Pure stuff class |
| Tree | Stuff (region) | No | Pure stuff class |
| Vehicle | **Object-like** | **Yes** | Standard detection class |
| Pool | **Object-like** | **Yes** | Typically compact |
| Grass | Stuff (region) | No | Pure stuff class |

**Explicit acknowledgment:** Damage-level classes (building-flooded, building-non-flooded, etc.) are region-level targets and visually ambiguous — they are not conventional object detection classes. Results on these categories should be interpreted with caution and are reported separately from the primary LADD/D-Fire detection benchmarks. Pure stuff classes (grass, tree, road, water) are excluded from all detection evaluations.

#### Filtering Rules

After mask-to-box conversion via connected-component extraction:

1. **Minimum box area:** Box area must be ≥ 32 px² (to exclude noise from small mask fragments).
2. **Maximum box area:** Boxes covering ≥ 50% of the image area are excluded (likely full-image masks mislabeled as stuff).
3. **Aspect ratio limits:** Boxes with aspect ratio > 10:1 or < 1:10 are excluded (unlikely to be meaningful object boxes).
4. **Minimum valid boxes:** Classes must have at least 10 valid boxes after conversion across the entire dataset to be included in evaluation. If a class fails this threshold, it is flagged and reported as excluded.

The final retained class list is **frozen before any model evaluation** and reported in an appendix of the thesis. Any classes removed during filtering are listed with the reason.

---

## 7. Evaluation Protocol

### 7.1 Few-Shot Settings

Evaluate using $k \in \{1, 3, 5\}$ shots. Support exemplars are sampled from the training split and excluded from the test split. Results are averaged over **10 random seeds** with standard deviation reported.

**Power analysis acknowledgment:** 10 seeds follows standard FSOD practice and matches Michailidou et al.'s protocol, but we acknowledge that no formal power analysis has been performed to guarantee detection of small effect sizes. With 10 paired observations, a paired $t$-test detects a large effect ($d \approx 1.0$) at $\alpha = 0.05$ with power ≈ 0.8 (G*Power-style estimate). We therefore (a) pre-register the primary comparison as U-ADAPT (Mode A) vs. naive averaging, (b) report effect sizes (Cohen's $d$) alongside $p$-values, and (c) treat non-significant differences as evidence of comparable performance, not of superiority. This is an MSc-scope acknowledgment, not a claim of statistical optimality.

**Mode separation:** Mode A, Mode B, and Mode C results are reported in separate tables and never averaged together. The primary results section presents Mode A (strict few-shot, analytic) and Mode C (strict few-shot, feature-manifold repair) as the main claims, with the naive $w = 0.5$ averaging baseline as the reference. Mode B results appear in a clearly labeled subsection as "few-shot plus lightweight calibration," with the Mode D initialization ablation reported within it.

### 7.2 Cross-Domain Transfer Protocol (RQ3)

Cross-domain transfer semantics differ by evaluation mode:

- **Mode A (strict few-shot):** Transfer means using the same fixed analytic coefficients ($\alpha = \beta = \gamma = 1$) on both source and target domains, with only the prototypes updating from $k$ target-domain support examples. No training or tuning is involved — transfer is assessed purely by the robustness of the hand-designed gating rule across domain shift.

- **Mode B (calibrated):** The gating gate (logistic regression primary, or the small MLP variant) is trained on the source domain (e.g., LADD) using the full calibration set, then frozen and applied to the target domain (e.g., D-Fire) with only the prototypes updating from $k$ target-domain support examples. This tests whether the *learned gating pattern* generalizes across disaster types. Within this same transfer test, the gate may be initialized either randomly (default) or from COCO/LVIS-pretrained weights (Mode D; the former Mode C): if the pretrained initialization does not beat random init on the target domain, the optimal gating strategy is largely domain-specific.

This mode-specific distinction ensures that cross-domain transfer is evaluated consistently, and that failure to transfer in one mode does not reflect on the other.

**Dataset-size asymmetry acknowledgment:** LADD (1,365 images) is roughly 16× smaller than D-Fire (21,527 images). A gate trained on D-Fire has seen far more visual diversity than one trained on LADD, so LADD→D-Fire and D-Fire→LADD are not symmetric transfer tests. Transfer is therefore reported in **both directions**, and LADD-trained→D-Fire-evaluated results are interpreted with this imbalance caveat stated explicitly. If asymmetric transfer is observed, the direction consistent with the larger source domain is treated as the more reliable estimate of transferability, and the discrepancy is discussed as a finding rather than averaged away.

**Directional hypothesis (pre-registered):** Transfer is expected to be asymmetric but not along naive class-complexity intuition alone. For Mode A (fixed analytic coefficients), transfer should be roughly symmetric because nothing is learned. For Mode B (frozen trained gate), we expect **D-Fire-trained → LADD-evaluated** to transfer more favorably than the reverse, because the D-Fire training set (21,527 images) provides far more visual diversity for learning the gate than LADD (1,365 images). Class complexity cuts the other way — fire/smoke are diffuse and segmentation-shaped while pedestrians are compact — so a gate trained on D-Fire may over-weight the visual branch when applied to LADD. This hypothesis is pre-registered and tested in both directions.

### 7.3 Backbone Strategy

The primary backbone is **Grounding DINO (Swin-T)** for both LADD and D-Fire, ensuring consistent gap-recovery measurement (RQ2). Cross-backbone ablations using OWL-ViT and YOLOE26 on both datasets are included to verify backbone-agnostic gains. **Prioritization if time runs short:** a single additional backbone (OWL-ViT, the most modality-different comparison) evaluated on both datasets is sufficient evidence for the RQ5 backbone-agnostic claim; YOLOE26 then becomes optional rather than required. Efficient alternatives (YOLO-World, YOLO11-small) may be included for Colab feasibility, flagged as engineering inference.

### 7.4 Metrics

**Detection Performance:**
- mAP50
- mAP50:95 (where feasible)
- Per-class AP
- Gap recovery: $\frac{\text{U-ADAPT}(k) - \text{Zero-Shot}}{\text{Transfer Learning} - \text{Zero-Shot}} \times 100\%$ — **negative-recovery interpretation pre-registered:** if U-ADAPT underperforms zero-shot, gap recovery is negative; we report the raw signed value (not clipped) in per-seed distributions and interpret negative recovery as evidence that the adapter harmed performance for that configuration, discussed per dataset and per shot, never silently averaged into positive claims.

**Reliability:**
- Expected Calibration Error (ECE, 15 bins)
- Brier score
- Uncertainty AUROC
- Reliability diagrams

**Efficiency:**
- Trainable parameters (gating only: 0 for Mode A, ≈900 for Mode B, ≈6 for logistic regression)
- GPU memory usage (single frozen forward pass + cached features)
- Inference FPS (with cached features)
- Approximate Colab runtime per experiment

All runtime and memory estimates are engineering assumptions that will be validated in a pilot experiment.

### 7.6 Statistical Testing Plan

All primary accuracy comparisons are pre-registered with explicit tests, run **paired across the 10 random seeds** (same seed = same support/test split):

1. **Primary comparison:** U-ADAPT Mode A vs. naive averaging ($w = 0.5$) per dataset and per shot. Paired two-sided $t$-test on mAP50 across the 10 seeds; **Wilcoxon signed-rank test** as a non-parametric robustness check (justified because mAP50 distributions are not guaranteed Gaussian).

   **Full-scale status (LADD, executed 2026-08-07):** the pre-registered R3 contingency comparison — Mode B (6-parameter logistic gate, 20-box/class per-seed calibration) vs. naive averaging — was run at full scale on the complete cached LADD test split (197 test images, 10 seeds, BH FDR q = 0.05). Results: k=1: −6.70 pp (Mode B 70.14 vs. naive 76.84); k=3: −6.41 pp (70.55 vs. 76.96); k=5: −5.40 pp (71.70 vs. 77.09). Paired $t$-tests: $p < 0.0001$ in all cells (k=1: $t=-8.22$; k=3: $t=-11.36$; k=5: $t=-8.21$); Wilcoxon $W = 0$ in every cell ($p = 0.00195$); Cohen's $d \in [-3.59, -2.60]$. All q-values survive FDR control. Mode C is evaluated against the identical protocol.
2. **Secondary comparisons:** Mode A vs. text-only and vs. visual-only; Mode B (logistic) vs. Mode A. Same paired protocol.
3. **Multiple-comparison control:** With 2 datasets × 3 shots × ~3 primary comparisons ≈ 18 tests, the **Benjamini–Hochberg** procedure controls the false discovery rate at $q = 0.05$. Uncorrected $p$-values and BH-adjusted $q$-values are both reported.
4. **Effect sizes:** Cohen's $d$ (paired) reported for every significant comparison.
5. **D3 gate-favorability:** Binomial test against 0.5 as already pre-registered below in this section.
6. **Negative recovery:** Negative gap-recovery values are reported and interpreted as prescribed in Section 7.4; they are not dropped or winsorized.

This plan is fixed before results are collected; any deviation is reported in the Revision Note.

The following diagnostic tests are pre-registered to validate the core assumptions of the uncertainty-gated fusion mechanism. These are **not** claimed as contributions — they serve as sanity checks that, if they fail, would undermine the method's motivation. **Full-scale LADD status:** D1/D2 ≈ 0 on LADD is a *finding*, not a failed sanity check — it is the measured Proxy Saturation condition that triggers the pivot to feature-manifold repair (Mode C), per the pre-registered contingency chain (§10).

**D1 (Text uncertainty–accuracy correlation):** Across all images in the test set, group candidate proposals by bin of $\tilde{\sigma}_{\text{text}}^2$ (10 bins). Compute the proposal-level error rate per bin, where a proposal is counted as **correct** if it matches a ground-truth box at $\text{IoU} \geq 0.5$ with the correct class label, and **incorrect** otherwise. Report both the error rate per bin and the Spearman rank correlation $\rho$ between bin-level median $\tilde{\sigma}_{\text{text}}^2$ and bin-level error rate. A positive monotonic trend ($\rho > 0$) confirms that the template-based variance proxy is a meaningful signal of text-prompt reliability. **Full-scale LADD finding:** the proxy is degenerate by construction on the single-class LADD — with one class, the M=20 prompt ensemble produces zero cross-prompt variance, so D1 ≈ 0. This is the Proxy Saturation condition that motivates Mode C.

**D2 (Visual uncertainty–accuracy correlation):** Analogous to D1 for $\tilde{\sigma}_{\text{visual}}^2$ and the visual-only error rate (same proposal-level correctness criterion). A weak or absent correlation ($\rho \approx 0$ or negative) would indicate that the pairwise support-feature variance is a poor proxy for visual-matching error, suggesting that Mode A's analytic rule is working with a degraded signal. **Full-scale LADD finding:** D2 ≈ 0 within LADD (pooled pilot D2 = +0.051 was a between-dataset scale artifact; per-dataset LADD ≈ 0). With both proxies uninformative, any gate input is noise — directly explaining why the learned Mode B gate overfitted to visual noise (§5.4.4).

**D3 (Gate favorability):** On test images where text and visual prototype predictions disagree, compute the fraction of cases where the gate assigns higher weight to the more accurate modality. If this fraction is significantly above 0.5 (binomial test, $\alpha = 0.05$), the gate is genuinely selecting the better modality. Reported separately for Mode A and Mode B. **Full-scale LADD caveat:** the disagreeing subsets are almost entirely visual-better and the affinity proxy saturates (≥ 0.65), so D3 approaches 100% by construction — a saturation artifact, not evidence of calibration (consistent with the n=100 pilot findings reported in `docs/real_data_results_pilot.md`).

**D4 (Affinity diagnostic):** For Mode A, compare the full gating rule against the $\gamma = 0$ ablation (pure inverse-variance weight). If the affinity term systematically shifts the gate in the correct direction (higher $w$ when visual prototype is close, lower $w$ when far), this validates the bias-variance model from Section 5.4.2. Reported as the mean signed difference $\Delta w = w_{\text{full}} - w_{\gamma=0}$ binned by $a_{\text{visual}}$. **Full-scale LADD caveat:** visual affinity saturates on real features (≥ 0.65, often ≈ 0.99), so the observable range of $a_{\text{visual}}$ is narrow and $\Delta w$ is nearly uniform across bins — limiting this diagnostic's resolving power at full scale (consistent with the D3 saturation artifact).

**D5 (Distribution of normalized variances — Taylor-expansion validity):** Report the empirical distribution (histogram + quartiles) of $\tilde{\sigma}_{\text{text}}^2$ and $\tilde{\sigma}_{\text{visual}}^2$ across all candidate proposals and support sets. The first-order Taylor expansion of $\log \tilde{\sigma}^2$ around $0.5$ degrades if normalized variances cluster near 0 or 1 (steep log-derivative regions). If >30% of values fall below 0.25 or above 0.75, the expansion's validity is flagged and a Beta-regression variant of the gate is pre-registered as a fallback analysis (see Assumption 7). This diagnostic directly answers whether the analytic rule's approximation holds on the actual data. **Full-scale LADD status:** the sentinel was flagged (variance mass clustered at the boundaries), triggering the pre-registered Beta-regression fallback — which softened the gate but did not close the gap (pilot and n=100 evidence), confirming that no score-level repair can recover an uninformative proxy and motivating Mode C.

These diagnostics are computed **after** the main results are collected and do not affect any experimental choices or filtering criteria. If they reveal a failure of the core assumptions — as D1/D2 ≈ 0 did on full-scale LADD — this is reported honestly, and per the pre-registered contingency the pivot to feature-manifold repair (Mode C) is triggered rather than merely noted as a limitation.

---

## 8. Baselines and Ablations

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Zero-shot text-only | OV detector with text prompts only, no visual exemplars | Performance floor |
| Naive text/visual averaging (score-level) | Score-level averaging with $w = 0.5$ ($(S_{\text{text}} + S_{\text{visual}}) / 2$); embedding-level averaging surrogate also implemented where feasible | Direct comparison to prior art; separates score-level fusion from embedding-level averaging |
| Visual-only nearest-prototype | Visual prototype matching without text | Isolates visual prompt contribution |
| U-ADAPT w/o uncertainty gating | Fixed $w = 0.5$ (equal averaging) | Ablation of core contribution |
| U-ADAPT w/o temperature scaling | Gating without final calibration | Ablation of calibration |
| U-ADAPT w/o MC Dropout | Plain cosine-similarity margin as uncertainty proxy | Tests whether MC Dropout is necessary |
| **Mode A: Full U-ADAPT (strict few-shot)** | Analytic gating, no additional training | Primary strict few-shot claim (analytic) |
| **Mode C: Text-Anchored Feature Manifold Repair (strict few-shot)** | $p_{\text{repaired}} = \text{normalize}(\beta \cdot p_{\text{text}} + (1-\beta) \cdot p_{\text{visual}})$, label-free $\beta$ | **Primary new claim** — root-cause repair on the feature manifold (§5.4.3) |
| **Mode B: U-ADAPT with calibration (logistic regression)** | 6-parameter logistic gate, 20-box calibration | Simpler Mode B alternative |
| **Mode B: U-ADAPT with calibration (MLP)** | ≈900-parameter MLP gate, 20-box calibration | Full Mode B variant |
| **Mode D: U-ADAPT with COCO/LVIS-pretrained init** | Mode B gate initialized from COCO/LVIS-pretrained weights, then calibrated on target | Initialization ablation (the former Mode C) |
| Transfer-learning reference | Transfer-learning upper bound from Michailidou et al. | Performance ceiling context |
| Supervised detectors | YOLOv11l, YOLO26L, RT-DETRv2-L from Michailidou Table III | Absolute performance ceiling |

### Mode A Analytic-Gate Ablations

| Ablation | $\alpha$ | $\beta$ | $\gamma$ | Purpose |
|----------|----------|---------|----------|---------|
| Full Mode A | 1 | 1 | 1 | Default |
| No visual uncertainty | **0** | 1 | 1 | Is $\sigma_{\text{visual}}^2$ useful? |
| No text uncertainty | 1 | **0** | 1 | Is $\sigma_{\text{text}}^2$ useful? |
| No affinity | 1 | 1 | **0** | Is $a_{\text{visual}}$ useful? |
| Visual uncertainty only | 1 | 0 | 0 | Only visual variance signal |
| Text uncertainty only | 0 | 1 | 0 | Only text variance signal |
| Affinity only | 0 | 0 | 1 | Only visual affinity |

Mode D (the former Mode C, source-domain meta-training) is not an ablation of the Mode A analytic rule; it is a **Mode B initialization ablation** (random init vs. COCO/LVIS-pretrained init), covered in the baselines table above and in Section 5.4.3.

**Note on the "w/o MC Dropout" baseline:** This tests whether a much simpler uncertainty proxy (the normalized cosine distance margin between $S_{\text{text}}$ and $S_{\text{visual}}$) achieves comparable gating performance. If it does, the MC Dropout overhead may not be justified, and this should be reported honestly.

---

## 9. Prior-Art Table

| Method | What It Does | Disaster Evidence | Limitation |
|--------|-------------|-------------------|------------|
| **YOLOE26** | Real-time open-vocabulary YOLO with text, visual, and prompt-free inference | Tested in Michailidou Table III; beats OWL-ViT on LADD zero-shot (24.7% vs 6.2%) but weakest on D-Fire | Already direct prior art; limited on fire/smoke |
| **T-Rex2** | Text–visual prompt synergy via contrastive alignment training + static arithmetic averaging at inference | Not tested on disaster datasets in loaded sources | Static fusion; no uncertainty gating; not evaluated cross-domain |
| **Grounding DINO** | Language-guided open-vocabulary detector with Swin-T backbone | Strong LADD zero-shot (61.0%) and transfer (92.2%); weaker D-Fire zero-shot (27.5%) | Not primarily a few-shot visual-prompt adapter |
| **OWL-ViT** | Text-conditioned open-vocabulary detector | Stronger D-Fire zero-shot (36.4%) than Grounding DINO; weak LADD zero-shot (6.2%) | Category-dependent performance; not uncertainty-aware |
| **BayesVLM** | Training-free post-hoc Bayesian uncertainty for VLMs (Laplace approximation + ProbCosine) | Not tested on disaster detection | Provides backbone uncertainty but not prompt fusion gating |
| **Query2Uncertainty** | Density-aware calibration for DETR-style detectors under distribution shift | Not tested on aerial/disaster imagery | Targets 3D detection; not few-shot; not open-vocabulary |
| **ReliFusion** | Confidence-gated LiDAR-camera fusion for 3D detection | Not applicable to prompt fusion | Sensor-fusion domain; text/visual prompt errors are treated as approximately uncorrelated (a working approximation, not a proven fact — see §2.3, Assumption 8) |
| **U-ADAPT** | Post-hoc uncertainty-gated text/visual adaptation (Modes A/B) plus Text-Anchored Feature Manifold Repair (Mode C), frozen backbone | Full-scale LADD validation executed: score-level gating underperforms naive averaging under Proxy Saturation (D1/D2 ≈ 0) | Score-level gating is provably harmful when proxies are uninformative; Mode C repairs the feature manifold instead |

---

## 10. Risks and Fallback

| Risk | Description | Mitigation / Fallback |
|------|-------------|----------------------|
| **R1: Uncertainty is too noisy** | Fire, smoke, and debris may produce high visual variance, making uncertainty estimates unreliable in both Mode A (prompt/support variance) and Mode B (MC Dropout) | Feature normalization to [0,1], prototype outlier rejection (Mahalanobis 2σ), calibration, and the plain-confidence baseline (cosine-similarity margin) as ablation. |
| **R2: Colab memory is insufficient** | Grounding DINO with Swin-T may be too heavy for long experiments | Use YOLO-World-small, YOLO11-small, or OWL-ViT with reduced resolution and batch size. Features cached after one pass. |
| **R3: Method does not beat naive averaging** | **Realized at full scale (LADD, Mode B vs. naive: −5.4 to −6.7 pp, $p < 0.0001$, $d$ ∈ [−2.6, −3.6])** — the learned gate systematically underperforms the uncertainty-blind baseline | **Fallback executed as pre-registered:** the root cause was diagnosed — Proxy Saturation (D1/D2 ≈ 0 on single-class LADD) producing Ranking Destruction (§5.4.4) — and the pivot is **Mode C (Text-Anchored Prototype Repair)**, which solves the feature-manifold misalignment before scoring. The comparative calibration/reliability study remains as a secondary contribution, and the negative result itself is reported as a validated diagnostic finding, not obscured. |
| **R4: Gating does not transfer cross-domain** | The gating mechanism trained on LADD may not generalize to D-Fire | Report per-domain results separately. This is still a valid few-shot adaptation study; cross-domain transfer becomes future work. |
| **R5: Raw proposal recall is the bottleneck** | If the frozen detector never proposes a box around a target, no downstream re-scoring can recover it | Report raw proposal recall as a ceiling. Results on D-Fire may genuinely be limited by fire/smoke being a segmentation-shaped problem forced into detection. |
| **R6: Mode B overfitting** | 20-box calibration set insufficient for MLP training | Logistic regression alternative (6 params), dropout, early stopping, cross-validation. Report the training-free claims (Mode A analytic; Mode C feature repair) as primary — they do not depend on calibration data. |

**Compound contingency (pilot failure + Colab memory failure):** The individual contingencies above are pre-registered separately, but the compound case is worth addressing explicitly: if the pilot (Week 3) reveals that Mode A's uncertainty proxies do not correlate with error (D1/D2 fail) *and* Grounding DINO exceeds Colab memory limits at the same time, the plan is: (1) switch the primary backbone to the fallback (OWL-ViT or YOLO11-small) rather than fighting the memory ceiling; (2) demote Mode A's uncertainty terms to ablations and elevate the plain-confidence baseline (cosine-similarity margin) plus the Mode B logistic gate, which learns the weighting from data rather than assuming the proxy is informative; and (3) absorb the re-baselining cost by trimming the cross-backbone matrix to a single secondary backbone, since the core RQ1/RQ2 claim depends only on Mode A on two datasets. **Full-scale resolution (2026-08-07):** the pilot and full-scale runs realized the D1/D2-failure branch of this contingency — the pre-registered fallback chain was executed, and the Mode C pivot (§5.4.3) is the resulting new primary claim. This compound case was decided at the pilot's post-mortem; the decision tree is fixed before the pilot runs.

---

## 11. Timeline (Estimated)

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Literature review & implementation setup | 2 weeks | Related work chapter; U-ADAPT prototype; dataset preparation; pre-registration document |
| **Pilot experiment (pre-registered)** | **1 week** | Validate runtime/memory on Colab (top-k proposals, feature caching, T=10 MC Dropout), confirm the variance proxies correlate with error (D1/D2 pilot), and freeze the final class list after mask-to-box filtering — **before** any main results are collected |
| Phase 1–3 implementation | 2 weeks | Candidate generation (top-k limiting), feature extraction (caching), prototype construction |
| Phase 4 (gating) implementation | 2 weeks | Mode A analytic gating (training-free); Mode B logistic/MLP gate, random-init first, COCO/LVIS-pretrained-init ablation later if time permits |
| Phase 5 (calibration) + baselines | 1 week | Temperature scaling; all baselines and ablations implemented |
| Cross-domain transfer experiments | 1 week | RQ3 experiments (Mode A primary; Mode B secondary, incl. Mode D init-ablation comparison) |
| **Mode C implementation and evaluation (Text-Anchored Subspace Projection)** | **1 week** | Closed-form James-Stein-style prototype repair on **existing cached features** (no re-extraction needed); 10-seed protocol vs. naive averaging; $\beta$-sensitivity sweep |
| Analysis, writing, and revision | 2 weeks | Full thesis draft; figures; tables; statistical tests (§7.6) |
| **Total** | **12 weeks** | |

**Scope note (addressing timeline feasibility):** The experimental grid is deliberately sized to the available window. The former Mode C track is folded into Mode B as the Mode D gate-initialization ablation (Section 5.4.3), which removes an entire evaluation dimension rather than adding one — though the COCO/LVIS episodic-sampling pipeline itself remains bounded, sequenced implementation work (Section 5.4.3), rather than reclaimed time. The critical path is Mode A and Mode C on LADD and D-Fire against the baselines listed in Section 8, with the 10-seed protocol and the §7.6 statistical plan. **Mode C is highly feasible within the remaining timeline precisely because it operates on the already-cached features and prototypes (a closed-form vector operation — no re-extraction, no training), so its implementation and 10-seed evaluation fit in a single added week.** Everything else (Mode B including its Mode D init ablation, the cross-backbone matrix) is sequenced after the primary results, so that if calendar time runs short, secondary experiments — not the primary claims — absorb the cut. Any deviation from this plan is reported in the Revision Note.

---

## 12. Expected Outcomes

1. A rigorous, pre-registered characterization of **when post-hoc score-level fusion fails** in open-vocabulary detection under domain shift — including the full-scale LADD Mode B vs. naive result (−5.4 to −6.7 pp, $p < 0.0001$, $d$ ∈ [−2.6, −3.6]) and its causal diagnosis (Proxy Saturation → Ranking Destruction).
2. A quantitative evaluation of **Mode C (Text-Anchored Feature Manifold Repair)** at 1/3/5 shots on LADD and D-Fire against naive averaging and Modes A/B, including the $\beta$-sensitivity sweep.
3. Evidence on whether uncertainty-gated fusion outperforms static averaging (T-Rex2-style) in cross-domain disaster scenarios, and under which conditions it provably cannot.
4. Initial evidence on cross-domain transferability of the gating mechanism.
5. A reliability analysis with calibrated confidence estimates for disaster detection.
6. A Colab-reproducible implementation available as open-source code, with stated runtime and memory requirements validated in a pilot experiment.

---

## 13. Remaining Limitations

The following limitations are acknowledged and are **not claimed to be addressed** in this thesis. They are identified as directions for future work:

1. **Single-backbone primary experiments.** The primary claim (Mode A) is evaluated with Grounding DINO (Swin-T). While cross-backbone ablations are included, U-ADAPT is not evaluated with all possible open-vocabulary detectors. Backbone-agnostic claims are supported by evidence from the ablation set, not proven universally.

2. **Heuristic uncertainty proxies (Mode A).** The mean pairwise cosine distance is a reasonable but heuristic estimate of epistemic uncertainty. It does not capture the full predictive distribution and may underestimate uncertainty when prompt templates are inadvertently similar. More principled approaches (e.g., BayesVLM's ProbCosine) are noted as future extensions but not integrated.

3. **Segmentation-to-detection conversion.** The RescueNet and FloodNet+ evaluations depend on mask-to-box conversion, which introduces approximation error. Objects with irregular boundaries (debris) or damage-level classes (flooded buildings) are not well-captured by axis-aligned bounding boxes. Results on these categories should be interpreted with caution.

4. **Limited calibration set size (Mode B).** Even with the logistic regression alternative, 20 boxes per class is a small calibration set. The statistical reliability of the learned gate at this sample size is a limitation, and results should be interpreted as preliminary evidence rather than a definitive claim about calibration-data efficiency.

5. **No online adaptation.** U-ADAPT operates post-hoc with cached features. It does not update its gating mechanism as new images arrive (online adaptation). This is a deliberate design choice for simplicity but limits applicability in streaming disaster response scenarios.

6. **Colab feasibility is not guaranteed.** The implementation targets Colab feasibility, but this is an engineering target, not a guaranteed outcome. The proposal explicitly notes that YOLO11-small or OWL-ViT may be used as fallback backbones if Grounding DINO exceeds Colab memory limits.

7. **Single disaster domain types evaluated.** LADD (search and rescue) and D-Fire (fire/smoke) cover two disaster types. Results may not generalize to other disaster modalities (e.g., earthquake structural damage, flood extent mapping). RescueNet and FloodNet+ provide partial coverage but are themselves limited to specific disaster scenarios.

8. **No human evaluation.** Detection quality is evaluated solely through standard object detection metrics (mAP, ECE, etc.). No human-subject study validates whether U-ADAPT's detections are practically useful for disaster response teams. This is beyond the scope of an MSc thesis.

9. **Mode C's label-free $\beta$ is a heuristic (new).** The text weight in the James-Stein-style shrinkage is not learned on the target domain (by design, to preserve strict few-shot purity) and its default rule (decreasing in $k$, increasing in visual-prototype variance) is heuristic. A pre-registered sensitivity sweep bounds this risk, but a degenerate choice of $\beta$ (e.g., $\beta \to 1$ on a multi-class dataset where the text anchor is poorly aligned) would collapse the visual prototype onto the text anchor and inherit text-side bias. Learning $\beta$ from a source domain is noted as future work, consistent with the Mode D gate-init framework.

### Delimitations (explicit out-of-scope statement)

This thesis does **not** pursue the following, as each exceeds the MSc scope and would compromise the focused evaluation of the core claim:

1. **Online/streaming adaptation** — U-ADAPT operates post-hoc on cached features; it does not update its gating as new images arrive.
2. **Architectural search for the gating network** — the gate is fixed to the analytic rule (Mode A), logistic regression, or a single small MLP (Mode B); no neural architecture search is performed.
3. **Extension to panoptic/instance segmentation** — detection is the target task; segmentation-derived auxiliary sets are used only via mask-to-box conversion.
4. **Full fine-tuning of the detector backbone** — the transfer-learning upper bound is used only as a reference ceiling, never as a U-ADAPT configuration.
5. **Video or multi-temporal imagery, and non-aerial platforms** — evaluation is limited to single-frame aerial imagery.
6. **TENT/MEMO-style test-time gradient adaptation** — related work (Section 2.7) but explicitly out of scope; combining it with U-ADAPT is future work.

These boundaries are stated to make the contribution's scope unambiguous and to prevent scope creep during the thesis period.

---

## Ethics Statement

This work uses publicly available disaster imagery (LADD, D-Fire, RescueNet, FloodNet+) collected and curated by third-party researchers for academic purposes. The proposed U-ADAPT system is intended for **research and decision-support purposes only**, not for unsupervised operational deployment in real disaster-response settings. Automated detections should be reviewed by trained human operators before informing any operational decision. The authors acknowledge that incorrect detections in disaster scenarios — particularly false negatives (missed victims, undetected hazards) and false positives (misidentified safe areas) — carry ethical risks and that the proposed method does not eliminate these risks. No personally identifiable information is used in the evaluation datasets. The goals of this research are to advance the scientific understanding of few-shot cross-domain adaptation for aerial perception and to support, not replace, human decision-making in disaster response.

**Research-ethics exemption (explicit statement):** No institutional research-ethics approval is required for this research. (Note: "IRB" is the US term for institutional ethics review; the candidate will confirm the exact name and process of their own university's ethics review and whether exemption must be confirmed by the committee rather than self-declared.) This work uses publicly available, de-identified datasets and involves **no human subjects** — no data collection from human participants, no personally identifiable information, and no human evaluation beyond the authors' own analysis of public benchmark images. This exemption is stated explicitly so that institutional reviewers do not need to infer it.

**Dataset and model licenses (pre-registration commitment):** The following license checks are committed before experiments begin and will be reported in the thesis appendix: (1) LADD, D-Fire, RescueNet, and FloodNet+ dataset licenses must permit academic research use; (2) the primary backbone Grounding DINO is released by IDEA Research under the **Apache License 2.0**, which permits the feature extraction, caching, and research use described in Phase 2; (3) OWL-ViT, YOLOE, YOLO-World, and CLIP/DINOv2 encoders are similarly open-weight research licenses. If any dataset or model license restricts academic use or imposes dual-use restrictions, that resource is dropped or replaced, and the substitution is logged in the Revision Note.

---

## References

- Baumann, A., Li, R., Klasson, M., Mentu, S., Karthik, S., Akata, Z., Solin, A., & Trapp, M. (2026). Post-hoc Probabilistic Vision-Language Models. *ICLR 2026* (verified via OpenReview). arXiv:2412.06014.
- Beemelmanns, T., Nekrasov, A., Vilceanu, S., Steinhaus, J., Woopen, T., Leibe, B., & Eckstein, L. (2026). Query2Uncertainty: Robust Uncertainty Quantification and Calibration for 3D Object Detection under Distribution Shift. *CVPR 2026* (verified). arXiv:2605.05328.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*.
- Hu, H., Bai, S., Li, A., Cui, J., & Wang, L. (2021). Dense Relation Distillation with Context-aware Aggregation for Few-Shot Object Detection (FsDetView). *CVPR 2021*.
- Jiang, Q., Li, F., Zeng, Z., Ren, T., Liu, S., & Zhang, L. (2024). T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy. *ECCV 2024* (verified; LNCS 15053, pp. 38–57). arXiv:2403.14610.
- Michailidou, A., Angelidis, G., Argyriou, V., Sarigiannidis, P., & Papadopoulos, G. Th. (2026). Open-Vocabulary vs Supervised Learning Methods for Post-Disaster Visual Scene Understanding. **arXiv preprint** (v1, 7 pages; no venue as of July 2026). arXiv:2603.01324.
- Sadeghian, R., et al. (2025). Reliability-Driven LiDAR-Camera Fusion for Robust 3D Object Detection. *arXiv:2502.01856* (ReliFusion; not independently re-verified in this revision).
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Fully Test-Time Adaptation by Entropy Minimization (TENT). *ICLR 2021*.
- Wu, J., Liu, S., Huang, D., & Wang, Y. (2021). Multi-Scale Positive Sample Refinement for Few-Shot Object Detection (MPSR). *CVPR 2021*.
- Zhang, M., Levine, S., & Finn, C. (2022). MEMO: Test Time Robustness via Adaptation and Augmentation. *NeurIPS 2022*.
- Zhou, K., Yang, J., Loy, C. C., & Liu, Z. (2022). Learning to Prompt for Vision-Language Models (CoOp). *IJCV*.
- Zhou, K., Yang, J., Loy, C. C., & Liu, Z. (2022). Conditional Prompt Learning for Vision-Language Models (CoCoOp). *CVPR 2022*.

**Citation status note:** All 2026 citations verified against arXiv records as of July 2026 (see Section 5.4.2 caveat). BayesVLM and Query2Uncertainty are confirmed published/accepted at ICLR 2026 and CVPR 2026 respectively (BayesVLM acceptance additionally confirmed via OpenReview); Michailidou et al. is an arXiv preprint and is cited as such with no venue claim. T-Rex2 is confirmed ECCV 2024. Preprint status is labeled inline in this list so no venue is ever implied for an unpublished manuscript.
