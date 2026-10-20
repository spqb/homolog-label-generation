# Comparing missing label treatments in conditional sequence generation

This experiment tests whether exact marginalization over missing annotations improves label-conditioned sequence generation relative to the original zero-label treatment in annaDCA. It addresses the reviewer's question about whether unlabeled sequences can contribute to learning label–hidden couplings through posterior label probabilities.

The comparison changes only the treatment of missing labels within the same installed annaDCA 0.2.3 release. The document describes the protocol implemented by `src/pipeline/missing_labels.py` and `src/analysis/missing_labels/analyze.py`; it does not report an observed improvement or a completed full experiment.

## Experimental design

Two conditioned RBMs are trained for each selected RR split. The **zero** arm uses the default training behavior. The **posterior** arm adds `--marginalize_missing_labels`. Both arms load the same locked PyPI release, rather than comparing two package versions.

Each pair uses the same aligned sequences, observed annotations, architecture, optimization settings, training budget, and random seed. Training and sampling seeds equal the split seed. This controls the initial random seed; it does not imply identical subsequent trajectories once the updates differ. Across the three splits, variation combines differences in data composition and model randomness, rather than isolating either source of variation.

| Setting | Default |
| --- | --- |
| Family and split threshold | RR, t1 = 0.4 |
| Nominal annotated training size | 100 |
| Split seeds | 1, 2, and 3 |
| Models | Zero and posterior for each split, six models total |
| Hidden units | 500 |
| Learning rate | 0.005 |
| Training budget | 10,000 parameter updates |
| Persistent chains and batch size | 5,000 |
| Gibbs steps per training update | 10 |
| Sequence reweighting | Disabled |
| Training backend | Triton, CUDA, float32 |
| Conditioned generation | 1,000 sequences per label |
| Gibbs steps for generation | 1,000 |

The runner calls the training budget `--updates` and passes it to annaDCA as `--nepochs`. This parameter counts gradient updates, not complete passes through the dataset. Unspecified annaDCA settings retain their local defaults, including centered gradients and label-term scaling of one.

## Data and annotation visibility

For each split, the source is `outputs/RR/t1_0.4/seed_<seed>/datasets_for_rbm/rbm_dataset_ntrain_100.csv`. Both arms read `sequence_align` as the aligned sequence and `train_only` as the training label. Blank `train_only` entries denote missing annotations. The `true_label` column is used for evaluation and class-coverage checks, but is not passed as the training label.

The inspected input files contain 96 observed annotations each, despite the nominal size of 100. Split 1 contains 27,568 sequences, split 2 contains 21,208, and split 3 contains 22,072. All eight reference classes occur among the observed annotations. The runner records the actual counts in each split's manifest so these values can be checked against the files used for a run.

The evaluation reference is the same sequence collection used for training, grouped by its reference labels. Most of those labels were hidden during training. Consequently, the experiment measures recovery of label-specific distributions in a transductive setting: sequences may have been seen without their annotations. It does not measure generation quality against an independent set of unseen natural sequences.

## Training objective and positive phase

Let v denote an aligned sequence, y a categorical label with K possible values, and h the binary hidden units. Write the joint model as p(v,y,h) = exp(−E(v,y,h))/Z. Let U_kj be the coupling between label k and hidden unit j, b_k the label bias, and a_j(v) the hidden field produced by the sequence and hidden bias before including the label. Define m_j(v,k) = sigmoid(a_j(v) + U_kj).

### Posterior marginalization

The target is the observed-data log likelihood, with equal sequence weights in this experiment:

$$
\mathcal L_{\mathrm{post}}(\theta)
=\frac{1}{N}\left[
\sum_{i\in L}\log p_\theta(v_i,y_i)
+\sum_{i\in U}\log\sum_{k=1}^{K}p_\theta(v_i,k)
\right].
$$

Here L and U are the annotated and unannotated sequence sets. For each unannotated sequence, the implementation computes the exact model posterior over the finite label vocabulary:

$$
\pi_k(v)=p_\theta(y=k\mid v)
=\operatorname{softmax}_k\left[b_k+\sum_j\operatorname{softplus}(a_j(v)+U_{kj})\right].
$$

The positive-phase sufficient statistics are then

$$
\mathbb E[\mathbf 1_{y=k}\mid v]=\pi_k(v),\qquad
\mathbb E[h_j\mid v]=\sum_k\pi_k(v)m_j(v,k),
$$

$$
\mathbb E[\mathbf 1_{y=k}h_j\mid v]=\pi_k(v)m_j(v,k).
$$

In particular, the label–hidden statistic is a joint expectation. Multiplying the posterior label vector by the averaged hidden vector would generally give a different answer. annaDCA 0.2.3 adds a conditional covariance correction to recover the joint expectation in both centered and uncentered updates. The original missing annotations remain unchanged in the dataset; posterior values are computed from the current parameters for the training batch.

For the ordinary uncentered likelihood gradient, each parameter update is its data-conditioned sufficient statistic minus the model statistic. For example,

$$
\frac{\partial\mathcal L_{\mathrm{post}}}{\partial U_{kj}}
=\frac{1}{N}\left[
\sum_{i\in L}\mathbf 1_{y_i=k}m_j(v_i,k)
+\sum_{i\in U}\pi_k(v_i)m_j(v_i,k)
\right]
-\mathbb E_{p_\theta(v,y,h)}[\mathbf 1_{y=k}h_j].
$$

The experiment uses the implementation's centered updates, with corresponding centering and bias corrections. The formula above identifies the underlying sufficient statistics. The missing-label positive phase is evaluated exactly at fixed model parameters, whereas the negative-phase model expectation is estimated using persistent contrastive divergence. The complete training gradient is therefore not an exact evaluation of the likelihood gradient.

### Original zero label treatment

For an unannotated sequence, the original mode substitutes an all-zero label vector. Its positive-phase hidden mean is sigmoid(a_j(v)). The uncentered label and label–hidden data statistics for that row are zero, while the sequence, hidden, and sequence–hidden statistics still contribute.

This procedure does not sum over the categorical label values. To express the distinction as an objective, define A(v,k) = log ∑_h exp(−E(v,k,h)), and let A_0(v) be the same expression with the label vector replaced by zero. Ignoring numerical pseudocounts and expressing the underlying uncentered update, the original procedure corresponds to the surrogate

$$
\mathcal S_0(\theta)
=\frac{1}{N}\left[\sum_{i\in L}A(v_i,y_i)+\sum_{i\in U}A_0(v_i)\right]-\log Z_\theta.
$$

The zero vector is not one of the categorical label states in Z. This surrogate should therefore not be described as the observed-data likelihood obtained by marginalizing a missing categorical label. Marginalization replaces A_0(v) by log ∑_k exp(A(v,k)).

The negative phase remains active in the zero arm, and centered updates include centering corrections. Thus, saying that all label-parameter gradients are masked for unlabeled data would be inaccurate. The precise distinction is that the zero arm removes label-dependent terms from the unannotated positive-phase hidden field and supplies zero raw label statistics; the posterior arm supplies posterior-weighted joint statistics. Both arms leave observed-label treatment and negative-phase sampling unchanged in form.

## Conditional generation

After training, each arm generates the same number of sequences for every label. A separate `targets.csv` contains a complete target label for every requested sample. This is necessary because conditioned sampling rejects missing target labels. The sequence values in that file provide dimensions and alphabet information; the sampler initializes its own chains.

Each generated sequence is grouped by its requested label, not by a label predicted after generation. There is no PLM embedding or classifier inference in this comparison. Agreement with a requested class therefore means agreement with sequence statistics of that class, rather than experimentally demonstrated function or external classifier confirmation.

## Generation metrics

### Connected correlation recovery

For each split and reference label, the analysis estimates natural single-site frequencies f_i(a) and pair frequencies f_ij(a,b), and calculates connected correlations

$$
C_{ij}(a,b)=f_{ij}(a,b)-f_i(a)f_j(b).
$$

It estimates the corresponding quantities from sequences generated with that target label and reports Pearson correlation between natural and generated entries over distinct site pairs i < j and residue-state pairs. The implementation reuses the project's connected-correlation functions. Higher values indicate better recovery of the pattern of pairwise dependencies; Pearson correlation alone does not establish agreement in their absolute scale. Undefined correlations are retained as missing values rather than silently omitted from a split's macro-average.

### Symmetric KL divergence in PCA space

A two-component PCA is fit once per split on one-hot encodings of the natural reference sequences. Both generated collections are projected using that fitted PCA. A single 20 by 20 histogram grid is constructed per split, covering the natural points and both generated collections. The same bin edges are used for both arms and every label; points outside the natural range are not discarded.

For each label, normalized natural and generated histograms P and Q are compared using

$$
\mathrm{D}_{\mathrm{symKL}}(P,Q)=\tfrac12\mathrm{KL}(P\Vert Q)+\tfrac12\mathrm{KL}(Q\Vert P).
$$

Before normalization, `1e-10` is added to every histogram bin, as in the main generation analysis, so the divergence remains finite when only one distribution occupies a bin. Natural logarithms are used. Lower values indicate better overlap in this projection. This metric only assesses two-dimensional projected structure and depends on the pseudocount and histogram resolution; `--bins` allows a sensitivity check.

### Diversity and resemblance to natural sequences

For each split, arm, and requested class, the analysis also reports the fraction of unique generated sequences, mean pairwise identity within the generated class, mean nearest-natural-sequence identity, and fraction of generated sequences exactly matching a natural sequence. Identity is computed over all aligned sites, including gaps. Nearest matches are sought across the entire natural training CSV, not only within the requested class.

These diagnostics help identify reduced diversity or copying. They are interpreted together with distribution recovery; neither maximal diversity nor minimal natural-sequence identity is itself an optimization target.

## Aggregation and interpretation

The analysis first reports every metric separately for each label, arm, and split. It then takes an equal-weight average across labels within each split, preventing common classes from dominating the summary. Across splits it reports the mean, sample standard deviation, and valid count of these macro-averages.

The paired difference is always posterior minus zero. A positive difference in correlation recovery and a negative difference in symmetric KL favor posterior marginalization. The main figure connects the two arm values within each split in separate correlation and symmetric-KL panels. Per-label tables should accompany these summaries to reveal class-specific effects.

Three paired splits provide descriptive evidence with limited precision. Labels and generated sequences within one fitted model are not independent training replicates. The comparison does not establish statistical equivalence when differences are small, and it does not justify claiming superiority before the results are inspected. Report both improvements and degradations across labels and splits.

A short training budget may leave either arm unconverged. Assess runtime and stability with a pilot pair and, if necessary, repeat both arms at a longer matched budget in a separate output directory. The current runner does not automatically compare intermediate checkpoints or test Gibbs-chain mixing.

## Execution and reproducibility

Run the pilot from the repository root with `uv run --locked bash src/pipeline/run_experiments.sh --profile missing-labels --train-sizes 100 --seeds 1`. Running the same command without `--seeds 1` selects the default three splits. Training uses Triton by default and streams output to the terminal while preserving logs. `--no-triton` selects the PyTorch training backend.

The default output directory is `outputs/missing_labels/RR/t1_0.4/ntrain_<budget>/`. Each split has a manifest containing parameters, dataset and source hashes, actual label counts, and vocabulary. The analysis verifies that the reference CSV still matches its recorded hash.

When both arms already have nonempty checkpoints and sample files, the runner proceeds directly to analysis and preserves the original manifest, even if current command-line defaults or the installed release have changed. Existing checkpoints are not retrained. If only some stages are missing, the original configuration check still applies before those stages run. Completion markers are optional for reuse, and absent timing information remains unknown.

File presence does not prove convergence or completion of the requested update budget. When reusing models, verify the saved checkpoint's actual update count and original manifest before describing a pair as a matched 10,000-update comparison. The recorded settings for reused runs take precedence over the defaults listed in this document.

The analysis outputs `by_label.csv`, `by_seed.csv`, `summary.csv`, `paired_differences.csv`, `runtime.csv`, `analysis_config.json`, and the paired comparison figure in PDF and PNG formats. Analysis alone can be rerun with the `missing-labels` profile through `src/pipeline/run_experiments.sh`.

## Implementation sources

The protocol is implemented in `src/pipeline/missing_labels.py` and `src/analysis/missing_labels/analyze.py`, with usage instructions in this directory's [README.md](README.md). Connected-correlation evaluation is reused from `src/analysis/analyze_generated_samples.py`.

The training distinction is implemented in the annaDCA 0.2.3 modules `annadca/train.py`, `annadca/rbm/missing_labels.py`, and `annadca/rbm/categorical/grad.py`. Both arms use the same installed release. The posterior flag changes the missing-label positive phase and its sufficient statistics; it does not invoke an external annotation model.
