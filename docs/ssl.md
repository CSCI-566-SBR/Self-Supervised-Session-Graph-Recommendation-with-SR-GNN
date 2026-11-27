# SR-GNN SSL: Cross-Dataset Results and Interpretation

Five-seed evaluation found dataset-dependent effects from SR-GNN SSL. Identical-pair masking consistently recovered much of the unmasked Diginetica deficit, while Unmasked Recency consistently improved RetailRocket Recall@20.

These results do not establish a general MRR or robustness improvement. The five-seed comparisons are reported first; the broader seed-2028 ablations are retained below as supporting evidence.

## Five-seed results

The repeated evaluation is complete for four models on Diginetica and RetailRocket, using seeds 2024–2028. All 40 model–dataset–seed combinations have clean results and evaluations on the nine corrupted conditions.

Within each dataset, training and test checksums match across the compared models and seeds. Each model's recorded settings are unchanged apart from seed and experiment identity.

Values below are mean ± sample standard deviation across the five seeds, multiplied by 100. Corrupted scores first average the nine conditions within each seed; variation is then measured across seeds.

### Diginetica

| Model | Clean Recall@20 | Clean MRR@20 | Mean corrupted Recall@20 | Mean corrupted MRR@20 |
|---|---:|---:|---:|---:|
| SR-GNN | 49.575 ± 0.159 | 17.118 ± 0.037 | 46.379 ± 0.145 | 15.912 ± 0.042 |
| Unmasked Recency | 49.146 ± 0.083 | 16.795 ± 0.086 | 45.913 ± 0.066 | 15.546 ± 0.091 |
| Masked Recency | 49.461 ± 0.166 | 17.098 ± 0.038 | 46.245 ± 0.137 | 15.865 ± 0.028 |
| Masked Uniform | 49.524 ± 0.120 | 17.092 ± 0.013 | 46.300 ± 0.110 | 15.859 ± 0.017 |

Unmasked Recency reduced clean MRR@20 relative to SR-GNN in all five seeds. The paired mean difference was −0.323 percentage points (pp), and mean corrupted MRR was lower by 0.366 pp.

Masked Recency improved clean and mean corrupted MRR over Unmasked Recency in all five seeds. The paired mean improvements were 0.303 pp and 0.319 pp, respectively.

Masking therefore consistently recovered much of the unmasked deficit. It did not consistently improve on the backbone: Masked Recency's mean clean MRR was 0.020 pp below SR-GNN, and its mean corrupted MRR was lower in every seed.

The small clean-MRR advantage observed for Masked Recency at seed 2028 did not persist in the five-seed mean. The earlier Recall-retention advantage also disappeared: mean Recall degradation was 6.503%, versus 6.446% for SR-GNN.

### RetailRocket

| Model | Clean Recall@20 | Clean MRR@20 | Mean corrupted Recall@20 | Mean corrupted MRR@20 |
|---|---:|---:|---:|---:|
| SR-GNN | 56.115 ± 0.116 | 32.174 ± 0.035 | 53.862 ± 0.090 | 30.829 ± 0.047 |
| Unmasked Recency | 56.311 ± 0.179 | 32.172 ± 0.077 | 54.065 ± 0.206 | 30.824 ± 0.077 |
| Masked Recency | 56.269 ± 0.158 | 32.174 ± 0.057 | 54.022 ± 0.163 | 30.839 ± 0.068 |
| Masked Uniform | 56.266 ± 0.122 | 32.178 ± 0.053 | 54.012 ± 0.160 | 30.850 ± 0.062 |

Unmasked Recency improved clean and mean corrupted Recall@20 over SR-GNN in all five seeds. Paired mean gains were 0.196 pp and 0.203 pp, respectively.

This is the clearest repeatable positive result. It indicates that the correct item entered the top 20 slightly more often under both clean and corrupted inputs.

The corresponding MRR gain did not persist. Unmasked Recency's mean clean MRR difference was −0.002 pp, and its mean corrupted MRR difference was −0.006 pp. The seed-2028 MRR improvement was not representative of the five-seed average.

Masked variants also remained near SR-GNN on MRR, with mixed paired outcomes. Masked Uniform had the highest mean corrupted MRR at 30.850, versus 30.829 for SR-GNN, but exceeded the baseline in only three of five seeds.

### Robustness and recency weighting

Higher corrupted Recall does not automatically establish reduced sensitivity to noise. For Unmasked Recency on RetailRocket, mean relative Recall degradation improved in only two of five seeds, despite higher absolute mean corrupted Recall in every seed.

Its five-seed mean Recall degradation was 3.989%, compared with 4.014% for SR-GNN. Mean relative MRR degradation was slightly worse: 4.193% versus 4.180%.

On Diginetica, every SSL variant had worse mean relative MRR degradation than SR-GNN in every seed. The five-seed means were 7.436% for Unmasked Recency, 7.208% for Masked Recency, and 7.217% for Masked Uniform, versus 7.044% for SR-GNN.

Masked Uniform and Masked Recency showed small, mixed differences on both datasets. These results do not establish a consistent advantage from recency weighting over uniform sampling.

### Final-report interpretation

Across five seeds, contrastive augmentation showed dataset-dependent effects. Unmasked Recency consistently reduced Diginetica accuracy, while identical-pair masking recovered much of that deficit.

On RetailRocket, Unmasked Recency consistently improved clean and corrupted Recall@20, but provided no corresponding mean MRR improvement. Recency weighting and masking did not deliver a universal advantage.

The positive finding is a small, repeatable RetailRocket Recall gain. The diagnostic finding is consistent Diginetica masking recovery. Neither supports a claim of general accuracy or robustness superiority over SR-GNN.

Five seeds provide stronger descriptive evidence than a single initialization. No formal significance or equivalence tests have been performed; SASRec and secondary ablations remain single-seed references.


## Single-seed Diginetica evaluation context

SR-GNN and its SSL variants use the same backbone and recommendation training settings. The SSL variants add a contrastive objective with loss weight 0.01 and temperature 0.2. SASRec provides a separate sequential-model reference.

All models were evaluated on the same 60,411 test examples. Robustness was measured using deletion, substitution, and mixed corruption, each at 5%, 10%, and 20%.

The results use one training seed, 2028. Checkpoints were selected using validation MRR@20. These comparisons do not establish reliability across random seeds.

## Understanding the variants

Augmentation and identical-pair masking control different parts of SSL. Augmentation determines the two model inputs. Masking determines which paired sessions contribute to InfoNCE.

### How the augmented views are created

**Uniform edge deletion** samples eligible transition occurrences without replacement using equal weights, $w_i=1$. The final observed transition is protected.

**Recency-aware edge deletion** favors older transition occurrences. With $M=L-1$ original transitions and one-based position $i$, the implemented weight is $w_i=((M-i)/(M-1))^\gamma$ for $i=1,\ldots,M-1$. The final transition is excluded from deletion.

**Recency-aware substitution** replaces earlier item identities with samples from the training vocabulary, excluding the replaced item and held-out target. It protects the final click and independently rebuilds each view's nodes, aliases, and adjacency.

A deletion budget counts transition occurrences, rather than unique binary graph edges. Repeated transitions can keep an edge present after an occurrence is removed.

Nearest rounding converts the requested edit count to the nearest integer. Stochastic rounding randomly rounds up or down to match that count in expectation. Short-prefix safeguards apply in both cases.

### Which paired sessions contribute to InfoNCE

**Unmasked SSL** includes every session pair in the batch as an anchor and a negative candidate, even when its two graph inputs are identical.

For Recency deletion at 20% with nearest rounding, approximately 74.64% of Diginetica pairs were identical in the first training epoch. This percentage is specific to that setting; it is not a universal rate for every augmentation.

Identical views provide no positive-pair variation from the augmentation, while still participating in negative-pair discrimination. This does not by itself prove that their gradients are harmful or that learned representations have collapsed.

**Masked SSL** retains only pairs whose actual model inputs differ. It filters both rows and columns of the similarity matrix, removing excluded pairs as both anchors and negative candidates.

If $K$ of the original $B$ pairs remain, the loss is $(K/B)\,\mathrm{InfoNCE}_K$. For $K<2$, the contrastive loss is differentiable zero. Recommendation loss still uses all $B$ examples.

Masking therefore changes the eligible examples, the negative pool, and the effective loss scale. The observed recovery cannot be attributed to only one of those changes.

Variants without “Masked” in their names use unmasked InfoNCE. All masked variants in the tables use a 20% budget and nearest rounding.

## Single-seed Diginetica clean recommendation accuracy

Scores are multiplied by 100; higher is better. A difference between two displayed scores is measured in percentage points (pp).

| Model / augmentation | Recall@20 | MRR@20 | NDCG@20 | Seconds/epoch |
|---|---:|---:|---:|---:|
| SR-GNN | 49.506 | 17.100 | 24.249 | 44.0 |
| SASRec | 52.179 | 17.879 | 25.469 | 59.5 |
| Uniform 10%, nearest | 49.165 | 16.830 | 23.957 | 54.6 |
| Uniform 10%, stochastic | 49.173 | 16.813 | 23.947 | 55.0 |
| Uniform 20%, nearest | 49.218 | 16.791 | 23.942 | 54.6 |
| Uniform 20%, stochastic | 49.198 | 16.798 | 23.944 | 53.8 |
| Recency 10%, nearest | 49.224 | 16.835 | 23.972 | 54.9 |
| Recency 10%, stochastic | 49.158 | 16.811 | 23.943 | 55.3 |
| Recency 20%, nearest | 49.223 | 16.765 | 23.923 | 54.6 |
| Recency 20%, stochastic | 49.180 | 16.790 | 23.933 | 54.3 |
| Masked Uniform 20%, nearest | 49.481 | 17.081 | 24.229 | 59.1 |
| Masked Recency 20%, nearest | 49.357 | 17.114 | 24.227 | 58.4 |
| Masked Recency substitution 20%, nearest | 49.370 | 17.085 | 24.211 | 62.7 |

Epoch times include training and validation. They are recorded runtime averages, rather than controlled hardware benchmarks or inference-latency measurements.

## Single-seed Diginetica performance under corrupted inputs

The following scores are arithmetic means across the nine corruption conditions, weighted equally. R@20 means Recall@20, and N@20 means NDCG@20.

Relative drop is computed separately for each condition as `100 × (clean score − corrupted score) / clean score`, then averaged. Higher absolute scores and lower relative drops are better.

| Model / augmentation | Mean R@20 | Mean MRR@20 | Mean N@20 | Recall drop | MRR drop |
|---|---:|---:|---:|---:|---:|
| SR-GNN | 46.272 | 15.883 | 22.584 | 6.533% | 7.117% |
| SASRec | 49.381 | 16.924 | 24.098 | 5.363% | 5.345% |
| Uniform 10%, nearest | 45.935 | 15.580 | 22.267 | 6.570% | 7.428% |
| Uniform 10%, stochastic | 45.963 | 15.567 | 22.264 | 6.529% | 7.411% |
| Uniform 20%, nearest | 46.007 | 15.560 | 22.269 | 6.523% | 7.331% |
| Uniform 20%, stochastic | 45.999 | 15.569 | 22.275 | 6.502% | 7.320% |
| Recency 10%, nearest | 45.968 | 15.581 | 22.274 | 6.615% | 7.447% |
| Recency 10%, stochastic | 45.963 | 15.568 | 22.265 | 6.499% | 7.396% |
| Recency 20%, nearest | 45.983 | 15.546 | 22.253 | 6.582% | 7.272% |
| Recency 20%, stochastic | 45.964 | 15.557 | 22.258 | 6.538% | 7.346% |
| Masked Uniform 20%, nearest | 46.268 | 15.848 | 22.555 | 6.494% | 7.215% |
| Masked Recency 20%, nearest | 46.179 | 15.881 | 22.560 | 6.439% | 7.208% |
| Masked Recency substitution 20%, nearest | 46.146 | 15.830 | 22.517 | 6.530% | 7.345% |

## Interpreting the single-seed Diginetica results

### Unmasked graph deletion reduced accuracy

Every tested unmasked Uniform and Recency variant underperformed plain SR-GNN on clean MRR@20. Scores ranged from 16.765 to 16.835, compared with 17.100 for the baseline: a deficit of 0.266–0.336 pp.

Average corrupted MRR@20 ranged from 15.546 to 15.581, compared with 15.883 for SR-GNN. Changing the deletion budget or rounding rule did not remove this deficit.

### Short sessions often produced identical views

With nearest rounding, 81.42% of training prefixes received no deletions at a 10% budget. At a 20% budget, 58.15% still received no deletions.

Matching random draws and repeated transitions can also produce identical binary graphs after edits. For Recency deletion at 20% with nearest rounding, only 25.36% of view pairs were distinct in the first training epoch.

### Identical-pair masking recovered the MRR penalty

For Uniform deletion at 20% with nearest rounding, masking increased clean MRR@20 from 16.791 to 17.081. Average corrupted MRR@20 increased from 15.560 to 15.848.

For Recency deletion at the same budget and rounding, clean MRR@20 increased from 16.765 to 17.114. Average corrupted MRR@20 increased from 15.546 to 15.881.

These matched seed-2028 comparisons motivated the repeated evaluation above. Five-seed results confirm recovery relative to unmasked SSL, but do not isolate which part of the modified objective caused it.

### Near-baseline accuracy does not establish superiority

Masked Uniform achieved the highest clean Recall and NDCG among the SSL variants. Both remained below SR-GNN: Recall@20 was 49.481 versus 49.506, and NDCG@20 was 24.229 versus 24.249.

Masked Recency achieved the highest clean SSL MRR@20 at 17.114. Its advantage over SR-GNN was only 0.014 pp, while its Recall and NDCG were lower.

No SSL variant exceeded SR-GNN's average corrupted Recall, MRR, or NDCG. Uniform and Recency exhibited small metric-dependent trade-offs, without a consistent recency advantage.

SASRec remained the strongest reference on clean accuracy and all three average corrupted metrics. Its architecture differs from SR-GNN, so the matched SSL comparisons are the more direct tests of the contrastive extension.

### Robustness depends on the metric being measured

Masked Recency slightly reduced the mean relative Recall drop: 6.439%, compared with 6.533% for SR-GNN. Its mean relative MRR drop was worse: 7.208%, compared with 7.117%.

A lower relative drop can occur alongside lower absolute corrupted accuracy when the clean starting score is also lower. Robustness claims therefore need both absolute scores and relative degradation.

### Substitution did not resolve substitution sensitivity

Masked Recency substitution scored 17.085 on clean MRR@20 and 15.830 on average corrupted MRR@20. Both were below SR-GNN.

Its relative MRR degradation was also worse than SR-GNN at each of the three substitution test rates. Exposure to substituted items in the auxiliary objective did not produce the intended robustness benefit under the tested settings.

There was no unmasked-substitution counterpart. Its near-baseline accuracy therefore cannot be used as a third independent demonstration of the effect of masking.

### Near-baseline accuracy required additional computation

Masked edge-deletion variants averaged 58.4–59.1 seconds per epoch, compared with 44.0 seconds for SR-GNN: approximately 33–34% longer. Masked substitution averaged 62.7 seconds, approximately 42% longer.

The extra recorded training time was not accompanied by a consistent accuracy or robustness improvement. These measurements do not imply additional inference cost, because evaluation uses the ordinary SR-GNN forward pass.

## Single-seed RetailRocket results

### Evaluation context

RetailRocket was evaluated on the same 58,052 test examples across all models and conditions. The SSL variants used the same clean training data, backbone, optimizer, and training schedule as plain SR-GNN.

The deletion-augmentation baseline used perturbed training prefixes, while retaining the same validation and test sets. All reported results use seed 2028, with checkpoints selected by validation MRR@20.

All SSL variants below use a 20% budget and nearest rounding. Scores are multiplied by 100. Corrupted means and relative drops follow the same definitions as the Diginetica tables.

### Clean recommendation accuracy

| Model / augmentation | Recall@20 | MRR@20 | NDCG@20 | Seconds/epoch |
|---|---:|---:|---:|---:|
| SR-GNN | 56.231 | 32.214 | 37.687 | 61.5 |
| SASRec | 59.085 | 33.723 | 39.493 | 57.1 |
| SR-GNN + deletion augmentation | 56.269 | 32.177 | 37.665 | 55.6 |
| Unmasked Recency 20% | 56.434 | 32.251 | 37.759 | 100.5 |
| Masked Uniform 20% | 56.194 | 32.208 | 37.671 | 104.5 |
| Masked Recency 20% | 56.148 | 32.182 | 37.641 | 105.0 |
| Masked Recency substitution 20% | 56.236 | 32.182 | 37.660 | 115.1 |

### Performance averaged across the nine corrupted conditions

| Model / augmentation | Mean R@20 | Mean MRR@20 | Mean N@20 | Recall drop | MRR drop |
|---|---:|---:|---:|---:|---:|
| SR-GNN | 53.958 | 30.887 | 36.143 | 4.042% | 4.120% |
| SASRec | 57.162 | 32.549 | 38.142 | 3.255% | 3.480% |
| SR-GNN + deletion augmentation | 53.987 | 30.827 | 36.102 | 4.054% | 4.195% |
| Unmasked Recency 20% | 54.248 | 30.911 | 36.224 | 3.874% | 4.155% |
| Masked Uniform 20% | 53.955 | 30.857 | 36.115 | 3.985% | 4.194% |
| Masked Recency 20% | 53.919 | 30.841 | 36.093 | 3.970% | 4.167% |
| Masked Recency substitution 20% | 54.005 | 30.838 | 36.111 | 3.967% | 4.176% |

### Unmasked Recency showed small absolute gains

Unmasked Recency increased clean Recall@20 from 56.231 to 56.434, a gain of 0.203 pp. Clean MRR@20 increased from 32.214 to 32.251, a gain of 0.037 pp.

Average corrupted Recall@20 increased from 53.958 to 54.248, a gain of 0.290 pp. Average corrupted MRR@20 increased from 30.887 to 30.911, a gain of 0.024 pp.

These are observed gains at one seed. They suggest that unmasked SSL's effect may depend on the dataset; they do not establish a reliable improvement across repeated training.

### Absolute accuracy and relative robustness give different answers

Unmasked Recency reduced the mean relative Recall drop from 4.042% to 3.874%. Its mean relative MRR drop was slightly worse: 4.155% versus 4.120%.

This means its higher average corrupted MRR does not establish lower overall sensitivity to corruption. Its clean starting MRR was also higher.

At 20% corruption, its benefits were concentrated in mixed and substitution noise. Deletion MRR was effectively unchanged, with a tiny decrease before rounding.

| Corruption | SR-GNN Recall@20 | Unmasked Recency Recall@20 | SR-GNN MRR@20 | Unmasked Recency MRR@20 |
|---|---:|---:|---:|---:|
| Deletion 20% | 55.536 | 55.750 | 31.867 | 31.866 |
| Mixed 20% | 51.268 | 51.721 | 29.295 | 29.360 |
| Substitution 20% | 50.088 | 50.474 | 28.621 | 28.738 |

For substitution at 20%, Unmasked Recency improved Recall@20 by 0.386 pp and MRR@20 by 0.117 pp. The relative substitution drops also improved: Recall from 10.924% to 10.561%, and MRR from 11.155% to 10.893%.

These condition-specific improvements should be reported alongside the mixed outcome in average relative MRR degradation. They do not imply a uniform benefit across every condition and metric.

### Masking did not improve the RetailRocket baseline

Masked Uniform and Masked Recency remained close to plain SR-GNN, but both scored lower on clean and average corrupted Recall, MRR, and NDCG.

Masking Recency reduced clean MRR@20 from 32.251 to 32.182 and average corrupted MRR@20 from 30.911 to 30.841. This contrasts with Diginetica, where the matched masking change recovered the unmasked MRR deficit.

The contrast supports testing dataset dependence across seeds. It does not justify describing masking as a universal correction for identical views.

### Substitution did not provide a consistent advantage

Masked substitution slightly improved clean and average corrupted Recall@20 over SR-GNN: 56.236 versus 56.231, and 54.005 versus 53.958.

Its clean and average corrupted MRR and NDCG were lower. At 20% substitution noise, its MRR@20 was 28.559 versus 28.621 for plain SR-GNN.

The substitution-aware objective therefore did not consistently improve recommendation quality or robustness under these settings. A small Recall increase alone does not establish success on substitution resilience.

### Reference models and computational cost

SASRec achieved the highest clean and average corrupted Recall, MRR, and NDCG among the evaluated models. This remains a single-seed reference comparison.

Passive deletion augmentation gave a small Recall increase over SR-GNN, but lower clean and average corrupted MRR and NDCG. It did not provide a consistent improvement.

SSL averaged 100.5–115.1 seconds per epoch, compared with 61.5 seconds for plain SR-GNN: approximately 63–87% longer. Substitution was the slowest SSL variant.

These recorded times include training and validation and are not a controlled hardware benchmark. Near-baseline performance still required additional training computation, without changing the inference architecture.

## Interpreting the single-seed cross-dataset comparison

Unmasked graph-deletion SSL reduced Diginetica MRR across the tested settings. Unmasked Recency slightly improved RetailRocket's absolute scores at seed 2028.

Identical-pair masking recovered Diginetica's MRR deficit, but reduced RetailRocket's Recency scores. Its effects depend on the dataset and on the accompanying changes to contrastive examples, negatives, and loss scale.

Neither dataset establishes a consistent advantage from recency-aware substitution. The overall evidence supports small, dataset- and metric-dependent SSL effects rather than a general accuracy or robustness improvement.

These single-seed differences motivated the completed five-seed comparison above; that comparison refines which gains persist.

## Why evaluate multiple seeds?

### Distinguishing small effects from training variability

Masked Recency exceeds plain SR-GNN by only 0.014 pp clean MRR@20 on Diginetica. On RetailRocket, Unmasked Recency gains 0.203 pp Recall@20 and 0.037 pp MRR@20 at seed 2028.

Such small differences need comparison with variability from initialization, batch ordering, and stochastic augmentation. Repeated training estimates typical performance and dispersion instead of relying on one random trajectory.

The repeated evaluation tests whether masking recovery persists across training seeds. Multiple variants evaluated at one seed are matched ablations, not independent seed replications.

### Testing whether the dataset-dependent pattern persists

Unmasked Recency reduces Diginetica MRR but slightly improves RetailRocket MRR at seed 2028. Masking improves the Diginetica comparison, while masked Recency is slightly below the RetailRocket baseline.

The completed comparisons show that the Diginetica penalty and RetailRocket Recall gain persist, while RetailRocket's small MRR gain does not. Five seeds do not automatically prove significance, equivalence, or a universal dataset effect.

The midterm roadmap explicitly includes five-seed evaluation. The purpose is to quantify uncertainty around the final claims, regardless of whether the results favor SSL.

### Focusing the repeated comparisons

The repeated study includes four variants on both datasets:

| Variant | Main comparison |
|---|---|
| Plain SR-GNN | Baseline performance and variability |
| Unmasked Recency 20% | Whether the dataset-dependent SSL effect persists |
| Masked Recency 20% | Whether masking consistently changes that effect |
| Masked Uniform 20% | Whether recency weighting helps under the same masking rule |

All SSL variants here use nearest rounding. Seeds 2024–2027 provide four additional trainings per variant and dataset; existing comparable seed-2028 results supply the fifth measurement.

This is 32 additional trainings: four variants × two datasets × four new seeds. Including the eight existing seed-2028 results gives 40 model–dataset–seed measurements. The completed repeated-seed outcomes are reported in the five-seed tables above.

SASRec, deletion augmentation, unmasked Uniform, and substitution remain single-seed reference or secondary comparisons. This limits compute while prioritizing the hypotheses about the SR-GNN extension and recency weighting.

SASRec's observed Recall@20 lead is 2.67 pp on Diginetica and 2.85 pp on RetailRocket. Those larger single-seed gaps motivate lower replication priority; they do not establish that SASRec is insensitive to seed variation.

Substitution did not deliver a consistent accuracy or robustness benefit in the single-seed study. It is excluded from repeated evaluation for compute prioritization, not because a negative outcome has been proven across seeds.

### Interpreting the repeated results

For each dataset and model, report mean and sample standard deviation across seeds. Compare SSL-minus-baseline differences within matching seeds, and inspect whether their directions and magnitudes are consistent.

Compute relative corruption degradation within each seed before averaging across seeds. The nine corruption conditions characterize robustness; they are not nine independent training replications.

Keep data splits, test corruptions, model settings, and checkpoint selection fixed. This measures variability from repeated training rather than differences introduced by changing the evaluation protocol.

The final report can state: “We focus multi-seed evaluation on SR-GNN and its primary SSL variants, including matched masked Uniform and Recency comparisons. SASRec and secondary ablations remain seed-2028 reference results.”

## Limits of the interpretation

The primary comparisons cover five seeds; the broader ablation tables cover seed 2028 only. Statistical significance and practical equivalence have not been tested. Consistency over five seeds should not be presented as a universal guarantee.

Masking changes the eligible examples, the negative pool, and the effective contrastive-loss scale. Recovery cannot be attributed solely to removing identical pairs without an additional control for those effects.

Identical augmented inputs are also different from learned representation collapse. The presence of unchanged graph pairs alone does not demonstrate that the model's representations collapsed.

The evidence supports repeatable effects within the tested datasets and fixed contrastive settings. It does not establish general behavior across recommendation datasets, alternative hyperparameters, or every source of training randomness.
