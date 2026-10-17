# Exploratory decomposition of the round-target result

**Status:** Post-hoc exploratory analysis in the same research sequence. It is not a separate preregistered test or evidence for a particular attention head.

## Validation

- Targets in the analyzed score set: 552
- Prompt-level scoring records: 3864
- Maximum absolute recomputation error for A_T: 2.22e-16
- Maximum absolute error for equal S: 4.44e-16
- Maximum absolute error for mean unequal S: 4.44e-16

## Class-specific decomposition of equal and unequal arms

All values below are nats per answer token: mean conditional log-probabilities averaged across answer tokens. A_T is mean equal-prompt S minus the mean of the six unequal-prompt S values.

| class_short                |   n_targets |   mean_equal_S |   sd_equal_S |   mean_unequal_S |   sd_unequal_S |   mean_A_T |   sd_A_T |   median_A_T |   positive_A_T |
|:---------------------------|------------:|---------------:|-------------:|-----------------:|---------------:|-----------:|---------:|-------------:|---------------:|
| nonround                   |         276 |      0.0748507 |     0.331379 |        0.0401817 |       0.195944 |   0.034669 | 0.215737 |   0.00812801 |            145 |
| round_target_operand_ends0 |         138 |      1.28409   |     0.837529 |        1.10347   |       0.620317 |   0.180622 | 0.346672 |   0.155798   |            100 |
| round_target_operand_ends5 |         138 |      0.603449  |     0.588367 |        0.275752  |       0.31093  |   0.327697 | 0.36626  |   0.240021   |            122 |

## End-5 class contrast and dependence sensitivity

The current reproduction script explicitly uses a standard-normal reference (`use_t=False`) for HC3 and clustered covariance results. The original statsmodels version and inference setting for the saved p-values were not recorded, so their exact software-level provenance remains unresolved.

- HC3, end-5 versus non-round comparison: `{"term": "round_target_operands_end5", "estimate": 0.29302824791310866, "standard_error": 0.03262265587604113, "p_value_two_sided": 1.3711954713679509e-17, "ci95_low": 0.2288791814892569, "ci95_high": 0.3571773143369604}`
- Cluster-by-magnitude-band sensitivity (43 bands): `{"term": "round_target_operands_end5", "estimate": 0.29302824791310866, "standard_error": 0.05978905488936775, "p_value_two_sided": 1.4670434859686624e-05, "ci95_low": 0.17236905021206797, "ci95_high": 0.4136874456141494}`

The point estimate remains positive, but the cluster sensitivity gives a larger standard error. This still does not separate target roundness from the digit-5 doubling confound.

The JSON status describes this decomposition as post hoc and outside the preregistered analysis, while one output key uses a `primary_preregistered` label. No separate registration record is archived to resolve that label. Accordingly, the manuscript reports the decomposition as exploratory.

## Structural aliasing: target roundness and equal-operand last digit

For equal splits, T = 2a. If T is divisible by 10 but not by 20, a ends in 5. If T is divisible by 20, a ends in 0. If T is even and not divisible by 10, a cannot end in 0 or 5. Thus the primary comparison labels are deterministically tied to the operand's last digit; adding both as separate regression factors cannot identify their independent effects.

## Equal and unequal arms show a large control-baseline shift

- Non-round targets: mean equal S = 0.0749; mean unequal S = 0.0402; A_T = 0.0347.
- Round targets with equal operands ending in 5: mean equal S = 0.6034; mean unequal S = 0.2758; A_T = 0.3277.
- Round targets with equal operands ending in 0: mean equal S = 1.2841; mean unequal S = 1.1035; A_T = 0.1806.

For operands ending in 0, the equal-prompt score is 0.6806 above the end-5 class, but the unequal-control baseline is 0.8277 higher. Their difference yields a lower A_T by 0.1471. This supports a control-baseline explanation for the non-monotonic A_T pattern; a 'ceiling effect' itself is not established.

### Unequal-control properties

| class_short                |   n_unequal_split_rows |   mean_operand_gap |   mean_relative_gap |   mean_count_operands_ending_0_or_5 |   fraction_any_operand_ending_0_or_5 |   fraction_both_operands_ending_0_or_5 |
|:---------------------------|-----------------------:|-------------------:|--------------------:|------------------------------------:|-------------------------------------:|---------------------------------------:|
| nonround                   |                    828 |            770.604 |            0.269371 |                            0.259662 |                             0.259662 |                               0        |
| round_target_operand_ends0 |                    414 |            540.821 |            0.194602 |                            2        |                             1        |                               1        |
| round_target_operand_ends5 |                    414 |            765.86  |            0.26844  |                            0.202899 |                             0.101449 |                               0.101449 |

## Digit-length interaction (exploratory)

| class_short                |   target_digits |   n_targets |   mean_equal_S |   mean_unequal_S |   mean_A_T |   sd_A_T |   positive_A_T |
|:---------------------------|----------------:|------------:|---------------:|-----------------:|-----------:|---------:|---------------:|
| nonround                   |               3 |          32 |     -0.0630372 |       -0.0402695 | -0.0227676 | 0.189308 |             17 |
| nonround                   |               4 |         244 |      0.0929343 |        0.0507327 |  0.0422017 | 0.218193 |            128 |
| round_target_operand_ends0 |               3 |          16 |      2.4817    |        2.00001   |  0.481688  | 0.375166 |             15 |
| round_target_operand_ends0 |               4 |         122 |      1.12703   |        0.985891  |  0.141138  | 0.324142 |             85 |
| round_target_operand_ends5 |               3 |          16 |      1.30316   |        0.393277  |  0.909886  | 0.5773   |             16 |
| round_target_operand_ends5 |               4 |         122 |      0.511683  |        0.260338  |  0.251345  | 0.244688 |            106 |

A properly specified model omitting the digit-count main effect (which is collinear with magnitude-band fixed effects) estimated the extra 3-digit class difference as `{"term": "round_target_operands_end5:three_digit_target", "estimate": 0.7235105497480935, "standard_error": 0.19311748699413722, "p_value_two_sided": 0.0005406744306060998, "ci95_low": 0.3337836827509463, "ci95_high": 1.1132374167452408}`. The 3-digit class difference estimate with cluster-by-band SEs is `{"estimate": 0.9326535165309924, "standard_error": 0.18993479965758397, "p_value_two_sided": 1.4233270095512895e-05, "ci95_low": 0.5493495726135352, "ci95_high": 1.3159574604484496}`. This was not preregistered and is exploratory.

## Operand parity within non-round targets (exploratory)

| operand_parity   |   n_targets |   mean_equal_S |   mean_unequal_S |   mean_A_T |   sd_A_T |   positive_A_T |
|:-----------------|------------:|---------------:|-----------------:|-----------:|---------:|---------------:|
| even             |         144 |      0.189362  |       0.0829596  |  0.106402  | 0.252236 |            104 |
| odd              |         132 |     -0.0500706 |      -0.00648515 | -0.0435854 | 0.128162 |             41 |

The even-minus-odd operand contrast after magnitude-band adjustment is `{"term": "even_operand", "estimate": 0.14732290109340818, "standard_error": 0.03363209592625725, "p_value_two_sided": 7.731707993023949e-05, "ci95_low": 0.07945058367719368, "ci95_high": 0.21519521850962267}`. But operand parity is exactly linked to whether the even target is divisible by 4, so this does not distinguish operand parity from target modulo-4 structure.

## Last-digit factor within non-round targets (exploratory)

|   operand_last_digit |   n_targets |   mean_equal_S |   mean_unequal_S |   mean_A_T |   sd_A_T |   positive_A_T |
|---------------------:|------------:|---------------:|-----------------:|-----------:|---------:|---------------:|
|                    1 |          35 |     -0.0303993 |      0.00192638  | -0.0323257 | 0.108458 |             12 |
|                    2 |          36 |      0.296766  |      0.14365     |  0.153116  | 0.387275 |             24 |
|                    3 |          35 |     -0.0807034 |     -0.0228092   | -0.0578942 | 0.125463 |             10 |
|                    4 |          37 |      0.201556  |      0.0952107   |  0.106345  | 0.204275 |             27 |
|                    6 |          35 |      0.192147  |      0.0926307   |  0.0995166 | 0.218495 |             27 |
|                    7 |          31 |     -0.0308576 |     -0.00934425  | -0.0215133 | 0.121673 |             10 |
|                    8 |          36 |      0.0667175 |      0.000275325 |  0.0664422 | 0.129354 |             26 |
|                    9 |          31 |     -0.0569077 |      0.00530743  | -0.0622151 | 0.156526 |              9 |

The joint last-digit factor test with cluster-by-band covariance is `{"statistic": 4.541544862190421, "p_value": 0.0007622311601410081, "df_num": 7}`. After Holm correction across the seven digit-vs-1 contrasts, no individual contrast passes 0.05 under the cluster-by-band sensitivity; this is exploratory and should not be treated as a preregistered claim.

## Answer token length distribution

| target_class                          |   1 |   2 |   3 |
|:--------------------------------------|----:|----:|----:|
| nonround_sum_equal_operands_not_round |  33 | 242 |   1 |
| round_sum_equal_operands_not_round    |  17 | 121 |   0 |
| round_sum_equal_operands_round        |  17 | 121 |   0 |

## Interpretation

- For even equal splits T=2a, round targets with a not divisible by 10 necessarily have a ending in 5; round targets with a divisible by 10 have a ending in 0. Non-round even targets cannot have a ending in 0 or 5.
- Therefore the primary class contrast cannot separate target roundness from the operand-ending-5/doubling pattern in this design.
- The end-0 class has a high unequal-control baseline; the difference in A_T is smaller than the equal-prompt-score difference because the unequal baseline is also substantially higher.
- Digit length interaction and parity/last-digit analyses are post hoc exploratory. They do not identify a causal mechanism or attention head.
- Cluster-by-magnitude-band results are a sensitivity analysis, not a complete model of dependency caused by shared neighboring-answer tokens.


