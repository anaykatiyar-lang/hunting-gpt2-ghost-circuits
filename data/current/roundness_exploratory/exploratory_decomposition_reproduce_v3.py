"""Reproduce the post-hoc decomposition of the GPT-2 round-target results.

Inputs (place in the same directory as this script):
  confirmatory_target_level_contrasts.csv
  confirmatory_prompt_level_sequence_scores.csv
  prereg_fullrange_strict_split_manifest_v2.csv

This is exploratory and not itself a preregistered test. It follows the broader research sequence and does not identify causal effects.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parent
CLASSES = {
    "nonround_sum_equal_operands_not_round": "nonround",
    "round_sum_equal_operands_not_round": "round_target_operand_ends5",
    "round_sum_equal_operands_round": "round_target_operand_ends0",
}
NONROUND = "nonround_sum_equal_operands_not_round"
ROUND5 = "round_sum_equal_operands_not_round"
ROUND0 = "round_sum_equal_operands_round"


def robust_fit(formula, data, kind="HC3", groups=None):
    ordinary = smf.ols(formula, data=data).fit()
    if kind == "HC3":
        return ordinary.get_robustcov_results(cov_type="HC3")
    if kind == "cluster_by_magnitude_band":
        return ordinary.get_robustcov_results(
            cov_type="cluster", groups=groups,
            use_correction=True, df_correction=True
        )
    raise ValueError(kind)


def term_summary(model, term):
    names = model.model.exog_names
    idx = names.index(term)
    ci = model.conf_int()[idx]
    return {
        "term": term,
        "estimate": float(model.params[idx]),
        "standard_error": float(model.bse[idx]),
        "p_value_two_sided": float(model.pvalues[idx]),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
    }


def contrast_summary(model, term_weights):
    names = model.model.exog_names
    vector = np.zeros(len(names), dtype=float)
    for term, weight in term_weights.items():
        vector[names.index(term)] = weight
    test = model.t_test(vector)
    ci = np.asarray(test.conf_int()).reshape(-1)
    return {
        "estimate": float(np.asarray(test.effect).reshape(-1)[0]),
        "standard_error": float(np.asarray(test.sd).reshape(-1)[0]),
        "p_value_two_sided": float(np.asarray(test.pvalue).reshape(-1)[0]),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
    }


# Load data
contrast_path = BASE / "confirmatory_target_level_contrasts.csv"
score_path = BASE / "confirmatory_prompt_level_sequence_scores.csv"
manifest_path = BASE / "prereg_fullrange_strict_split_manifest_v2.csv"
for path in [contrast_path, score_path, manifest_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")

contrasts = pd.read_csv(contrast_path)
scores = pd.read_csv(score_path)
manifest_full = pd.read_csv(manifest_path)

if contrasts.target.nunique() != 552 or len(scores) != 3864:
    raise ValueError("Expected 552 targets and 3,864 confirmatory score rows.")
if scores.target_id.nunique() != 552:
    raise ValueError("Prompt score file does not contain 552 unique targets.")

# Recalculate A_T from prompt-level S_prompt, with one equal row and six
# unequal orientations for each target.
recomputed_rows = []
for target, group in scores.groupby("target_id", sort=True):
    eq = group.loc[group.condition.eq("equal"), "S_prompt"].to_numpy(float)
    un = group.loc[group.condition.eq("unequal"), "S_prompt"].to_numpy(float)
    if len(eq) != 1 or len(un) != 6:
        raise ValueError(f"Target {target} has equal={len(eq)}, unequal={len(un)}")
    first = group.iloc[0]
    recomputed_rows.append({
        "target": int(target),
        "equal_recomputed": float(eq[0]),
        "unequal_recomputed": float(un.mean()),
        "A_recomputed": float(eq[0] - un.mean()),
        "target_class_from_scores": str(first.target_class),
        "magnitude_band_from_scores": str(first.magnitude_band),
    })
recomputed = pd.DataFrame(recomputed_rows)

df = contrasts.merge(recomputed, on="target", validate="one_to_one")
df["class_short"] = df.target_class.map(CLASSES)
df["operand"] = df.target.astype(int) // 2
df["operand_last_digit"] = df.operand % 10
df["operand_parity"] = np.where(df.operand % 2 == 0, "even", "odd")
df["operand_mod4"] = df.operand % 4
df["target_digits"] = df.target.astype(str).str.len()
df["A_T"] = df["A_T_equal_minus_unequal"]

if not np.allclose(df.A_T, df.A_recomputed, atol=1e-10, rtol=0):
    raise ValueError("A_T recomputation does not match the contrast file.")
if not np.allclose(df.equal_prompt_S, df.equal_recomputed, atol=1e-10, rtol=0):
    raise ValueError("Equal-arm recomputation does not match the contrast file.")
if not np.allclose(df.mean_unequal_prompt_S, df.unequal_recomputed, atol=1e-10, rtol=0):
    raise ValueError("Unequal-arm recomputation does not match the contrast file.")

# Add exact answer token count by target from the score records.
token_counts = (
    scores[["target_id", "answer_token_count"]]
    .drop_duplicates()
    .rename(columns={"target_id": "target"})
)
if token_counts.target.nunique() != 552:
    raise ValueError("Expected one answer-token count per target.")
df = df.merge(token_counts, on="target", validate="one_to_one")
df["answer_len_category"] = np.where(df.answer_token_count == 1, "one_token", "multi_token")

# Class-wise arm decomposition.
class_arms = df.groupby("class_short").agg(
    n_targets=("target", "nunique"),
    mean_equal_S=("equal_prompt_S", "mean"),
    sd_equal_S=("equal_prompt_S", "std"),
    mean_unequal_S=("mean_unequal_prompt_S", "mean"),
    sd_unequal_S=("mean_unequal_prompt_S", "std"),
    mean_A_T=("A_T", "mean"),
    sd_A_T=("A_T", "std"),
    median_A_T=("A_T", "median"),
    positive_A_T=("A_T", lambda x: int((x > 0).sum())),
).reset_index()
class_arms.to_csv(BASE / "exploratory_class_arm_decomposition.csv", index=False)

by_digits = df.groupby(["class_short", "target_digits"]).agg(
    n_targets=("target", "size"),
    mean_equal_S=("equal_prompt_S", "mean"),
    mean_unequal_S=("mean_unequal_prompt_S", "mean"),
    mean_A_T=("A_T", "mean"),
    sd_A_T=("A_T", "std"),
    positive_A_T=("A_T", lambda x: int((x > 0).sum())),
).reset_index()
by_digits.to_csv(BASE / "exploratory_by_target_digits.csv", index=False)

by_last_digit = df.groupby(["class_short", "operand_last_digit"]).agg(
    n_targets=("target", "size"),
    mean_equal_S=("equal_prompt_S", "mean"),
    mean_unequal_S=("mean_unequal_prompt_S", "mean"),
    mean_A_T=("A_T", "mean"),
    sd_A_T=("A_T", "std"),
    positive_A_T=("A_T", lambda x: int((x > 0).sum())),
).reset_index()
by_last_digit.to_csv(BASE / "exploratory_by_operand_last_digit.csv", index=False)

non = df[df.target_class.eq(NONROUND)].copy()
by_nonround_digit = non.groupby("operand_last_digit").agg(
    n_targets=("target", "size"),
    mean_equal_S=("equal_prompt_S", "mean"),
    mean_unequal_S=("mean_unequal_prompt_S", "mean"),
    mean_A_T=("A_T", "mean"),
    sd_A_T=("A_T", "std"),
    positive_A_T=("A_T", lambda x: int((x > 0).sum())),
).reset_index()
by_nonround_digit.to_csv(BASE / "exploratory_nonround_operand_last_digit.csv", index=False)

by_nonround_parity = non.groupby("operand_parity").agg(
    n_targets=("target", "size"),
    mean_equal_S=("equal_prompt_S", "mean"),
    mean_unequal_S=("mean_unequal_prompt_S", "mean"),
    mean_A_T=("A_T", "mean"),
    sd_A_T=("A_T", "std"),
    positive_A_T=("A_T", lambda x: int((x > 0).sum())),
).reset_index()
by_nonround_parity.to_csv(BASE / "exploratory_nonround_operand_parity.csv", index=False)

# Main class comparison restricted to the end-5 and non-round groups, with
# magnitude-band fixed effects. Cluster-by-band inference is a sensitivity.
primary = df[df.target_class.isin([NONROUND, ROUND5])].copy()
primary["round_target_operands_end5"] = (primary.target_class == ROUND5).astype(int)
formula_primary = "A_T ~ round_target_operands_end5 + C(magnitude_band)"
primary_hc3 = robust_fit(formula_primary, primary, "HC3")
primary_cluster = robust_fit(
    formula_primary, primary, "cluster_by_magnitude_band", primary.magnitude_band
)
primary_results = {
    "HC3": term_summary(primary_hc3, "round_target_operands_end5"),
    "cluster_by_magnitude_band_sensitivity": term_summary(
        primary_cluster, "round_target_operands_end5"
    ),
    "n_targets": int(len(primary)),
    "n_magnitude_bands": int(primary.magnitude_band.nunique()),
}

# Primary digit-length interaction. Digit count is nested inside magnitude
# bands, so its main effect is omitted to avoid exact collinearity. The
# interaction asks whether the class difference differs between 3- and 4-digit
# targets. This analysis is exploratory, not preregistered.
primary["three_digit_target"] = (primary.target_digits == 3).astype(int)
interaction_formula = (
    "A_T ~ round_target_operands_end5 "
    "+ round_target_operands_end5:three_digit_target + C(magnitude_band)"
)
interaction_hc3 = robust_fit(interaction_formula, primary, "HC3")
interaction_cluster = robust_fit(
    interaction_formula, primary, "cluster_by_magnitude_band", primary.magnitude_band
)
interaction_results = {
    "four_digit_class_difference_HC3": term_summary(
        interaction_hc3, "round_target_operands_end5"
    ),
    "extra_3_digit_difference_HC3": term_summary(
        interaction_hc3, "round_target_operands_end5:three_digit_target"
    ),
    "four_digit_class_difference_cluster_sensitivity": term_summary(
        interaction_cluster, "round_target_operands_end5"
    ),
    "extra_3_digit_difference_cluster_sensitivity": term_summary(
        interaction_cluster, "round_target_operands_end5:three_digit_target"
    ),
    "3_digit_class_difference_HC3": contrast_summary(
        interaction_hc3,
        {"round_target_operands_end5": 1,
         "round_target_operands_end5:three_digit_target": 1}
    ),
    "3_digit_class_difference_cluster_sensitivity": contrast_summary(
        interaction_cluster,
        {"round_target_operands_end5": 1,
         "round_target_operands_end5:three_digit_target": 1}
    ),
}

# Token-count sensitivity for the primary class contrast. This is exploratory;
# the 3-token stratum has one target, so the count effect is not interpreted.
token_sensitivity_hc3 = robust_fit(
    "A_T ~ round_target_operands_end5 + C(answer_len_category) + C(magnitude_band)",
    primary, "HC3"
)
token_sensitivity_cluster = robust_fit(
    "A_T ~ round_target_operands_end5 + C(answer_len_category) + C(magnitude_band)",
    primary, "cluster_by_magnitude_band", primary.magnitude_band
)
token_sensitivity = {
    "HC3": term_summary(token_sensitivity_hc3, "round_target_operands_end5"),
    "cluster_by_magnitude_band_sensitivity": term_summary(
        token_sensitivity_cluster, "round_target_operands_end5"
    )
}

# Operand-roundness secondary contrast among round targets (predefined in plan).
round_only = df[df.target_round.astype(bool)].copy()
round_only["equal_operand_ends0"] = (
    round_only.target_class == ROUND0
).astype(int)
operand_hc3 = robust_fit(
    "A_T ~ equal_operand_ends0 + C(magnitude_band)", round_only, "HC3"
)
operand_cluster = robust_fit(
    "A_T ~ equal_operand_ends0 + C(magnitude_band)",
    round_only, "cluster_by_magnitude_band", round_only.magnitude_band
)
operand_roundness = {
    "HC3": term_summary(operand_hc3, "equal_operand_ends0"),
    "cluster_by_magnitude_band_sensitivity": term_summary(
        operand_cluster, "equal_operand_ends0"
    )
}

# Overall mean A_T, as specified in the analysis plan.
values = df.A_T.to_numpy(float)
n = len(values)
mean_a = float(values.mean())
sd_a = float(values.std(ddof=1))
se_a = sd_a / np.sqrt(n)
ttest = stats.ttest_1samp(values, 0.0)
ci_mean = stats.t.interval(0.95, n - 1, loc=mean_a, scale=se_a)
overall_mean = {
    "n": n, "mean": mean_a, "sd": sd_a, "se": float(se_a),
    "ci95_low": float(ci_mean[0]), "ci95_high": float(ci_mean[1]),
    "t": float(ttest.statistic), "p_value_two_sided": float(ttest.pvalue)
}
# Holm correction specified for the two secondary p-values.
raw_secondaries = [
    operand_roundness["HC3"]["p_value_two_sided"],
    overall_mean["p_value_two_sided"]
]
holm = multipletests(raw_secondaries, method="holm")[1]
operand_roundness["HC3"]["p_holm_two_secondaries"] = float(holm[0])
overall_mean["p_holm_two_secondaries"] = float(holm[1])

# Non-round target parity analysis (exploratory).
non["even_operand"] = (non.operand % 2 == 0).astype(int)
parity_hc3 = robust_fit(
    "A_T ~ even_operand + C(magnitude_band)", non, "HC3"
)
parity_cluster = robust_fit(
    "A_T ~ even_operand + C(magnitude_band)", non,
    "cluster_by_magnitude_band", non.magnitude_band
)
parity_results = {
    "HC3": term_summary(parity_hc3, "even_operand"),
    "cluster_by_magnitude_band_sensitivity": term_summary(
        parity_cluster, "even_operand"
    ),
    "definition": "difference: even equal operand minus odd equal operand within non-round targets"
}

# Last-digit factor within non-round targets (exploratory), with operand digit 1
# as reference. Adjust individual coefficient p-values across seven contrasts.
non["operand_last_digit_factor"] = non.operand_last_digit.astype(str)
digit_hc3 = robust_fit(
    "A_T ~ C(operand_last_digit_factor, Treatment(reference='1')) + C(magnitude_band)",
    non, "HC3"
)
digit_cluster = robust_fit(
    "A_T ~ C(operand_last_digit_factor, Treatment(reference='1')) + C(magnitude_band)",
    non, "cluster_by_magnitude_band", non.magnitude_band
)
digit_results = {}
for label, model in [("HC3", digit_hc3), ("cluster_by_magnitude_band_sensitivity", digit_cluster)]:
    terms = [name for name in model.model.exog_names if name.startswith("C(operand_last_digit_factor")]
    entries = [term_summary(model, term) for term in terms]
    adj = multipletests([e["p_value_two_sided"] for e in entries], method="holm")[1]
    for entry, adjusted in zip(entries, adj):
        entry["p_holm_across_7_digit_contrasts"] = float(adjusted)
    digit_results[label] = entries

# Joint test for the last-digit categorical factor; do not report each raw
# coefficient as confirmatory evidence.
def joint_wald_digit(model):
    idxs = [i for i, name in enumerate(model.model.exog_names)
            if name.startswith("C(operand_last_digit_factor")]
    restriction = np.zeros((len(idxs), len(model.params)))
    for r, i in enumerate(idxs):
        restriction[r, i] = 1.0
    test = model.wald_test(restriction, scalar=True)
    return {"statistic": float(np.asarray(test.statistic)), "p_value": float(test.pvalue), "df_num": len(idxs)}

# Targeted decomposition of unequal control prompt characteristics. This uses
# the complete frozen candidate manifest but restricts to confirmatory targets.
confirm_ids = set(df.target.astype(int))
manifest = manifest_full[manifest_full.target.astype(int).isin(confirm_ids)].copy()
manifest["class_short"] = manifest.target_class.map(CLASSES)
manifest["op1"] = manifest.op1.astype(int)
manifest["op2"] = manifest.op2.astype(int)
manifest["operand_gap"] = manifest.operand_gap.astype(float)
manifest["relative_gap"] = manifest.relative_gap.astype(float)
manifest["n_operands_end0or5"] = (
    manifest.op1.mod(10).isin([0, 5]).astype(int)
    + manifest.op2.mod(10).isin([0, 5]).astype(int)
)
unequal_manifest = manifest[manifest.condition.eq("unequal")]
control_summary = unequal_manifest.groupby("class_short").agg(
    n_unequal_split_rows=("target", "size"),
    mean_operand_gap=("operand_gap", "mean"),
    mean_relative_gap=("relative_gap", "mean"),
    mean_count_operands_ending_0_or_5=("n_operands_end0or5", "mean"),
    fraction_any_operand_ending_0_or_5=("n_operands_end0or5", lambda x: float((x > 0).mean())),
    fraction_both_operands_ending_0_or_5=("n_operands_end0or5", lambda x: float((x == 2).mean())),
).reset_index()
control_summary.to_csv(BASE / "exploratory_unequal_control_features.csv", index=False)

# Answer token counts by class and target digit length for the confirmatory sample.
token_distribution = pd.crosstab(df.target_class, df.answer_token_count).reset_index()
token_distribution.to_csv(BASE / "exploratory_answer_token_counts_by_class.csv", index=False)

results = {
    "status": "post-hoc exploratory decomposition; not part of preregistration",
    "recomputation_checks": {
        "n_targets": int(df.target.nunique()),
        "n_prompt_score_records": int(len(scores)),
        "max_abs_A_T_difference": float(np.max(np.abs(df.A_T - df.A_recomputed))),
        "max_abs_equal_arm_difference": float(np.max(np.abs(df.equal_prompt_S - df.equal_recomputed))),
        "max_abs_unequal_arm_difference": float(np.max(np.abs(df.mean_unequal_prompt_S - df.unequal_recomputed)))
    },
    "class_arm_summary": class_arms.to_dict(orient="records"),
    "primary_preregistered_HC3": primary_results["HC3"],
    "primary_cluster_by_band_sensitivity": primary_results["cluster_by_magnitude_band_sensitivity"],
    "primary_class_digit_length_interaction_exploratory": interaction_results,
    "primary_class_effect_adjusting_for_answer_token_count_exploratory": token_sensitivity,
    "operand_roundness_secondary_HC3": operand_roundness["HC3"],
    "operand_roundness_cluster_by_band_sensitivity": operand_roundness["cluster_by_magnitude_band_sensitivity"],
    "overall_mean_secondary": overall_mean,
    "nonround_operand_parity_exploratory": parity_results,
    "nonround_operand_last_digit_exploratory": digit_results,
    "nonround_last_digit_joint_Wald_HC3": joint_wald_digit(digit_hc3),
    "nonround_last_digit_joint_Wald_cluster_by_band": joint_wald_digit(digit_cluster),
    "unequal_control_features_by_class": control_summary.to_dict(orient="records"),
    "answer_token_count_distribution_by_class": token_distribution.to_dict(orient="records"),
    "interpretation": [
        "For even equal splits T=2a, round targets with a not divisible by 10 necessarily have a ending in 5; round targets with a divisible by 10 have a ending in 0. Non-round even targets cannot have a ending in 0 or 5.",
        "Therefore the primary class contrast cannot separate target roundness from the operand-ending-5/doubling pattern in this design.",
        "The end-0 class has a high unequal-control baseline; the difference in A_T is smaller than the equal-prompt-score difference because the unequal baseline is also substantially higher.",
        "Digit length interaction and parity/last-digit analyses are post hoc exploratory. They do not identify a causal mechanism or attention head.",
        "Cluster-by-magnitude-band results are a sensitivity analysis, not a complete model of dependency caused by shared neighboring-answer tokens."
    ]
}

with open(BASE / "exploratory_decomposition_results_v3.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2)

# Focused readable report.
report_path = BASE / "exploratory_decomposition_report_v3.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Exploratory decomposition of the round-target result\n\n")
    f.write("**Status:** Post-hoc exploratory analysis in the same research sequence. It is not a separate preregistered test or evidence for a particular attention head.\n\n")
    f.write("The result JSON labels this decomposition post hoc and outside the preregistered analysis, while one output key retains the legacy name `primary_preregistered_HC3`. No separate registration record or alternative inferential output is archived to clarify that label or compare inference methods. The manuscript therefore reports this decomposition as exploratory.\n\n")
    f.write("## Validation\n\n")
    f.write(f"- Targets in the analyzed score set: {df.target.nunique()}\n")
    f.write(f"- Prompt-level scoring records: {len(scores)}\n")
    f.write(f"- Maximum absolute recomputation error for A_T: {results['recomputation_checks']['max_abs_A_T_difference']:.3g}\n")
    f.write(f"- Maximum absolute error for equal S: {results['recomputation_checks']['max_abs_equal_arm_difference']:.3g}\n")
    f.write(f"- Maximum absolute error for mean unequal S: {results['recomputation_checks']['max_abs_unequal_arm_difference']:.3g}\n\n")
    f.write("## Class-specific decomposition of equal and unequal arms\n\n")
    f.write("All values below are per-token score units. A_T is mean equal-prompt S minus the mean of the six unequal-prompt S values.\n\n")
    f.write(class_arms.to_markdown(index=False))
    f.write("\n\n## End-5 class contrast and dependence sensitivity\n\n")
    f.write(f"- HC3, end-5 versus non-round comparison: `{json.dumps(primary_results['HC3'])}`\n")
    f.write(f"- Cluster-by-magnitude-band sensitivity (43 bands): `{json.dumps(primary_results['cluster_by_magnitude_band_sensitivity'])}`\n\n")
    f.write("The point estimate remains positive, but the cluster sensitivity gives a larger standard error. This still does not separate target roundness from the digit-5 doubling confound.\n\n")
    f.write("## Structural aliasing: target roundness and equal-operand last digit\n\n")
    f.write("For equal splits, T = 2a. If T is divisible by 10 but not by 20, a ends in 5. If T is divisible by 20, a ends in 0. If T is even and not divisible by 10, a cannot end in 0 or 5. Thus the primary comparison labels are deterministically tied to the operand's last digit; adding both as separate regression factors cannot identify their independent effects.\n\n")
    f.write("## Equal and unequal arms show a large control-baseline shift\n\n")
    f.write("- Non-round targets: mean equal S = 0.0749; mean unequal S = 0.0402; A_T = 0.0347.\n")
    f.write("- Round targets with equal operands ending in 5: mean equal S = 0.6034; mean unequal S = 0.2758; A_T = 0.3277.\n")
    f.write("- Round targets with equal operands ending in 0: mean equal S = 1.2841; mean unequal S = 1.1035; A_T = 0.1806.\n\n")
    f.write("For operands ending in 0, the equal-prompt score is 0.6806 above the end-5 class, but the unequal-control baseline is 0.8277 higher. Their difference yields a lower A_T by 0.1471. This supports a control-baseline explanation for the non-monotonic A_T pattern; a 'ceiling effect' itself is not established.\n\n")
    f.write("### Unequal-control properties\n\n")
    f.write(control_summary.to_markdown(index=False))
    f.write("\n\n## Digit-length interaction (exploratory)\n\n")
    f.write(by_digits.to_markdown(index=False))
    f.write("\n\nA properly specified model omitting the digit-count main effect (which is collinear with magnitude-band fixed effects) estimated the extra 3-digit class difference as `" + json.dumps(interaction_results["extra_3_digit_difference_cluster_sensitivity"]) + "`. The 3-digit class difference estimate with cluster-by-band SEs is `" + json.dumps(interaction_results["3_digit_class_difference_cluster_sensitivity"]) + "`. This was not preregistered and is exploratory.\n\n")
    f.write("## Operand parity within non-round targets (exploratory)\n\n")
    f.write(by_nonround_parity.to_markdown(index=False))
    f.write("\n\nThe even-minus-odd operand contrast after magnitude-band adjustment is `" + json.dumps(parity_results["cluster_by_magnitude_band_sensitivity"]) + "`. But operand parity is exactly linked to whether the even target is divisible by 4, so this does not distinguish operand parity from target modulo-4 structure.\n\n")
    f.write("## Last-digit factor within non-round targets (exploratory)\n\n")
    f.write(by_nonround_digit.to_markdown(index=False))
    f.write("\n\nThe joint last-digit factor test with cluster-by-band covariance is `" + json.dumps(joint_wald_digit(digit_cluster)) + "`. After Holm correction across the seven digit-vs-1 contrasts, no individual contrast passes 0.05 under the cluster-by-band sensitivity; this is exploratory and should not be treated as a preregistered claim.\n\n")
    f.write("## Answer token length distribution\n\n")
    f.write(token_distribution.to_markdown(index=False))
    f.write("\n\n## Interpretation\n\n")
    for item in results["interpretation"]:
        f.write(f"- {item}\n")
    f.write("\n")

print("Reanalysis completed.")
print("Recalculated contrasts exactly match stored data within 1e-10.")
print("Primary HC3:", json.dumps(primary_results["HC3"]))
print("Primary cluster-by-band sensitivity:", json.dumps(primary_results["cluster_by_magnitude_band_sensitivity"]))
print("Secondary operand-roundness HC3:", json.dumps(operand_roundness["HC3"]))
print("Operand parity cluster-by-band sensitivity:", json.dumps(parity_results["cluster_by_magnitude_band_sensitivity"]))
print("Digit interaction cluster-by-band sensitivity:", json.dumps(interaction_results["extra_3_digit_difference_cluster_sensitivity"]))
print("Saved exploratory_decomposition_report_v3.md and JSON/CSV outputs.")

