# New-target equal-operand study: preregistered protocol

**Status:** Frozen before any new-target scoring  
**Frozen:** 2026-10-12 23:44:37 Asia/Kolkata (UTC+05:30)  
**Primary model:** GPT-2 Small (`gpt2-small`, loaded with TransformerLens)  
**Analysis unit:** target sum; the planned sample is 20 targets

This protocol specifies the questions, stimuli, measurements, exclusions, and decision rules for a follow-up study of the equal-operand effect. No new-target model scores were used to choose these decisions. The earlier seven single-digit targets are excluded from this follow-up.

## 1. Research questions

1. Does the equal-operand preference observed in the earlier analysis remain positive across new, even two-digit target sums?
2. Does a comparable preference appear when the operator is replaced by a comma, making the prompt a repeated-number pattern without an addition operator?
3. Across targets, is the measured equal-operand advantage associated with the model's target-token preference on fixed neutral prompts?

Operand-order symmetry is outside the scope of this study. This protocol does not test whether reversing unequal operands changes the result. Other models are also outside scope; a separate protocol will be frozen before any other-model scoring, and only if the primary GPT-2 Small criteria below are met.

## 2. Hypotheses

- **H1, primary:** For the new target set, the mean arithmetic equal-operand advantage is positive and meets every “effect stays” criterion in Section 7.
- **H2a, secondary:** The equal-operand advantage is positive in the matched comma-prompt control.
- **H2b, secondary:** The arithmetic advantage is greater than the comma-control advantage.
- **H2c, secondary:** The arithmetic advantage is associated with target-token preference measured on the neutral prompts.

The neutral-prompt measure is a contextual token-preference measure. It is not a direct measurement of GPT-2's training-corpus frequency or evidence about why a preference was learned.

## 3. Model and scoring metric

Use the GPT-2 Small checkpoint `gpt2-small` with its paired GPT-2 tokenizer, loaded through TransformerLens. Record the exact checkpoint identifier, library versions, device, and code revision in the results record. The existing project environment reports TransformerLens 3.6.0, Transformers 5.18.0, PyTorch 2.11.0, and NumPy 2.1.3; record the versions actually used for this run rather than assuming they match.

For a prompt (p) and target (T), let (L_p(x)) be the next-token logit for the single-token, leading-space representation of (x). Define:

\[
S(p,T)=L_p(T)-\frac{L_p(T-1)+L_p(T+1)}{2}.
\]

For each target and template, define (A_T) as the score for the equal-operand prompt minus the mean score across the selected unequal splits, including both operand orders. Average the three template-level contrasts to obtain one arithmetic (A_T) per target. Define the comma-control (A_T^{comma}) in the same way using comma prompts. Define (\Delta_T=A_T-A_T^{comma}).

The target is the inferential unit. Templates, operand splits, and their two orders contribute to the target-level estimate; they are not counted as independent target observations.

## 4. Tokenization eligibility and exclusions

Before scoring, audit tokenization for every candidate target and its planned prompts. A target is eligible only if:

- (T-1), (T), and (T+1) each have a single leading-space GPT-2 token for next-token scoring;
- the doubled operand (T/2) is represented by one token in each of the three exact prompt templates; and
- at least one unequal split with two distinct operands has both operands represented by one token in every exact arithmetic and comma template, in both operand orders.

Check the actual token IDs in the complete prompt contexts, not only isolated strings. Use the same tokenizer and prompt encoding for the full run, and record the tokenization audit. Ineligible targets are excluded before scoring and listed with the reason. Do not exclude or replace targets based on model scores. Multi-token answer alternatives are outside the primary analysis.

## 5. Target selection

Candidate targets are even integers from 38 through 98 inclusive. Targets 4 through 36 are excluded because they were used in the earlier analysis. After the tokenization audit, sample 20 eligible targets without looking at any model scores, using Python's `random.Random` and seed **2026101201**. Sample by these fixed strata:

| Stratum | Candidate target sums | Number sampled |
|---|---:|---:|
| Low | 38–58, even | 7 |
| Middle | 60–78, even | 6 |
| High | 80–98, even | 7 |

If a stratum has fewer eligible targets than its allocation, take all eligible targets in that stratum and fill the remaining allocation from the other strata in proportion to their remaining eligible counts, using the same seed. If fewer than 20 targets are eligible overall, use all eligible targets and report the reduced sample. Do not add targets after inspecting scores.

## 6. Prompts and controls

Use these three fixed arithmetic templates:

1. `{a} + {b} =`
2. `{a}+{b}=`
3. `Q: {a} + {b} = ?\nA:`

For each target (T), the equal-operand prompt uses (a=b=T/2). For example, target 40 uses `20 + 20 =`, `20+20=`, and `Q: 20 + 20 = ?\nA:`.

For the unequal controls, enumerate distinct, two-digit operand pairs whose sum is (T), requiring both operands to be at least 10 and each operand to pass the tokenization audit. Treat a pair and its reversal as one split. Select up to six splits uniformly without replacement using seed **2026101202 + T**; use all available splits if fewer than six qualify. Score both orders for every selected split, for example `19 + 21 =` and `21 + 19 =` for target 40. Average across the selected splits and both orders within each template before computing the target contrast.

The matched repetition control replaces the plus sign with a comma while keeping the operands, spacing pattern, templates, and scoring target the same:

1. `{a} , {b} =`
2. `{a},{b}=`
3. `Q: {a} , {b} = ?\nA:`

Thus, for target 40, compare `20 , 20 =` with the same unequal operand controls, and likewise in the other templates. This is a prompt-pattern control; it is not intended to be a natural-language arithmetic task.

For the secondary target-token preference analysis, score each target after these fixed neutral prompts:

- `The object in the box =`
- `The word on the page =`
- `Yesterday at the store =`
- `The item sequence number =`

For each target, average the four next-token logits for (T) to obtain (B_T). Report the Spearman correlation between arithmetic (A_T) and (B_T). Interpret it only as an association with target-token preference in these contexts, not as a corpus-frequency result.

## 7. Decision rule: when the effect “stays”

H1 is supported only if **all four** conditions hold:

1. The mean arithmetic (A_T) is positive and its 95% target-level bootstrap interval excludes zero.
2. The exact one-sided sign-flip test across target-level (A_T) values has (p<0.05).
3. The mean arithmetic (A_T) is at least **0.10 logit units**.
4. At least **70%** of target-level (A_T) values are positive (round the required count up to the next whole target).

Use 10,000 bootstrap resamples, resampling targets within their sampling strata and preserving the observed number of targets in each stratum. Use bootstrap seed **2026101203**. For the exact sign-flip test, enumerate all sign assignments over the target-level values and calculate the one-sided probability of a mean at least as large as observed.

If one or more criteria fail, report that the preregistered criteria were not all met. This outcome does not establish that the effect is absent. If tokenization leaves too few eligible targets for a meaningful test, report the reduced sample and the resulting uncertainty; do not silently relax a threshold.

The effect is described as showing an arithmetic-context increment over the repetition control only if H2b has a positive mean (\Delta_T), its 95% stratified target bootstrap interval excludes zero, and its Holm-adjusted one-sided test has (p<0.05). Passing this comparison does not by itself establish an addition mechanism. If the comma control also shows a positive effect, report that result directly; a nonsignificant difference does not prove that the arithmetic and comma effects are equal.

## 8. Multiple comparisons

H1 is the single primary test and is not adjusted. The secondary family consists of exactly three tests: H2a (comma-control mean (A_T^{comma}>0)), H2b (mean (\Delta_T>0)), and H2c (Spearman association between (A_T) and (B_T), two-sided). Apply Holm correction across these three secondary p-values. Report raw and adjusted p-values, effect estimates, and intervals. Label any other analyses exploratory.

## 9. Reporting and data handling

Report every selected target, its stratum, tokenization eligibility, all template-level scores, selected unequal splits and orders, target-level arithmetic and comma contrasts, neutral-prompt logits, and any exclusions. Preserve raw prompt-level outputs and the analysis table so reported values can be traced to prompts. Record the code revision and software versions. Do not overwrite earlier study data or combine this follow-up with the earlier seven-target analysis as if they were one preregistered sample.

## Analysis outcome and implementation record

The analysis below is part of the same research program and is reported in sequence with the earlier target scans. This record preserves the original plan while making the executed analysis and its limits explicit.

- **Analysis record entered 2026-10-14:** The analyzed set contains 20 even targets. Selection used seed 42 and ranges 38–56, 58–76, and 78–98, instead of seed 2026101201 and ranges 38–58, 60–78, and 80–98 in this plan. The original numeric bins contain 8/6/6 selected targets; the revised ranges label them 7/6/7. The saved data use one arithmetic template and one unequal split in both operand orders per target, rather than the planned three arithmetic templates and up to six splits. The comma and natural-language conditions also differ from the planned controls.
- **Primary arithmetic result:** Mean contrast +0.39392; exact one-sided sign-flip $p=0.013618$; stratified 95% bootstrap interval [+0.09556, +0.71692]; 8/20 positive. The mean, interval, and p-value criteria pass; the 70% positive-target criterion fails. The four criteria are therefore not all met.
- **Exploratory observation:** Six selected targets divisible by ten average +1.4313; the other 14 average −0.0507. This split was identified after inspecting the results and is not a confirmatory test.
- **Analysis artifacts:** The prompt-level logits, derived per-target table, and analysis script are listed in `data/MANIFEST.csv`.
