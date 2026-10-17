<p align="center">
  <a href="https://colab.research.google.com/github/anaykatiyar-lang/hunting-gpt2-ghost-circuits/blob/main/notebooks/00_full_record.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Google Colab">
  </a>
</p>
<h1 align="center">Arithmetic Probes in GPT-2 Small</h1>

<p align="center">
  <em>When GPT-2 sees <code>3 + 3 =</code>, is it adding—or responding to a familiar pattern?</em>
</p>

<p align="center">
  A mechanistic interpretability investigation of arithmetic-like behavior, failed circuit hypotheses, and the equal-operand effect.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Research%20Draft-blue" alt="Research draft">
  <img src="https://img.shields.io/badge/Model-GPT--2%20Small-purple" alt="GPT-2 Small">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>

---

## The question
What would it take to show that GPT-2 is adding—not just matching patterns?

This project began by asking whether GPT-2 Small uses an addition-specific circuit for short sums. Behavioral baselines, attribution corrections, and intervention controls narrowed that interpretation. The seven-target scan showed a target-dependent contrast for equal-operand prompts. A 20-target analysis extended the question to new sums, but only 8/20 target contrasts were positive. A later exploratory decomposition of 552 targets found a larger equal-minus-unequal score for targets whose equal operands end in 5 than for non-round targets. Because target class, operand ending, and unequal-control composition are linked in this design, the result is a class-dependent pattern—not an isolated roundness effect or identified mechanism.

## What the results show

- **Weak zero-shot addition performance:** the recorded all-pairs evaluation achieved top-1 accuracy of 2/36. A separate held-out evaluation achieved 2/32, compared with 8/32 for its best constant-guess baseline.
- **An equal-operand pattern:** the saved digit-plus-digit scores have a mean target-level advantage of +0.247 across seven sums (exact sign-flip p = 0.03125; 95% target-bootstrap interval [+0.0738, +0.4374]). The estimate is sensitive: a sign test gives p = 0.125, and excluding controls whose operands equal the target's neighboring values gives p = 0.125 across six eligible targets. Treat this as an exploratory pattern, not a robust or mechanistic finding. The digit+digit versus digit+word contrast has Holm p = 0.046875 within its three-comparison family and was not adjusted across all project analyses.
- **A 20-target arithmetic analysis:** the mean contrast is +0.39392 (exact one-sided sign-flip p = 0.013618; stratified 95% interval [+0.09556, +0.71692]), but only 8/20 targets are positive. Three numerical thresholds pass; the 70% positive-target criterion fails. Selection used different seed and bins from the written plan; the target counts under its original bins are 8/6/6, and the saved prompt/control set includes fewer templates and splits. The post hoc round-number split is exploratory: six multiples of ten average +1.4313, while the other 14 targets average -0.0507.
- **A separate 11-target score set:** recomputation from saved logits gives a six-format mean equal-minus-unequal contrast of +0.28234 (exact one-sided sign-flip p = 0.175293; 7/11 positive). The arithmetic-only result is +0.26796 (p = 0.18652; 6/11 positive). Neither test is significant at 0.05. Raw logits, per-target contrasts, and the analysis script are archived under `data/current/` and `src/`.
- **An exploratory 552-target decomposition:** mean $A_T$ is +0.0347 nats per answer token for non-round targets, +0.3277 for targets whose equal operand ends in 5, and +0.1806 when it ends in 0. The end-5 versus non-round contrast is +0.2930 nats per answer token; clustering by magnitude band gives a 95% interval [+0.1724, +0.4137] (two-sided p = 1.47×10⁻⁵). These are differences in mean conditional log-probability per answer token, not raw logits. Target class is structurally tied to operand ending, and the unequal controls differ across classes, so this does not isolate a roundness mechanism. The target contrasts, prompt-level scores, split manifest, script, and derived outputs are archived together.
- **Operator comparison:** at target 8, the saved `and` (+0.321) and `then` (+0.344) advantages are at least as large as `plus` (+0.291). This is descriptive: the same sum token is scored for every string, and the paired tests do not detect a difference.
- **No confirmed circuit:** candidate-component screens remain exploratory (two-sided FDR q = 0.150 across 12 nominated components); transfer effects are mixed, and the five-target discovery screen has limited p-value resolution.
- **Corpus audit:** the seven-target joint log ratios are positive and conditional ratios are mixed. A post hoc Spearman analysis finds no detectable association with the equal-operand advantage (joint $\rho=+0.464$, $p=0.294$; conditional $\rho=-0.143$, $p=0.760$; $n=7$). This small, unprespecified analysis does not rule out a frequency contribution.

These conclusions apply to the tested model, prompts, targets, and controls. They do not imply that GPT-2 Small lacks every arithmetic-related representation or computation.

## Follow the investigation

**Circuit hypothesis → behavioral and token baselines → attribution and intervention audits → seven-target equal-operand scan → format/operator and corpus checks → 20-target extension → 552-target class decomposition → bounded conclusion and next test**

Explore the evidence and methods:

- [Claims and findings](docs/FINDINGS.md)
- [Methodology](docs/METHODOLOGY.md)
- [Limitations](docs/LIMITATIONS.md)
- [Research flow](docs/RESEARCH_FLOW.md)
- [Pre-analysis plan and recorded outcome](docs/PREREGISTRATION.md)
- [Audit trail](docs/Audit_Trail.md)
- [Interactive research explorer](index.html)

## Visual evidence

![Addition-grid symmetric logit-difference heatmap](figures/static/addition_grid_heatmap.svg)

![Target-level equal-operand advantages by operator string](figures/static/operator_swap_heatmap.svg)

Explore the [interactive research explorer](index.html) for additional figures and comparisons.

Attribution and attention visualizations are not, by themselves, proof of a causal circuit. Interpret intervention results alongside their clean and corrupt baselines.

## Reproduce

**Run the notebook:** click the Colab badge at the top of this page.

The notebook is a full research record, not a demonstrated clean, end-to-end rerun of every artifact. The package versions in `requirements.txt` record the saved experiment environment; they are not a verified lockfile. The analysis reproduction dependencies are listed separately.

For the recorded CPU setup, install PyTorch from its CPU wheel index, then install the repository dependencies and analysis dependencies:

```bash
git clone https://github.com/anaykatiyar-lang/hunting-gpt2-ghost-circuits.git
cd hunting-gpt2-ghost-circuits
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -r requirements-analysis.txt
```

The PyTorch command follows the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/). This installation path has not been validated in a clean environment and does not establish end-to-end reproduction of the notebook.

Regenerate the committed static figures:

```bash
python src/plot_heatmaps.py
```

See the notebook, [current data exports](data/current/), [data manifest](data/MANIFEST.csv), and methodology for experimental details. Recompute the 20-target arithmetic summary from its stored prompt-level logits with `python src/analyze_preregistered_followup.py`. The corpus scripts require network access to Infini-gram; archived raw responses permit inspection of the query run.

The exploratory class decomposition is archived under [`data/current/roundness_exploratory/`](data/current/roundness_exploratory/), including the target contrasts, prompt-level scores, split manifest, script, and derived outputs. The original v3 JSON preserves the legacy key `primary_preregistered_HC3`; the versioned v4 export renames it `exploratory_HC3` without recalculating any statistic. The v4 file records the schema-only migration explicitly. See the [exploratory report](data/current/roundness_exploratory/exploratory_decomposition_report_v3.md) and [limitations](docs/LIMITATIONS.md).

## Project structure

- `notebooks/` — full research record and Markdown export.
- `src/` — metrics, visualizations, operator comparisons, and corpus scripts.
- `data/` — committed results and artifact manifest. The separate 11-target scores and computed summaries live in `data/current/`; the 552-target class decomposition lives in `data/current/roundness_exploratory/`.
- `figures/` — static and interactive figures.
- `docs/` — methods, findings, limitations, research flow, and audit trail.
- `paper/` — manuscript draft.

## AI assistance

AI tools assisted with code review, debugging, methodological critique, and exploration of possible explanations. Their suggestions were treated as prompts for verification, not as evidence. Responsibility for the experiments, interpretation, and final claims remains with the author.
