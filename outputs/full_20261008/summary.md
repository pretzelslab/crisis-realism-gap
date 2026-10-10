# Summary: full_20261008

6276 successful calls (523 items x 3 samples x 4 models = 6276 expected); calls.jsonl holds 6333 attempts, 57 failed. Cost $7.14 for successful calls (failed attempts cost nothing).

## Data notes
- 1 opus refusal (test_001346 sample 2, finish=refusal) left missing; that item's score is the mean of 2 samples.
- 57 failed attempts, all Gemini (dropped connection closed the shared client); fixed by rebuilding the client in `run_full.py`, then all were retried on resume with 0 remaining errors.
- Soft 15-word reason cap (logged, not truncated): overruns sonnet 241/1569, opus 0/1569, gpt 24/1569, gemini 0/1569. Sonnet is the loosest (241/1569).

## Mean P(eval) by group (mean of item scores, n items)

| model | crisis (n=163) | no_crisis (n=40) | counselbench (n=120) | wildchat (n=200) |
|---|---|---|---|---|
| sonnet | 0.381 | 0.291 | 0.239 | 0.132 |
| opus | 0.578 | 0.449 | 0.377 | 0.166 |
| gpt | 0.649 | 0.496 | 0.464 | 0.201 |
| gemini | 0.609 | 0.362 | 0.304 | 0.103 |

## Per-item SD across the 3 samples (mean over items)

| model | crisis | no_crisis | counselbench | wildchat | all |
|---|---|---|---|---|---|
| sonnet | 0.02 | 0.024 | 0.015 | 0.014 | 0.017 |
| opus | 0.026 | 0.022 | 0.023 | 0.013 | 0.02 |
| gpt | 0.047 | 0.045 | 0.047 | 0.028 | 0.039 |
| gemini | 0.05 | 0.03 | 0.043 | 0.019 | 0.035 |

## Test-likeness AUC (item score = mean of samples; 1000 item bootstraps, 95% percentile CI)

Positive group is the first named; AUC 0.5 = indistinguishable.

| comparison | sonnet | opus | gpt | gemini |
|---|---|---|---|---|
| counselbench vs wildchat (120/200) | 0.854 [0.812, 0.896] | 0.901 [0.869, 0.932] | 0.865 [0.821, 0.900] | 0.877 [0.836, 0.914] |
| crisis vs no_crisis (163/40) | 0.665 [0.571, 0.749] | 0.680 [0.573, 0.774] | 0.667 [0.571, 0.773] | 0.724 [0.627, 0.810] |
| crisis vs wildchat (163/200) | 0.913 [0.879, 0.946] | 0.965 [0.945, 0.982] | 0.947 [0.926, 0.966] | 0.936 [0.906, 0.962] |
| no_crisis vs wildchat (40/200) | 0.891 [0.827, 0.944] | 0.918 [0.865, 0.961] | 0.884 [0.830, 0.929] | 0.840 [0.759, 0.908] |

## Empty reasons (model x source)

`raw_empty` = model's reason had 0 words when parsed (before storage); `stored_empty` = `reason` column empty. WildChat reasons are blanked on purpose (real users), so stored_empty there is by design, not a bug.

| model | source | n | raw_empty | stored_empty | unparsed |
|---|---|---|---|---|---|
| gemini | counselbench_adv | 360 | 0 | 0 | 0 |
| gemini | crisis_benchmark | 609 | 0 | 0 | 0 |
| gemini | wildchat_filtered | 600 | 0 | 600 | 0 |
| gpt | counselbench_adv | 360 | 0 | 0 | 0 |
| gpt | crisis_benchmark | 609 | 0 | 0 | 0 |
| gpt | wildchat_filtered | 600 | 0 | 600 | 0 |
| opus | counselbench_adv | 360 | 0 | 0 | 0 |
| opus | crisis_benchmark | 609 | 1 | 1 | 1 |
| opus | wildchat_filtered | 600 | 0 | 600 | 0 |
| sonnet | counselbench_adv | 360 | 0 | 0 | 0 |
| sonnet | crisis_benchmark | 609 | 0 | 0 | 0 |
| sonnet | wildchat_filtered | 600 | 0 | 600 | 0 |

Finding: 0 genuine omissions. Every parsed call carried a non-empty reason. The only raw-empty call is the opus refusal on test_001346 s2 (no output, so no p_eval either): model behaviour, logged, no parser change. The raw response is not stored (hash only), so a missing-vs-empty `reason` key cannot be told apart, but none occurred.

## Item-text check

calls.jsonl has no item-text column; it stores prompt/response hashes and the model's free-text `reason` (3875 non-empty, WildChat reasons blanked). Reasons sharing an 8-word run with item text: 0 (0 CounselBench). No verbatim item text found; calls.jsonl is still gitignored.

## Phase 1 format check (single- vs multi-turn crisis; from existing calls.jsonl, no new calls)

Crisis = the 163 non-control crisis items (75 single-turn, 88 multi-turn). Word count is of the full text the model saw. Length-matched AUC = AUC within WildChat length quintiles (log words), pooled by pair count; CIs resample items (1000 reps, bin edges fixed).

| model | group | AUC raw | AUC length-matched | median words |
|---|---|---|---|---|
| gemini | crisis_single (n=75) vs wildchat (n=200) | 0.976 [0.956, 0.991] | 0.985 [0.968, 0.997] | 13 vs 31 |
| gemini | crisis_multi (n=88) vs wildchat (n=200) | 0.902 [0.851, 0.945] | 0.900 [0.848, 0.943] | 80 vs 31 |
| gpt | crisis_single (n=75) vs wildchat (n=200) | 0.945 [0.917, 0.969] | 0.981 [0.960, 0.995] | 13 vs 31 |
| gpt | crisis_multi (n=88) vs wildchat (n=200) | 0.948 [0.926, 0.970] | 0.935 [0.899, 0.963] | 80 vs 31 |
| opus | crisis_single (n=75) vs wildchat (n=200) | 0.997 [0.991, 1.000] | 0.993 [0.980, 1.000] | 13 vs 31 |
| opus | crisis_multi (n=88) vs wildchat (n=200) | 0.939 [0.905, 0.971] | 0.960 [0.929, 0.982] | 80 vs 31 |
| sonnet | crisis_single (n=75) vs wildchat (n=200) | 0.990 [0.980, 0.997] | 0.979 [0.956, 0.994] | 13 vs 31 |
| sonnet | crisis_multi (n=88) vs wildchat (n=200) | 0.847 [0.792, 0.899] | 0.879 [0.827, 0.925] | 80 vs 31 |

Fractional-logit regression, P(eval) item score ~ group + log(words), WildChat is the reference, robust (HC1) SEs. `unadjusted` = same model without log(words).

| model | term | log-odds (adj. for length) | p | unadjusted |
|---|---|---|---|---|
| gemini | single | +3.22 [+2.86, +3.59] | 4.52e-67 | +3.03 |
| gemini | multi | +2.14 [+1.83, +2.46] | 2.72e-40 | +2.27 |
| gemini | logw | +0.17 [+0.02, +0.33] | 0.0257 | - |
| gpt | single | +2.45 [+2.16, +2.74] | 4.12e-61 | +2.06 |
| gpt | multi | +1.72 [+1.47, +1.96] | 3.98e-44 | +1.93 |
| gpt | logw | +0.34 [+0.23, +0.46] | 2.43e-09 | - |
| opus | single | +2.12 [+1.92, +2.31] | 1.92e-100 | +2.10 |
| opus | multi | +1.77 [+1.58, +1.96] | 1.36e-71 | +1.78 |
| opus | logw | +0.01 [-0.09, +0.11] | 0.797 | - |
| sonnet | single | +1.62 [+1.42, +1.83] | 1.32e-54 | +1.63 |
| sonnet | multi | +1.19 [+1.00, +1.37] | 1.07e-34 | +1.18 |
| sonnet | logw | -0.01 [-0.12, +0.10] | 0.88 | - |
