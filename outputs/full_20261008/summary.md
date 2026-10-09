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

## Item-text check

calls.jsonl has no item-text column; it stores prompt/response hashes and the model's free-text `reason` (3875 non-empty, WildChat reasons blanked). Reasons sharing an 8-word run with item text: 0 (0 CounselBench). No verbatim item text found; calls.jsonl is still gitignored.
