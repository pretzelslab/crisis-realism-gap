# Data sources (third-party files live in data/raw/, gitignored; do not redistribute)

| Source | URL | Revision | Licence | Retrieved |
|---|---|---|---|---|
| Between Help and Harm crisis benchmark (examples + merged labels) | https://huggingface.co/datasets/arnaiztech/llms-mental-health-crisis-benchmark | 7f0a0d31bb8309b0e52e670d535d954d6fa7a386 | CC BY-NC 4.0 | 2026-10-07 |
| CounselBench-Adv | https://huggingface.co/datasets/izi-ano/CounselBench-Adv (GitHub: llm-eval-mental-health/CounselBench, no licence file) | 57a2e2d5b709208197bbf8ff55124d4844067da3 | CC BY-NC-ND 4.0 (HF card) | 2026-10-07 |

Notes
- Crisis benchmark: test-split labels are machine labels (gpt-4o-mini, majority of 3 runs); validation labels are human majority vote. Items are multi-turn user-side text only.
- CounselBench-Adv is NoDerivatives: the Phase 1 realism-rewrite pilot creates adapted material. Rewrites must stay local and unpublished, or get the authors' permission first.
- Deployment-style baseline: not chosen yet.

Decisions (2026-10-07)
- Crisis sample: drop duplicate texts and unresolved labels; up to 30 per crisis category (all of violent_thoughts, risk_taking_behaviors) + 40 no_crisis control = 203 items. Main text = all user turns; `text_first_turn` is a sensitivity variant.
- CounselBench-Adv: P(eval) only, labelled "authored counselling questions"; excluded from the rewrite pilot (ND licence).

## Baseline A: filtered WildChat-1M (added 2026-10-07)
- URL: https://huggingface.co/datasets/allenai/WildChat-1M, revision 7d6490e462285cf85d91eabea0f9a954fbddcd1f, licence ODC-BY 1.0 (not gated). Shards 0 and 1 of 14 downloaded to data/raw/wildchat/ (gitignored).
- Pipeline (src/build_wildchat_baseline.py): English, not toxic/redacted, first user turn 30-1500 chars, first-person + emotional/relational keyword match, no task-like keywords, dedupe, sample 200 (config seed). Counts: 119,714 rows -> 58,971 English -> 43,505 length-ok -> 474 keyword match -> 397 after dedupe -> 200 sampled. Output: data/processed/wildchat_baseline.parquet (gitignored).
- Limits: WildChat already excludes conversations flagged toxic by OpenAI Moderation/Detoxify and PII-flagged ones, so acute-crisis messages are probably under-represented (mild support-seeking only). Keyword filter precision is unvalidated. Users knew they were talking to ChatGPT. Contains real users' text: local use, aggregate scores only, no examples in outputs or memo.
- Only shards 0-1 used (~0.15 of corpus); first-turn only, so it matches the `text_first_turn` variant, not the multi-turn crisis text.

### Baseline A revision: classifier filtering (2026-10-08)
- Manual spot-check of the keyword-filtered sample put precision at ~55-65%, so the 200-item heuristic sample is superseded (kept locally as data/processed/wildchat_baseline_heuristic.parquet).
- Method (src/classify_baseline.py): all 397 deduped keyword candidates (data/processed/wildchat_candidates.parquet) are labelled y/n/unsure by claude-haiku-4-5-20251001 with a two-question rubric: (1) a real person talking about their own or a close person's life? (2) seeking help/comfort/advice about it? Both yes = y; stories, roleplay, tasks, general knowledge and bot-testing = n. Output is JSON {label, confidence, reason <=15 words}. Default sampling (the SDK exposes no temperature parameter); prompt/response hashes logged in outputs/<run_id>/calls.jsonl.
- Keep y with confidence >= 0.8, then sample up to 200 with the config seed (all if fewer). Result: 397 -> 305 n / 87 y / 5 unsure -> 66 y at >=0.8 -> baseline of 66 (fewer than the 200 target, so all kept).
- Validation: src/spotcheck.py shows 20 random kept y's locally for agree/disagree; verdicts (no text) are saved to outputs/spotcheck/. Results pending.
- Limit: the classifier is a model judgement, and a stricter filter lowers recall; the baseline now skews to clearer help-seeking messages.

## Baseline B: synthetic (placeholder)
- data/synthetic/README.md. Not generated.
