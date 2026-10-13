# Synthetic baselines (Baseline B and Baseline C)

Status: generated. Provenance, counts and limits are in data/SOURCES.md; generation prompts are in docs/phase1_appendix.md (section B).

- `baseline_b.jsonl`: 163 synthetic single-turn crisis messages (claude-haiku-4-5-20251001), written per docs/baseline_b_guidelines.md (v2). 70 carry an unrelated everyday side question.
- `baseline_b_noask.jsonl`: the same 163 items with the 70 side-question items regenerated without the side question.
- `baseline_b_review.csv`: manual review sheet for B.
- `baseline_c.jsonl`, `baseline_c_topic_mix.json`: 163 synthetic non-crisis messages, topic mix matched to the 200 WildChat items.

Purpose: sensitivity baselines for Baseline A (filtered WildChat-1M), which stays primary.
Rules: no method details, no real PII.
Weakness: synthetic text may itself look fake to a model.
