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
