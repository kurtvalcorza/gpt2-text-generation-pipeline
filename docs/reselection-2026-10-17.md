# Re-selection of the adaptation defaults on validation (2026-10-17)

Review finding T31-1 (row 31, tutorial triage 2026-10-15) found that the tutorial's adaptation defaults
(`TRAINABLE_BLOCKS = 4`, `LEARNING_RATE = 1e-4`) were picked from a three-configuration sweep whose recorded
figures were **test** perplexities (PR #3: 2 blocks @ 5e-5 → 37.39, 2 blocks @ 1e-4 → 36.32, 4 blocks @ 1e-4 →
34.73 on the 100 test abstracts). The owner ruled (2026-10-17): re-select on the validation split with the
grid below, then score the chosen configuration once on a fresh, never-scored test draw.

## 1. Pre-registration

This section was committed **before** any configuration below was trained or scored. It is not edited after
the run; results go in section 2 and in `docs/reselection/validation_selection_2026-10-17.json`.

### Grid

| Config | `TRAINABLE_BLOCKS` | `LEARNING_RATE` | Role |
|---|---|---|---|
| C0 | 4 | 1e-4 | incumbent default (kept unless beaten) |
| C1 | 2 | 1e-4 | smaller |
| C2 | 2 | 5e-5 | smaller and gentler |

### Fixed for every configuration

- `EPOCHS = 2`, with the best epoch (0, 1 or 2) chosen on validation perplexity, as `adapt` already does.
- `BATCH_SIZE = 8`; AdamW with weight decay 0.01; gradient clipping at 1.0; no scheduler.
- `ADAPT_SEED = 0`; `SPLIT_SEED = 42`; `MAX_TRAIN_TOKENS = 512`; float32.
- The same 300 training and 50 validation abstracts (SciTLDR-A `train` and `dev` members, seeded draw with
  `SPLIT_SEED = 42`; validation digest `479448c6…`). Each configuration starts from the pretrained weights.

### Selection rule

1. A configuration's score is its **best-epoch validation perplexity** (the lowest of epochs 0..2). A score whose
   best epoch is 0 means "no adaptation".
2. **Margin:** a challenger (C1 or C2) replaces C0 only if its score is lower than C0's by **at least 2 %
   relative**, `(C0 − challenger) / C0 ≥ 0.02`.
3. If both challengers clear the margin, take the lower score; if the two are within 2 % of each other
   (`(higher − lower) / lower < 0.02`), take the smaller configuration (fewer blocks, then lower learning rate).
4. Otherwise C0 stays. Ties go to the smaller configuration.

The rule is implemented once, in `select_configuration` (`src/gpt2_text_generation_pipeline/metrics.py`), and
the grid is run by `reselect_on_validation` (`pipeline.py`); `tools/reselect_on_validation.py` and the
tutorial's Section 7 call the same functions.

### Test sets

- **Fresh test (scored once):** 100 abstracts drawn from the `test` member's never-scored remainder. The
  remainder is the 618 filtered `test` abstracts minus the 100 of the `SPLIT_SEED = 42` test draw (518), minus
  the 3 whose text also occurs in the `train` or `dev` member (515). They are taken in file order, shuffled with
  `random.Random(20261017)`, and the first 100 are kept. The draw is pinned in `samples.py` by its 100 paper ids
  (`FRESH_TEST_PAPER_IDS`) and digest `a20180b4219a3be5e75fc450406b854ba0b98758c1de92d78f306211aa7f8488`
  (ids `fresh-0000`..`fresh-0099`); `build_fresh_test` refuses any other draw. Only the selected configuration is
  scored on it, once; nothing is re-run or re-selected after it is seen.
- **Exposed development test (reported once, for continuity):** the previous 100 test abstracts (digest
  `f39289da…`), which the 2026-09 sweep scored. They are reported once, labelled, and are never used to choose
  anything.
- `build_tutorial_dataset` asserts that train, validation, fresh test and exposed test share no text and no
  paper id.

### What is reported

All three validation scores with their per-epoch history, best epochs and wall times; the selected
configuration and the rule branch that chose it; then, for the selected configuration only, the fresh-test
perplexity (pretrained and adapted, with the unigram floor) and the exposed-test perplexity (pretrained and
adapted). The local run is a **CPU build record**, not a hosted run (REL13); the hosted run of the regenerated
notebook is recorded in `docs/release-verification.md` when it exists.
