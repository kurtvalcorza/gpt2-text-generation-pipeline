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

## 2. Result (CPU build record)

Run once with `tools/reselect_on_validation.py` at commit `d7f2c71` on a Windows workstation CPU (CPython 3.12.14,
`torch 2.14.0+cpu`, `transformers 4.57.6`, float32, 24 threads). This is a **CPU build record, not a hosted run**;
the hosted run of the regenerated notebook is pending (REL13) and goes in `docs/release-verification.md`.
Records: `docs/reselection/validation_selection_2026-10-17.json` (written before any test record was scored) and
`docs/reselection/fresh_test_2026-10-17.json`.

| Config | Blocks | lr | Validation perplexity by epoch (0 → 1 → 2) | Best epoch | Score | vs C0 | Wall |
|---|---|---|---|---|---|---|---|
| C0 | 4 | 1e-4 | 40.97 → 36.71 → 35.19 | 2 | **35.19** | — | 393 s |
| C1 | 2 | 1e-4 | 40.97 → 38.10 → 36.88 | 2 | 36.88 | 4.8 % worse | 313 s |
| C2 | 2 | 5e-5 | 40.97 → 39.07 → 37.98 | 2 | 37.98 | 7.9 % worse | 322 s |

Grid wall time 1,028 s. **Selected on validation: C0** (no challenger beat C0 by 2 % or more; the incumbent stays).
The defaults (`TRAINABLE_BLOCKS = 4`, `LEARNING_RATE = 1e-4`) are unchanged; they are now the configuration selected
on validation by the rule above rather than the one chosen from test scores. The selected configuration was then
trained once more (346 s; its validation curve repeated the grid run exactly) and scored once:

| Split | Records / tokens | Unigram floor | Pretrained | Adapted (C0) | Δ perplexity | Δ bits/token |
|---|---|---|---|---|---|---|
| Fresh test (never scored before) | 100 / 21,764 | 1,674.5 | 39.43 | **34.25** | −5.18 | −0.20 |
| Exposed development test (old 100, scored in 2026-09) | 100 / 19,779 | — | 40.17 | 34.73 | −5.44 | −0.21 |

Per-record perplexity on the fresh test: 14.0 / 39.7 / 97.9 (min / median / max) pretrained, 13.3 / 33.4 / 89.7
adapted. Adapter: 48 tensors, 113,410,784 bytes; reload parity on ten validation records 36.054856 both ways, 3/3
identical completions; 3/3 unseen-abstract continuations changed by the adaptation.
