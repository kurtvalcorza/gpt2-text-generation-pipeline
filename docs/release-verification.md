# Release verification

`tutorials/gpt2_text_generation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 15-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned corpus
  commit is the one allowed second hash);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `GPT2TextGenerationPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `check_split_disjoint`, `write_dataset_csv`, `validate_inputs` and `validate_settings` with the unseeded-sampling
  refusal probe, `pipe.generate` greedy twice and seeded-sampling twice with the sanity checks plus the frozen
  continuations of the unseen openings, `pipe.unigram_baseline`, `pipe.evaluate` on the frozen model with the
  floor assertion and on the validation and test splits after adaptation with the perplexity assertion,
  `pipe.adapt` with its explicit hyperparameters and `trainable_blocks=TRAINABLE_BLOCKS`, the single-prompt
  `evaluation_report`, `pipe.save_artifact`, `GPT2TextGenerationPipeline.from_artifact` and the reload-parity
  assertion, and the provenance fields `weight_format`, `weight_sha256` and the `corpus` block), the six expected
  `outputs/` paths, the learner-facing statements (two decoding modes, base language model, no score with a
  generation, domain adaptation measured by perplexity, the add-one unigram floor, reject not truncate, no
  dispersion estimate, the pad/EOS quirk, named exclusions, the Apache-2.0 corpus licence) and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, direct `from transformers import` / `GPT2LMHeadModel` / `GPT2TokenizerFast` /
  `.generate(input_ids` / `from huggingface_hub import` / `urllib.request` / `safetensors` / `torch.optim` /
  `.backward(` / `pipe._model` / `log_softmax(` use **outside the carried module cells**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI installs only `pytest`, `ruff` and `numpy` plus the package without its model dependencies (no torch, no
transformers), runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`; injected runner, tokenizer, scorer and corpus fetcher, temporary manifests, no weights
— `tests/test_model_backed.py` is skipped without `transformers` and the snapshot). These are source/provenance and unit
checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or torch that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/gpt2/` or the corpus cache `weights/scitldr/` (the standalone path writes the manifest itself,
   stages the missing file from the Hub, and fetches the three pinned SciTLDR-A files from the project repository,
   so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `GREEDY_MAX_NEW_TOKENS = 32`, `SAMPLE_MAX_NEW_TOKENS = 32`,
   `TEMPERATURE = 0.8`, `TOP_P = 0.9`, `SEED = 7`, `EPOCHS = 2`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 8`,
   `TRAINABLE_BLOCKS = 4`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3` (an interpreter restart after the install is expected where the runtime's
   preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `GPT2TextGenerationPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `validate_settings`, `evaluation_report`, `fetch_corpus`,
     `read_corpus`, `build_sample_dataset`, `filter_records`, `validate_dataset`, `check_split_disjoint`,
     `split_dataset`, `load_byod_dataset`, `write_dataset_csv`, `sequence_metrics`, `unigram_baseline` and the
     ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `openai-community/gpt2` at the immutable revision, and `verify_snapshot` returning its dict (15 files);
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source` `local-snapshot`;
   - Section 4: `fetch_corpus` fetching the three pinned files (3,155,015 / 1,124,865 / 1,204,107 bytes) from
     `raw.githubusercontent.com` into `weights/scitldr/`, 1,992 + 619 + 618 raw papers read, and the seeded draw of
     300 / 50 / 100 records with `check_split_disjoint` reporting no shared text and the three dataset digests
     `f6230198…` / `479448c6…` / `f39289da…`; `outputs/…_train.csv` written; the four dataset refusal probes each
     raising `ValueError`;
   - Section 5: the ceilings (`CONTEXT_LENGTH` 1024, `MAX_PROMPT_TOKENS` 1023, `MAX_NEW_TOKENS` 256,
     `DEFAULT_MAX_NEW_TOKENS` 32, `MAX_TEXT_CHARS` 4000, `VOCAB_SIZE` 50257, `EOS_TOKEN_ID = PAD_TOKEN_ID` 50256,
     `MAX_TRAIN_TOKENS` 512) surfaced; `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`,
     the sampling settings 0.8 / 0.9 / 7 recorded alongside, one recorded rejection finding from the
     unseeded-sampling probe); greedy `generate` twice (byte-identical) and seeded sampling twice (same seed
     reproduces) on the opening of the first test abstract with all eight sanity checks `True`, then three unseen
     dev-abstract openings continued by the frozen model;
   - Section 6: the unigram floor (≈ 1,720.9 on the sample) and the frozen model's test perplexity
     (≈ 40.17, 5.328 bits per token, 19,779 predicted tokens) on CPU float32, with the cell's assertion
     that the token counts agree and the frozen perplexity is below the floor;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 28,351,488 trainable of 124,439,808
     parameters, 300 training abstracts, and a two-epoch history with validation perplexity falling
     (40.97 → 36.71 → 35.19 in the recorded run; `best_epoch` 2);
   - Section 8: `pipe.evaluate` on the validation and test splits with the three-way comparison and
     `outputs/…_evaluation_report.json` written (the cell asserts the adapted test perplexity is below the frozen
     one — on the sample ≈ 34.73 versus ≈ 40.17);
   - Section 9: the three unseen openings continued again by the adapted model and printed beside the frozen
     completion and the actual continuation, the single-prompt `evaluation_report` verdict `not-measurable`,
     `outputs/…_completions.csv` written; `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,
     manifest.json}` (48 tensors, about 113 MB) and `GPT2TextGenerationPipeline.from_artifact`
     reloading it with an identical ten-record perplexity and 3/3 identical completions (the cell asserts both);
     `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the snapshot block
     (`weight_format`, `weight_sha256`), the `corpus` block, the inference-contract items, the comparison, the
     before/after completions, the artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `gpt2_text_generation_colab.ipynb` (`E2E`) | `f41f14f` / `c843f865` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-gpt2-text-generation` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 35 files, 560 MB staged from the Hub into a clean cache; comparison {perplexity: {unigram_floor: 1721, frozen: 40.17, adapted: 34.7}, bits_per_token: {unigram_floor: 10.75, frozen: 5.328, adapted: 5.117}, mean_nll: {unigram_floor: 7.451, frozen: 3.693, adapted: 3.547}, record_perplexity: {frozen: {min: 19.9, median: 40.6, max: 97}, adapted: {min: 19.1, median: 35.5, max: 85.3}}, delta_vs_frozen: {perplexity: -5.473, bits_per_token: -0.211}}; reload parity {perplexity_in_memory: 36.1, perplexity_reloaded: 36.1, identical_completions: 3, of: 3}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-gpt2-text-generation/v2/evidence/` in the workspace |
| `gpt2_text_generation_colab.ipynb` (`E2E`) | `1250eb9` / `75cb8bce` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `gpt2_text_generation_colab.ipynb` (`TASK-INFERENCE`, superseded) | `f4020ce` / `263771047488` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-gpt2-text-generation` v1) | PASSED — 9/9 code cells, 223.9 s (1 restart after the install cell); evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/gpt2_text_generation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/gpt2_text_generation_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `f41f14f` / `c843f865` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-gpt2-text-generation` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 251.2 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 35 files, 560 MB staged from the Hub into a clean cache; comparison {perplexity: {unigram_floor: 1721, frozen: 40.17, adapted: 34.7}, bits_per_token: {unigram_floor: 10.75, frozen: 5.328, adapted: 5.117}, mean_nll: {unigram_floor: 7.451, frozen: 3.693, adapted: 3.547}, record_perplexity: {frozen: {min: 19.9, median: 40.6, max: 97}, adapted: {min: 19.1, median: 35.5, max: 85.3}}, delta_vs_frozen: {perplexity: -5.473, bits_per_token: -0.211}}; reload parity {perplexity_in_memory: 36.1, perplexity_reloaded: 36.1, identical_completions: 3, of: 3}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-gpt2-text-generation/v2/evidence/` in the workspace |
| 2026-09-19 | `1250eb9` / `75cb8bce` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 15 entries because the snapshot was pre-staged → `verify_snapshot` 15 files → `from_pretrained` on CPU → `fetch_corpus` served from the pre-staged cache after its digest checks → 1,992 + 619 + 618 papers read, 300 / 50 / 100 drawn with `check_split_disjoint` clean and digests `f6230198…` / `479448c6…` / `f39289da…` → four dataset refusals → input manifest with the unseeded-sampling refusal → greedy twice and seeded sampling twice on a 26-token test opening with all eight sanity checks `True` → three unseen openings continued → unigram floor → frozen evaluation → `adapt` → validation + test evaluation → before/after continuations → adapter export → reload parity) | 285.5 s | **PASSED** — 11/11 code cells; unigram floor 1,720.9; frozen test perplexity 40.17 (5.328 bits per token, 19,779 tokens, 8.0 s; per-record 19.9 / 40.6 / 97.0); greedy 32 tokens in 0.61 s, `finished_by` `max_new_tokens`; `adapt` 28,351,488 of 124,439,808 params, 300 abstracts (62,583 training tokens), 2 epochs, 248.9 s, validation perplexity 40.97 → 36.71 → 35.19 (`best_epoch` 2, train loss 3.836 → 3.681); **adapted test 34.73 (5.118 bits per token; Δ −5.44 perplexity, −0.21 bits; per-record 19.0 / 35.4 / 84.5)**; 3/3 unseen continuations changed after adaptation, single-prompt report `not-measurable`; adapter 113,410,784 B / 48 tensors, SHA-256 `b83cfce8…`; reload parity exact (ten-record perplexity 36.188268 both ways, 3/3 identical completions); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `f4020ce` / `263771047488` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-gpt2-text-generation` v1) | Default sample path of the inference-only notebook: one synthetic prompt, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot` over 15 files, greedy and seeded-sampled generation with their determinism checks, `not-measurable` report, CSV + JSON exports | 223.9 s | **PASSED** — 9/9 code cells (1 restart after the install cell), 554 MB staged; history only |

## Current status

**Release-grade.** The `E2E` notebook blob `c843f865` (committed at `f41f14f`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 251.2 s, 35 files, 560 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.
