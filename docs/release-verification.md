# Release verification

`tutorials/gpt2_text_generation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.5 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 15-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; exactly two kernel cells — the isolated
  install (pinned `uv` wheel checked by size and SHA-256, managed CPython, `--require-hashes --only-binary :all:`
  over the carried lock) and the router to the isolated worker — and seven cells titled `Infrastructure` and
  collapsed; `NOTEBOOK_SOURCE` recorded in exports; the guided-layer markers, a `pipe.reset_to_pretrained()` in
  each of Sections 4–7, no assert without a message, no doubled braces in markdown, and the absence of the removed
  text (review GPT-M1..M3, GPT-m1..m4);
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
| Google Colab (supported user path) | Colab CPU or GPU runtime (Linux x86_64; CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean one-pass top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh Linux x86_64 CPU or GPU container; the committed notebook executed verbatim with **no repository checkout** (the notebook is standalone), its first cell building the isolated environment | Reproducible clean-room executor of the same class; promotion evidence when the run is one pass |
| Local harness (pre-flight only) | Workstation, sequential cell executor with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` (the isolated environment is skipped), a `google.colab` shim and pre-installed pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/gpt2/` or the corpus cache `weights/scitldr/` (the standalone path writes the manifest itself,
   stages the missing file from the Hub, and fetches the three pinned SciTLDR-A files from the project repository,
   so neither directory may be seeded);
3. choose **Run all once** without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `BYOD_PATH = ''`, `SPLIT_SEED = 42`, `GREEDY_MAX_NEW_TOKENS = 32`, `SAMPLE_MAX_NEW_TOKENS = 32`,
   `TEMPERATURE = 0.8`, `TOP_P = 0.9`, `SEED = 7`, `EPOCHS = 2`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 8`,
   `TRAINABLE_BLOCKS = 4`, `ADAPT_SEED = 0`); the run must complete every code cell with no error output and **no
   restart** — record `restarted: false`; a run that needed a restart is not promotion evidence (RUN1, RUN10);
4. verify that Section 1 builds the isolated environment (it prints the isolated Python 3.12.12 and the number of
   locked packages), that the runtime cell reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision
   recorded in `metadata.dimer.generated_from`, and that the imported core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3` (no restart: the kernel's own packages are never replaced);
5. verify every default-path stage completes:
   - the isolated environment built from the carried hash-locked lock with no GitHub access, and every later cell
     routed to it;
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
6. verify the exports exist and the interpretation section matches the observed path; then, in the same runtime,
   the **BYOD gate (REL12)**: set `USE_BYOD = True` with a representative corpus of at least 12 unique `{id, text}`
   records and **Run after** from Section 4 — Section 6 must report `'adapted': False`, and the run must pass
   through adaptation, evaluation, export and reload with the BYOD data source in the adapter metadata — and record
   one refused input (for example an empty upload or a non-UTF-8 file) with its one-line message; and one
   documented experiment (the Section 10 activity, `TRAINABLE_BLOCKS = 1`, **Run after** from Section 7) with
   reload parity holding;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `gpt2_text_generation_colab.ipynb` (`E2E`) | `f41f14f` / `c843f865` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-gpt2-text-generation` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **Passed only after a manual restart** — not a one-pass Run all, not promotion evidence (review GPT-M1). 11/11 code cells ok after a restart following the in-kernel install cell (pass 1 stopped at the stale-import guard: `numpy` loaded 2.0.2, installed 2.5.3); 35 files, 560 MB staged from the Hub into a clean cache; comparison {perplexity: {unigram_floor: 1721, frozen: 40.17, adapted: 34.7}, bits_per_token: {unigram_floor: 10.75, frozen: 5.328, adapted: 5.117}, mean_nll: {unigram_floor: 7.451, frozen: 3.693, adapted: 3.547}, record_perplexity: {frozen: {min: 19.9, median: 40.6, max: 97}, adapted: {min: 19.1, median: 35.5, max: 85.3}}, delta_vs_frozen: {perplexity: -5.473, bits_per_token: -0.211}}; reload parity {perplexity_in_memory: 36.1, perplexity_reloaded: 36.1, identical_completions: 3, of: 3}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-gpt2-text-generation/v2/evidence/` in the workspace |
| `gpt2_text_generation_colab.ipynb` (`E2E`) | `1250eb9` / `75cb8bce` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `gpt2_text_generation_colab.ipynb` (`TASK-INFERENCE`, superseded) | `f4020ce` / `263771047488` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-gpt2-text-generation` v1) | PASSED — 9/9 code cells, 223.9 s (1 restart after the install cell); evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/gpt2_text_generation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/gpt2_text_generation_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-17 | `a99d473` (`a99d473a41521315de67460614bc014aa4257ddf`, PR #9 head) / blob `92de745b` (`92de745bb3837797fc184d72c32d4d25063f9d18`; fetched at that commit and blob-verified before the session). Evidence under `docs/execution-evidence/2026-10-17/`: `gpt2_text_generation_colab_a99d473_colab-cli-t4.ipynb` SHA-256 `2bd14c6522cad78aeb2b5a9b2d8a3e78f0e92c2ad80619058987c492f4f47ea1`, `gpt2_text_generation_colab_a99d473_colab-cli-t4_exec.log` SHA-256 `7a71e7b8e7f9deed9ce62f9e9fc2babf2ee7dadc55f4838a1a1080adfef93a43`, `gpt2_text_generation_colab_a99d473_colab-cli-t4_run_summary.json` SHA-256 `43866e1b7a7b4f709d47c5a6447f6f4361a69fea7814f019f8d2bbe4dade919d` (copied byte for byte) | Colab CLI 0.7.4 sequential execution, fresh Colab **Tesla T4** (Colab account 2; session `suite-gpt2-a99d473-f6bc` started 2026-10-17T16:05:04Z and stopped after the run) via the workspace `colab-cli-serial-test-suite`; isolated Python 3.12.12, `cuda:0`, kernel Python 3.13.15; not a browser Run all | Default settings only (the pre-registered path, as in the row below) | 269.6 s (CLI wall, including the 52 s environment build; grid 97.6 s, training 34.6 s) | **PASSED — reproduction of the pre-registered evaluation (REL13: hosted, this blob).** 15/15 code cells in order, one pass, 0 error outputs; source identity with the committed blob OK (33/33 cells, param toggles only); the notebook printed `test_look` `reproduction`, `recorded_evaluation` the 87fdb2de run, `differs_from_preregistration` `[]`. Every grid, fresh-test and exposed-test figure equals the evaluation of record: C0 kept (35.2 / 36.92 / 38.0); fresh test floor 1,674.526, pretrained 39.435 → adapted 34.187, 99 of 100 easier, CI −0.227 to −0.185 bits; exposed development test 40.18 → 34.70. Reload parity 36.027571 both ways (record: 36.027572), 3/3; see the record below |
| 2026-10-17 | `25182a0` (`25182a0c94650babc8576e6879979db826731f2c`, an earlier PR #9 head) / blob `87fdb2de` (`87fdb2de346a2e43d7e4c5fa09d638ed78666131`; fetched at that commit and blob-verified before the session). Evidence under `docs/execution-evidence/2026-10-17/`: `gpt2_text_generation_colab_25182a0_colab-cli-t4.ipynb` SHA-256 `89c8621fc4a9b221ed01c8d4ee8e638e84d654ac8e427e5e4373bb63a0f96bcf`, `gpt2_text_generation_colab_25182a0_colab-cli-t4_exec.log` SHA-256 `8b6b3006050fa8258529f872df177dd2381674fae8422eced046ebe05ebf5af9`, `gpt2_text_generation_colab_25182a0_colab-cli-t4_run_summary.json` SHA-256 `8e08a1c791eac65efd686cafcbb0808b40034e28303befabee72cc9a7bf514a7` (copied byte for byte) | Colab CLI 0.7.4 sequential execution, fresh Colab **Tesla T4** (Colab account 2; session `suite-gpt2-25182a0-f6bc` started 2026-10-17T13:47:47Z and stopped after the run, 13:52:02Z) via the workspace `colab-cli-serial-test-suite`; isolated Python 3.12.12 (`torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`), kernel Python 3.13.15; not a browser Run all | Default settings only: isolated environment → three carried modules → stage and verify 15 snapshot files → SciTLDR-A sample (300 / 50 / 100 fresh test / 100 exposed development test) → generation contract → floor and pretrained evaluation → the pre-registered validation grid (C0 / C1 / C2) → two-epoch `adapt` of the selected configuration → fresh-test evaluation → exposed development test → continuations → adapter export → reload parity → run history | 267.1 s (CLI wall, including the 48 s environment build; grid 90.0 s, training 33.8 s) | **PASSED — the pre-registered evaluation of record (REL13: hosted, this blob).** 15/15 code cells in order, one pass, no restart, 0 error outputs; the notebook printed `test_look` `pre-registered evaluation`, `preregistered` `True`, `differs_from_preregistration` `[]`. Grid (validation perplexity, best epoch 2 each): C0 35.2, C1 36.92, C2 38.0 → **C0 kept** (no challenger beat C0 by 2 %; reproduces the recorded selection). Fresh test (100 abstracts, 21,764 tokens): floor 1,674.5, pretrained 39.43, **adapted 34.19** (34.187; Δ −5.248 perplexity, −0.206 bits per token; 99 of 100 easier, bootstrap 95 % CI −0.227 to −0.185 bits). Exposed development test (not independent): 40.18 → 34.70 (Δ −5.47; 99 of 100; CI −0.231 to −0.191 bits). Reload parity 36.027572 both ways, 3/3 identical completions. See the record below |
| 2026-09-19 | `f41f14f` / `c843f865` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-gpt2-text-generation` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 251.2 s | **Passed only after a manual restart** — not a one-pass Run all, not promotion evidence (review GPT-M1). 11/11 code cells ok after a restart following the in-kernel install cell; 35 files, 560 MB staged from the Hub into a clean cache; comparison {perplexity: {unigram_floor: 1721, frozen: 40.17, adapted: 34.7}, bits_per_token: {unigram_floor: 10.75, frozen: 5.328, adapted: 5.117}, mean_nll: {unigram_floor: 7.451, frozen: 3.693, adapted: 3.547}, record_perplexity: {frozen: {min: 19.9, median: 40.6, max: 97}, adapted: {min: 19.1, median: 35.5, max: 85.3}}, delta_vs_frozen: {perplexity: -5.473, bits_per_token: -0.211}}; reload parity {perplexity_in_memory: 36.1, perplexity_reloaded: 36.1, identical_completions: 3, of: 3}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-gpt2-text-generation/v2/evidence/` in the workspace |
| 2026-09-19 | `1250eb9` / `75cb8bce` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 15 entries because the snapshot was pre-staged → `verify_snapshot` 15 files → `from_pretrained` on CPU → `fetch_corpus` served from the pre-staged cache after its digest checks → 1,992 + 619 + 618 papers read, 300 / 50 / 100 drawn with `check_split_disjoint` clean and digests `f6230198…` / `479448c6…` / `f39289da…` → four dataset refusals → input manifest with the unseeded-sampling refusal → greedy twice and seeded sampling twice on a 26-token test opening with all eight sanity checks `True` → three unseen openings continued → unigram floor → frozen evaluation → `adapt` → validation + test evaluation → before/after continuations → adapter export → reload parity) | 285.5 s | **PASSED** — 11/11 code cells; unigram floor 1,720.9; frozen test perplexity 40.17 (5.328 bits per token, 19,779 tokens, 8.0 s; per-record 19.9 / 40.6 / 97.0); greedy 32 tokens in 0.61 s, `finished_by` `max_new_tokens`; `adapt` 28,351,488 of 124,439,808 params, 300 abstracts (62,583 training tokens), 2 epochs, 248.9 s, validation perplexity 40.97 → 36.71 → 35.19 (`best_epoch` 2, train loss 3.836 → 3.681); **adapted test 34.73 (5.118 bits per token; Δ −5.44 perplexity, −0.21 bits; per-record 19.0 / 35.4 / 84.5)**; 3/3 unseen continuations changed after adaptation, single-prompt report `not-measurable`; adapter 113,410,784 B / 48 tensors, SHA-256 `b83cfce8…`; reload parity exact (ten-record perplexity 36.188268 both ways, 3/3 identical completions); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-10-04 | review-fix branch (revised notebook, before commit) | Local pre-flight harness (Windows, CPython 3.12, CPU float32, `torch 2.14.0+cpu`, `transformers 4.57.6`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, snapshot and corpus pre-staged, `CUDA_VISIBLE_DEVICES=-1`) | (1) Default path Sections 1–6 (cells 0–7): 1,992 + 619 + 618 papers, splits 300 / 50 / 100 with digests `f6230198…` / `479448c6…` / `f39289da…`, all eight generation checks `True`, unigram floor 1,720.9, pretrained test perplexity 40.17 (19,779 tokens, `'adapted': False`), validation 40.97; the two-epoch Section 7 did not finish inside the 10-minute probe window on this (shared, busy) host and was not run further. (2) 12-record BYOD CSV via `BYOD_PATH` with no `google.colab`, Run all from Section 4 (splits 8 / 2 / 2): pretrained 35.87 → adapted 35.55, `'adapted': False` in Section 6, adapter 48 tensors with data source `BYOD (my_docs.csv)`, reload parity exact. (3) Section 10 activity (`TRAINABLE_BLOCKS = 1`, Run after from Section 7, after the four-block run): epoch-0 check 0.0, 35.87 → 35.67, adapter 12 tensors, reload parity exact. (4) `SEED = 8` rerun of Sections 5–6 after adaptation: pretrained test and validation perplexity identical to the first run, `'adapted': False`. (5) Refusals with one-line messages: 11 records ("Supply at least 12 unique records"), a Latin-1 CSV (not UTF-8), `USE_BYOD = True` with no upload dialog and no `BYOD_PATH`, a cancelled upload (got 0), and a second `adapt` without a reset | 206.5 s for (2)–(5) | **PASSED** — pre-flight only, **not** promotion evidence; the isolated-environment install path (Linux) and the default two-epoch training were not exercised |
| 2026-10-04 | `091ba0c` / `d88cc65843bf` | Colab CLI 0.7.4 sequential execution, fresh Colab VM, Tesla T4 (kernel Python 3.13.15; isolated Python 3.12.12, `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda: True`, `cuda:0`; empty Hugging Face cache, no repository checkout) — not a browser Run all | Default settings only (isolated-environment build → three carried modules → stage and verify 15 snapshot files → fetch and split the SciTLDR-A sample → generation contract → unigram floor and pretrained evaluation → two-epoch `adapt` → held-out evaluation → before/after continuations → adapter export → reload parity → run history) | 153.8 s (session wall time, including the 56 s environment build) | **PASSED — one pass, no restart, 0 errors.** 14/14 code cells in order; first hosted completion of the default two-epoch Section 7 for this revision (see the record below) |
| 2026-10-17 | `d7f2c71` (package; notebook regenerated in the next commit) | CPU build record, 2026-10-17, CPython 3.12.14, `torch 2.14.0+cpu`; not a hosted run. **Disclosed development look** (REL13; fresh-test score-once rule item 8): the fixer scored the fresh test on CPU during development; no setting was changed after it (the hosted run prints `differs_from_preregistration` `[]`), and it is not the evaluation of record (the 2026-10-17 Colab T4 run above is) | `tools/reselect_on_validation.py`: the pre-registered validation grid, then the selected configuration scored on the fresh test and the exposed development test (the pre-registered evaluation, CPU build record) (not a notebook execution) | 1,028 s grid + 346 s training | C0 kept; fresh test 39.43 → 34.25; see `docs/reselection-2026-10-17.md` |
| 2026-10-17 | `e533c44` (package, round 2) | CPU, same workstation and pins as the row above; not a hosted run | **Development look, repeated on CPU** (identical settings; nothing was changed in response to the test number; under score-once rule item 8 a CPU look is a disclosed development look, and the hosted T4 run of 2026-10-17 is the evaluation of record): one process per grid configuration, then C0 on the fresh and exposed test, keeping per-record NLLs for the paired comparison (peer review G-M1) | ~5–6 min per configuration | Grid curves, pretrained and adapted fresh test, exposed test and add-one floor identical to the build record; paired fresh test: 99 of 100 abstracts easier, token-weighted Δ −0.203 bits/token, bootstrap 95 % CI −0.224 to −0.182; see `docs/reselection/paired_2026-10-17.json` |
| 2026-09-14 | `f4020ce` / `263771047488` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-gpt2-text-generation` v1) | Default sample path of the inference-only notebook: one synthetic prompt, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot` over 15 files, greedy and seeded-sampled generation with their determinism checks, `not-measurable` report, CSV + JSON exports | 223.9 s | **PASSED** — 9/9 code cells (1 restart after the install cell), 554 MB staged; history only |

### 2026-10-17 Colab CLI T4 run of `a99d473` (blob `92de745b`): reproduction of the pre-registered evaluation

REL13: reproduction of the pre-registered evaluation. The requoted notebook (prose and constants only; the
`RECORDED_EVALUATION` constant names the run below) ran the default pre-registered path on a fresh Colab Tesla T4 on
Colab account 2 (session `suite-gpt2-a99d473-f6bc`, 15/15 code cells, 0 errors, 269.6 s). It printed `test_look`
`reproduction` and `differs_from_preregistration` `[]`. A normalised line diff of the two output notebooks finds every
grid, fresh-test (floor 1,674.526; 39.435 → 34.187; paired −0.206 bits, 99/100, CI −0.227 to −0.185; quartiles; era
split), exposed-test (40.18 → 34.70) and continuation line equal. Only run-specific values differ: timings (environment
52 s vs 48 s; grid 33.1 / 31.6 / 32.9 s vs 31.9 / 28.6 / 29.5 s; training 34.6 s vs 33.8 s; greedy 0.681 s vs 0.694 s),
the carried revision, the adapter SHA-256 prefix (`5365d452…` vs `442f899d…`, same 48 tensors and 113,410,784 bytes) and
the ten-record reload-parity perplexity, 36.027571 vs 36.027572 (a 1e-6 change; both runs equal in memory and reloaded,
3/3). The saved weights differ at bit level between the two GPU runs while every printed fresh-test metric is equal.
The evaluation of record is unchanged.

### 2026-10-17 Colab CLI T4 run of `25182a0` (blob `87fdb2de`): pre-registered evaluation of record

- **Executor:** Google Colab CLI 0.7.4 (`colab exec -f`) on a fresh Colab VM with a Tesla T4, Colab account 2, through the
  workspace `colab-cli-serial-test-suite` (session `suite-gpt2-25182a0-f6bc`, 2026-10-17 13:47:47–13:52:02 UTC, stopped
  after the run). Sequential CLI execution, not a browser Run all; forms and uploads were not exercised.
- **Source identity:** fetched at the full PR #9 head `25182a0c94650babc8576e6879979db826731f2c`; Git blob
  `87fdb2de346a2e43d7e4c5fa09d638ed78666131` checked before the VM was allocated; the executed notebook's 33 cells have the
  committed ids, order and sources. `NOTEBOOK_SOURCE.repository_revision` `cf5f0877d66b`, generator `build_notebook.py/2.1`,
  notebook spec 2.5.
- **Outcome:** 15/15 code cells in order (log cells 1..15), one pass, no restart, 0 error outputs; cells 4–6 (carried
  modules) print nothing by design; the only stderr line is the `loss_type=None` notice. Environment built in 48 s.
- **Evaluation status (REL13):** the notebook printed `test_look` `pre-registered evaluation`, `preregistered` `True` and
  `differs_from_preregistration` `[]`. This is the first hosted run of the pre-registered settings for this blob, so it is
  **the pre-registered evaluation of record**; a later hosted run with identical settings is a reproduction.
- **Grid (validation only, 50 abstracts):** C0 (4 blocks, 1e-4) 40.97 → 36.59 → 35.2; C1 (2 blocks, 1e-4) 40.97 → 38.07 →
  36.92; C2 (2 blocks, 5e-5) 40.97 → 39.07 → 38.0; best epoch 2 for each (31.9 / 28.6 / 29.5 s). Selected **C0** ("no
  challenger beat C0 by 2% or more; C0 stays"), equal to the recorded selection. Paired against C0 on validation: C1 +0.069
  bits per token (bootstrap 95 % CI +0.057 to +0.080; 48 of 50 worse), C2 +0.111 (+0.092 to +0.129; 48 of 50 worse).
- **Training (C0):** 28,351,488 of 124,439,808 parameters, 62,583 training tokens, train loss 3.836 → 3.6851, validation
  40.97 → 36.59 → 35.2, best epoch 2, 33.8 s; epoch-0 relative difference to Section 6: 0.0.
- **Fresh test (pinned draw `a20180b4…`, 100 abstracts, 21,764 tokens):** add-one floor 1,674.5 (add-0.1 reference 1,204.7);
  pretrained 39.435 (5.301 bits per token; per record 14.0 / 39.7 / 97.9); **adapted 34.187** (5.095 bits; per record 13.4 /
  33.2 / 90.1); Δ −5.248 perplexity, −0.206 bits per token. Paired per abstract: 99 of 100 easier, median −0.204 bits
  (quartiles −0.287 to −0.139), token-weighted −0.206, bootstrap 95 % CI −0.227 to −0.185 (one seed). Era probe: 9-character
  ids 36.63 → 31.95 (41), others 41.69 → 35.98 (59). `adaptation_outcome` "lowered held-out perplexity".
- **Exposed development test (the 100 abstracts the 2026-09 sweep scored; not independent evidence):** 40.18 → 34.70
  (Δ −5.47, 19,779 tokens, 99 of 100 easier, bootstrap 95 % CI −0.231 to −0.191 bits).
- **Development look (disclosed, rule item 8):** the CPU build record of 2026-10-17 (`d7f2c71`, `tools/reselect_on_validation.py`,
  CPython 3.12.14, `torch 2.14.0+cpu`) had scored the fresh test at 39.43 → 34.25 (Δ −5.18) with the same C0 choice
  (validation 35.19 / 36.88 / 37.98), and its round-2 repeat (`e533c44`) gave the paired CI −0.224 to −0.182. Those are
  development looks, not the record; the hosted figures above differ from them only in the last digits (CPU float32 versus
  T4).
- **Other outputs:** 3/3 unseen continuations changed after adaptation, single-prompt report `not-measurable`; adapter 4
  blocks, 48 tensors, 113,410,784 bytes; reload parity 36.027572 both ways on the ten validation records, 3/3 identical
  completions, reloaded best epoch 2; seven exports under `outputs/`; Section 10 history one row.
- **Notebook quotes that this run contradicts (needs requote; this record does not edit the notebook):** cell 24
  validation C0 / C1 / C2 35.19 / 36.88 / 37.98 (hosted 35.2 / 36.92 / 38.0) and the paired C1 +0.068 (+0.056 to +0.079;
  47 of 50) and C2 +0.110 (+0.092 to +0.128) (hosted +0.069 (+0.057 to +0.080; 48 of 50) and +0.111 (+0.092 to +0.129));
  cell 26 C0 curve 40.97 → 36.71 → 35.19 (hosted 40.97 → 36.59 → 35.2); cell 28 adapted 34.25, Δ −5.18 and −0.20 bits
  (hosted 34.19, −5.25, −0.21), quartiles −0.285 to −0.138 (−0.287 to −0.139), CI −0.224 to −0.182 (−0.227 to −0.185),
  era probe 32.03 / 36.04 (31.95 / 35.98), exposed 40.17 → 34.73, Δ −5.44, interval −0.230 to −0.190 (40.18 → 34.70,
  −5.47, −0.231 to −0.191); cell 30 reload-parity perplexity 36.054856 (36.027572); cells 0 and 22 call the T4 grid time
  an estimate of about 1.4 min (measured 90.0 s). Equal at the printed precision: floor 1,674.5, add-0.1 1,204.7,
  pretrained 39.43 (5.30 bits), 21,764 tokens, per-record spread 14–98, 99 of 100, median −0.204, maximum 98 → 90,
  48 tensors / 113,410,784 bytes, 3/3. Follow-up: requote these in `tools/notebook_template.py` and regenerate.
  Done on 2026-10-17 in the requote commits that follow this record; the notebook now names this run in
  `RECORDED_EVALUATION`, so a later identical hosted run is labelled a reproduction.
- **Evidence files** (`docs/execution-evidence/2026-10-17/`, byte-exact copies):
  - `gpt2_text_generation_colab_25182a0_colab-cli-t4.ipynb` — SHA-256 `89c8621fc4a9b221ed01c8d4ee8e638e84d654ac8e427e5e4373bb63a0f96bcf`
  - `gpt2_text_generation_colab_25182a0_colab-cli-t4_exec.log` — SHA-256 `8b6b3006050fa8258529f872df177dd2381674fae8422eced046ebe05ebf5af9`
  - `gpt2_text_generation_colab_25182a0_colab-cli-t4_run_summary.json` — SHA-256 `8e08a1c791eac65efd686cafcbb0808b40034e28303befabee72cc9a7bf514a7`
- **Not exercised:** a browser Run all, the BYOD gate (REL12) and its refusals, the Section 10 activity and the other next
  experiments, and a Colab CPU runtime.

### 2026-10-04 Colab CLI T4 run of `091ba0c` (blob `d88cc65843bf`)

- **Executor:** Google Colab CLI 0.7.4 (`colab exec -f`) on a fresh Colab VM with a Tesla T4. The CLI executes every
  code cell in order in one kernel; it is **not** a browser Run all, it records no execution counts (order is taken
  from the `Executing cell k/14` lines in the log), and it renders no forms.
- **Source identity:** the notebook was fetched from `raw.githubusercontent.com` at the full commit
  `091ba0c229ba1c46290c13b8a8a0054288f86d4e`; its Git blob `d88cc65843bf75bcc73f313b549a91e4f27f0e31` was checked
  before the VM was allocated, and the executed notebook's 14 code-cell sources equal the committed ones.
- **Outcome:** 14/14 code cells, one pass, **no restart** (`restarted: false`), 0 error outputs, no output asks for a
  restart. Cells 4–6 (the carried `metrics`, `pipeline` and `samples` modules) print nothing by design; the only
  stderr line is the `transformers` notice that `loss_type=None` falls back to `ForCausalLMLoss`.
- **Runtime (cells 1–3):** isolated environment `/content/dimer_isolated_env`, Python 3.12.12, 47 locked packages,
  built in 56 s; every later cell routed to it; `NOTEBOOK_SOURCE.repository_revision` `083a4cb34ff6`, generator
  `build_notebook.py/2.1`, notebook spec 2.2.
- **Model (cell 7):** `openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e` (MIT), 15 files
  (554,331,411 bytes) fetched and 15 verified; device `cuda:0`.
- **Observed metrics** (observations, not a benchmark): 1,992 + 619 + 618 papers, splits 300 / 50 / 100 with digests
  `f6230198…` / `479448c6…` / `f39289da…`, four refusal probes rejected; 26-token prompt, all eight generation checks
  `True`, one recorded rejection; unigram floor 1,720.9 (10.749 bits per token, 19,779 tokens); pretrained test
  perplexity 40.175 (printed 40.18; 5.328 bits per token; per-record 19.9 / 40.6 / 97.0; `'adapted': False`),
  validation 40.97; Section 7: epoch-0 relative difference to Section 6 **0.0**, 28,351,488 of 124,439,808
  parameters, 62,583 training tokens, validation perplexity 40.97 → 36.59 → 35.2 (train loss 3.836 → 3.6851,
  `best_epoch` 2, 32.7 s); held-out test 40.175 → **34.702** (5.117 bits per token; Δ −5.473 perplexity, −0.211 bits;
  per-record 19.1 / 35.5 / 85.3), `adaptation_outcome` "lowered held-out perplexity"; 3/3 unseen continuations
  changed, single-prompt report `not-measurable`; adapter 4 blocks, 48 tensors, 113,410,784 bytes; reload parity
  exact (ten-record perplexity 36.100799 both ways, 3/3 identical completions, reloaded `best_epoch` 2); six exports
  under `outputs/`; Section 10 history one row (4 blocks, Δ −5.473). Every number equals the previous version's
  Kaggle T4 run quoted in the worked answers, up to rounding (the pretrained test perplexity prints 40.18 where the
  prose quotes 40.17; both are 40.175).
- **Evidence files** (`docs/execution-evidence/2026-10-04/`, byte-exact copies):
  - `gpt2_text_generation_colab_091ba0c_colab-cli-t4.ipynb` — SHA-256 `44f505d877fed69fdc1451cb69c3e7d009b7cd738e4f2581319e361dee8853c7`
  - `gpt2_text_generation_colab_091ba0c_colab-cli-t4_exec.log` — SHA-256 `1569a40115a2684a038e2187dd7f7e7268c341c309aa7ccf3374476a6870bf3e`
  - `gpt2_text_generation_colab_091ba0c_colab-cli-t4_run_summary.json` — SHA-256 `d41fec9b387a1105113f2f92cb5a768c7ba416897f8504063f321f039c7ba111`
- **Not exercised:** a browser Run all, the BYOD gate (REL12, step 6) and its refusals, the Section 10 activity
  (`TRAINABLE_BLOCKS = 1`) and the other next experiments on the 300-abstract sample, and a Colab CPU runtime.

## Current status

**Candidate.** On 2026-10-17 the regenerated notebook (blob `87fdb2de`, commit `25182a0`) passed on a fresh Colab Tesla T4 under the Colab CLI (15/15 code cells in order, no restart, 0 errors, 267.1 s). That run is the pre-registered evaluation of record (REL13): grid C0 kept (validation 35.2 / 36.92 / 38.0); fresh test 39.43 → 34.19 (99 of 100 easier, CI −0.227 to −0.185 bits); exposed development test 40.18 → 34.70. The requoted notebook (blob `92de745b`, PR #9 head `a99d473`) then passed on a fresh Colab Tesla T4 on 2026-10-17 (Colab account 2; Colab CLI, 15/15 code cells, 0 errors, 269.6 s). REL13: reproduction of the pre-registered evaluation, this blob — the notebook printed `test_look` `reproduction` with no setting differing, and every grid, fresh-test and exposed-test figure equals the evaluation of record. Both are sequential CLI execution, not a browser Run all. Status stays **Candidate**: promotion still needs the BYOD and experiment gates (REL12) and the maintainer's approval.

REL13: earlier-blob history (blobs `d88cc65843bf` and older). The earlier promotion rested on the Kaggle Tesla T4 run of blob `c843f865` (`f41f14f`, 2026-09-19), which completed only after a manual restart following the in-kernel install; under NOTEBOOK_SPEC 2.2 (RUN1, RUN10) that is not a one-pass Run all, so the Notebook Review Framework v1 review (2026-10-03, GPT-M1) returned the repository to Candidate. The revised notebook builds an isolated, hash-locked environment instead (Linux x86_64 only), and also fixes the review's rerun-validity (GPT-M3), guided-layer (GPT-M2), BYOD (GPT-m1), assertion (GPT-m2), template (GPT-m3), timing (GPT-m4) and pretraining-overlap (GPT-m5) findings. Its local CPU pre-flight (above) is not promotion evidence. On 2026-10-04 the current blob `d88cc65843bf` (commit `091ba0c`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 under the Colab CLI (14/14 code cells, recorded above). That run is sequential CLI execution, not a browser Run all. Status stays **Candidate**: promotion still needs the BYOD and experiment gates of step 6 (REL12) and the maintainer's approval.

### 2026-10-17 re-selection on validation (row 31 review, T31-1) — CPU build record

REL13: CPU development look (disclosed, score-once rule item 8). The hosted run of 2026-10-17 above is the evaluation of record and replaces the expected values below (fresh test 34.19, exposed 40.18 → 34.70, validation 35.2 / 36.92 / 38.0, grid 90.0 s on a T4); the open hosted run named at the end of this subsection is done.

The notebook changed after the 2026-10-04 Colab CLI run: Section 7 now replays the pre-registered validation grid,
the test split is a pinned draw that no selection step reads (digest `a20180b4…`), scored only under the pre-registered settings (see docs/release-verification.md for the recorded evaluation and its reproductions), and the
old 100 test abstracts are printed once as the exposed development test. The numbers its worked answers quote come
from `tools/reselect_on_validation.py` on a workstation CPU (`docs/reselection-2026-10-17.md`, section 2), **not**
from a hosted run of the regenerated notebook. Expected values on the default path (CPU float32; a GPU may differ in
the last digits): validation C0 / C1 / C2 = 35.19 / 36.88 / 37.98 → C0; fresh test floor 1,674.5, pretrained 39.43,
adapted 34.25; exposed development test 40.17 → 34.73. The Section 6–9 expectations listed under the 2026-09 gate
above refer to the old test split. **Open:** one hosted run of the regenerated notebook (REL13/REL14), estimated at
about 4 min on a T4 (the 2026-10-04 session took 153.8 s; the grid adds an estimated 86 s, scaled from the CPU ratio of
the grid to the four-block run), plus the BYOD and experiment gates (REL12).
