# Release verification

`tutorials/gpt2_text_generation_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until the exact notebook revision has
executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime
evidence under DIMER Notebook Specification 1.0. This file is the durable release-gate record for
the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile and the notebook-spec version; `metadata.dimer` declares that profile and spec `1.0`;
- the fresh-runtime bootstrap (clone by canonical URL, `DIMER_TUTORIAL_REF`, detached checkout of
  the requested revision, restart-on-stale-import guard) and the recorded `REPO_SHA` in exports;
- `MODEL_ID`/`MODEL_REVISION` are imported from the package rather than hard-coded, the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `GPT2TextGenerationPipeline.from_pretrained`, `validate_settings` for both configurations,
  `generate` greedy twice and seeded-sampling with explicit `temperature`/`top_p`/`seed`), the
  ceiling constants imported from the package, the settings-echo, determinism and `finished_by`
  sanity checks, the learner-facing statements (two decoding modes, no adaptation, base language
  model, no quality metric, no metric reported, greedy deterministic, mandatory seed, pad/EOS quirk,
  reject-not-truncate) and the gated-off BYOD default listed in the validator; forbidden patterns
  (credential-in-URL, direct `transformers` or `huggingface_hub` calls that bypass the pipeline,
  `GPT2LMHeadModel`, `GPT2TokenizerFast`, `AutoModelForCausalLM`, `AutoTokenizer`, `model.generate(`,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

These are source/provenance checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim, cell by cell, in a fresh interpreter with a `google.colab` shim and `DIMER_TUTORIAL_REF` set to the candidate commit | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or Pillow that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, empty model cache, no pre-staged `model.safetensors` under `weights/gpt2/` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container
   executor above) with `DIMER_TUTORIAL_REF` set to the candidate commit, an empty Hugging Face
   cache, and no pre-staged `model.safetensors` under `weights/gpt2/`;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `GREEDY_MAX_NEW_TOKENS = 32`,
   `SAMPLE_MAX_NEW_TOKENS = 16`, `TEMPERATURE = 0.8`, `TOP_P = 0.9`, `SEED = 7`);
4. verify that Section 1 reports `repository_revision` equal to the candidate commit and that the
   installed core package versions equal the `pyproject.toml` pins (`torch==2.14.0`,
   `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`);
5. verify every default-path stage completes:
   - fresh bootstrap from GitHub at the candidate revision;
   - the synthetic prompt authored in code with its SHA-256 printed;
   - ceilings `CONTEXT_LENGTH = 1024`, `MAX_PROMPT_TOKENS = 1023`, `MAX_NEW_TOKENS = 256`,
     `DEFAULT_MAX_NEW_TOKENS = 32`, `MAX_TEXT_CHARS = 4000`, `VOCAB_SIZE = 50257`,
     `EOS_TOKEN_ID = PAD_TOKEN_ID = 50256` printed, both settings dicts returned by
     `validate_settings` (greedy with `None` temperature/top_p/seed; nucleus-sampling with 0.8/0.9/7)
     and the prompt accepted before model execution;
   - `stage_missing_files(..., allow_download=True)` reporting `['model.safetensors']` fetched from
     `openai-community/gpt2` at the immutable revision, `verify_snapshot` returning the 15-entry
     manifest, and `GPT2TextGenerationPipeline.from_pretrained` loading from the verified directory
     with `source 'local-snapshot'`;
   - greedy `generate` returning 7 prompt tokens and up to 32 new tokens with all seven sanity checks
     true, including a byte-identical repeat call, and the "no metric is reported" line printed;
   - seeded sampling returning up to 16 new tokens with all six sanity checks true, including a
     same-seed repeat that reproduces the completion, and the settings echoed as
     `nucleus-sampling` / 0.8 / 0.9 / 7;
   - `outputs/gpt2_text_generation_result.json` written with the prompt, both results (completion,
     text, token counts, `finished_by`, echoed settings), the checks, ceilings, repository SHA,
     model identifier, immutable model revision, snapshot summary, runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, NumPy,
   device), model identifier and immutable revision, whether the model cache and weights directory
   were clean, outcome, produced outputs, both completions with `finished_by` (as observations, not a
   metric), and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `tutorials/gpt2_text_generation_colab.ipynb` | | | | pending — queued to the GPU lane |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/gpt2_text_generation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/gpt2_text_generation_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

No execution of the notebook has been recorded. The only runtime measurements that exist for this
repository are the pipeline smoke run documented in `MODEL_CARD.md` (Windows venv, CPU float32,
`HF_HUB_OFFLINE=1`: `verify_snapshot` 0.32 s over 15 files, load 4.25 s, 32 greedy tokens from the
same 7-token prompt in 0.67 s with a byte-identical repeat, two seeded 16-token samples with seed 7
identical to each other in 0.59 s together). That run exercised the package, not this notebook, and
is not notebook execution evidence.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| — | — | — | Default sample path | — | pending — queued to the GPU lane |

## Current status

The notebook source is complete and passes the static checks above; **no clean-runtime execution
has been recorded**, so the registry status is **Candidate** and the manual-evidence row is pending.
Promotion requires a reviewer to confirm a recorded run against the notebook blob under review and
an integrator to promote it; promotion is not performed by the builder. The commit that adds a
recorded-execution row changes documentation only; the executed source is the commit named in the
row.
