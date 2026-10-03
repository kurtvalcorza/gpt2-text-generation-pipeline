# GPT-2 124M Domain-Adaptation E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/gpt2-text-generation-pipeline`  
**Notebook:** `tutorials/gpt2_text_generation_colab.ipynb`  
**Reviewed commit:** `1d93245155a5495217066f768e1ee2408f51c9a0` (`main`, confirmed with `gh api repos/kurtvalcorza/gpt2-text-generation-pipeline/commits/main`)  
**Notebook Git blob:** `c843f865192cc3adae8bcceb03188f03d2f6d9b2`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-19 (commit `f41f14f`); the notebook last changed in `724307d` (PR #3). Later commits on `main` touch the model card, tests and the validator, not the notebook or its carried modules.  
**Finding prefix:** `GPT`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.  
**Scope:** the generated `E2E` tutorial, the only notebook in `tutorials/`.

## Executive assessment

The engineering is careful. The notebook carries its three modules verbatim (`tools/build_notebook.py --check` and `tools/validate_release_assets.py` both pass at this commit). It digest-verifies the 15-file GPT-2 snapshot before loading it, fetches three digest-pinned SciTLDR-A files and draws a paper-disjoint 300 / 50 / 100 split, and shows four dataset refusals and an unseeded-sampling refusal. Both decoding modes are run with eight sanity checks. Perplexity is read against an add-one unigram floor and the frozen model, the last four blocks are fine-tuned with validation-perplexity epoch selection, and a safetensors adapter is reloaded with a perplexity and completion parity assertion. The closing interpretation is careful about what perplexity does and does not establish. A CPU run of the data stage in this review reproduced the recorded split exactly.

| Measure | This review (CPU) | Kaggle T4 record (blob `c843f865`) |
|---|---|---|
| Code cells completed | cells 5, 7, 9, 13 (carried modules + data stage); model cells not run with GPT-2 | 11/11 on pass 2; pass 1 stopped at the install guard |
| SciTLDR fetch | 1,992 + 619 + 618 papers, digests match, 2.7 s | identical |
| Split digests | `f6230198…` / `479448c6…` / `f39289da…` | identical |
| BYOD smallest accepted corpus | **12 records** (8, 10, 11 refused at the split) | not run |
| Rerun of Section 7 with `TRAINABLE_BLOCKS = 1` (stand-in model) | reload parity **fails** (perplexity 40,278.5 in memory vs 45,388.8 reloaded) | not run |
| Test perplexity unigram / frozen / adapted | not run | 1,720.9 / 40.18 / **34.70** |

Three problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (GPT-M1).** The recorded run stopped at the install cell's stale-module guard (`numpy: loaded=2.0.2, installed=2.5.3`) and passed only after a restart. `docs/release-verification.md` step 4 says a restart "is expected", yet the opening cell promises `Run all` with no intervention and the repository marks the blob `Release-grade` on that run.
2. **Guided layer largely absent (GPT-M2).** The notebook is declared `GUIDED`, but it has no audience statement, how-to-use section, roadmap, glossary, prediction prompt, checkpoint, troubleshooting section or conclusion template. The 1,114 lines of carried modules sit in three unlabelled, uncollapsed cells.
3. **The documented reruns measure against the wrong model and can crash (GPT-M3).** BYOD is to be run by re-running "from that cell" (Section 4) after the full run. The experiments change `TRAINABLE_BLOCKS`, `EPOCHS` or `SEED` and re-run. The model is loaded only in Section 3, and `adapt` starts from the pipeline's current weights and labels them "frozen model". The "frozen" numbers and continuations on a rerun therefore come from the already adapted model. The `TRAINABLE_BLOCKS = 1` experiment also exports an adapter that does not reproduce the in-memory model, so the Section 9 parity assertion fails (stand-in probe).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites (cell 1): basic Python; next-token prediction; argmax decoding vs sampling from a truncated distribution; what perplexity is |
| Supported runtime | "a fresh supported runtime (Google Colab or Jupyter, Python 3.12)"; CPU float32 by default, CUDA used when present |
| Promised outcomes | Pinned install; carried package; digest-verified snapshot; digest-pinned SciTLDR-A fetch, validation and leakage-free split; inference contract in both decoding modes with input manifest and rejection probe; unigram floor and frozen perplexity; bounded fine-tuning with explicit hyperparameters and validation epoch selection; independent test evaluation with a three-way comparison; before/after continuations; safetensors adapter export with verified reload parity; BYOD (CSV/JSON/JSONL/TXT) through the same cells |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `40cfc802` |
| Release status | **`Release-grade`** (`STATUS.md`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` Current status) |

### Evidence actually obtained

- **Source inspection.** I read all 25 cells (11 code). Cells 5, 7 and 9 carry `metrics.py` (82 lines), `pipeline.py` (738 lines) and `samples.py` (294 lines). I also read the relevant parts of `pipeline.py` (`from_pretrained`, `evaluate`, `adapt`, `save_artifact`, `load_artifact`, `from_artifact`) and `samples.py` (`split_dataset`, `load_byod_dataset`), the generator and template, `README.md`, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/`.
- **Documented execution evidence.** `docs/release-verification.md` row 2026-09-19 and the workspace run archive (`.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-gpt2-text-generation/v2/evidence/`: `run_summary.json`, `executed-pass1.ipynb`, `executed.ipynb`, `outputs/`). The run used a Kaggle Tesla T4, **the reviewed blob** (verified before execution) and a clean HF cache. Pass 1 failed with `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` Pass 2 completed 11/11. There is no Colab run of this blob, no hosted CPU run, no BYOD run and no experiment run.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py` on Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), Python 3.12.14, torch 2.13.0+cpu, transformers 4.57.6, numpy 2.5.3, all taken read-only from an existing workspace conda env. The installed torch is one minor version below the 2.14.0 pin. Nothing was installed, and the GPT-2 checkpoint was not downloaded.
  - **Probes (about 10 s plus the fetch):**
    - P0 static checks.
    - P1 the three carried cells executed in one namespace.
    - P2 cell 13 at defaults from an empty working directory, with a real HTTPS fetch of the three pinned files.
    - P3 the BYOD loader and split with 8–13-record CSVs, a TXT corpus and two malformed CSVs.
    - P4 **stand-in model**: a randomly initialised 12-block GPT-2 (`n_embd = 32`) with a byte tokenizer, wired like `from_pretrained`'s closures. It exercises what `adapt`, `evaluate`, `save_artifact` and `load_artifact` do on a rerun. It says nothing about the real model's numbers.
    - P5 `tools/build_notebook.py --check` (exit 0) and `tools/validate_release_assets.py` (exit 0, "static source validation only").
- **Not verified:** Sections 3 and 5–9 with the real model beyond the Kaggle record; any Colab run; any hosted CPU run; the real upload dialog; BYOD beyond the split; every optional experiment with the real model.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness.** Supply-chain and data-integrity handling is strong: each corpus file is refused on any digest mismatch, P2 digests are identical to the Kaggle record and generator parity holds. The defects are three:
  - the install pattern forces a restart (GPT-M1);
  - `adapt` starts from the pipeline's current weights and labels them "frozen model", and `save_artifact` captures only the blocks of the latest call, so the documented reruns compound training and can break reload parity (GPT-M3);
  - bare assertions turn a legitimate negative result into a crash (GPT-m2).
- **Scientific validity.** The design is sound for the default sample:
  - the floor and the frozen model are scored on the same 19,779 test tokens (asserted);
  - the epoch is selected on validation only, and the test split is untouched until the final evaluation;
  - the split is paper-disjoint, and the unseen openings come from `dev` records outside the validation draw;
  - the delta is labelled sample evidence with no dispersion estimate.

  One gap remains: the notebook does not state that these specific abstracts may overlap GPT-2's pretraining text (GPT-m5).
- **Promise fulfilment.** On the documented run the default-path promises are met, except one-pass `Run all` (GPT-M1). The BYOD path's stated minimum is wrong and it needs `google.colab` (GPT-m1). Its documented rerun measures against the adapted model (GPT-M3), and a non-improving adaptation stops before export (GPT-m2).
- **Learner experience.** The prose is precise. "Look for" and "Watch" notes appear in Sections 4, 7 and 8, before/after continuations are printed beside the actual text, and the closing interpretation is careful. Against that, there is no guided layer and no troubleshooting (GPT-M2). The experiments name no rerun scope (GPT-M3), template brace artefacts and a mid-word prompt slice mar the output (GPT-m3), and the timing claims name no environment (GPT-m4).
- **Spec conformance.**
  - Unresolved applicable MUSTs: RUN1, RUN10, ENV6, REL2, REL11 (GPT-M1); DAT13, DAT14 (GPT-M3); DAT12, DAT19 (GPT-m1); DAT13 (GPT-m2); UX12 (GPT-m4); DAT9 (GPT-m5).
  - SHOULD deviations: GDL1–GDL14, UX8 (GPT-M2); GDL10, UX5 (GPT-M3); EXE2, UX10 (GPT-m1); RUN9, UX10 (GPT-m2); SRC10 (GPT-m3); EXE5 (GPT-S2).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` "with no configuration edit" | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 11/11 | cell 2 warns the cell may stop with a restart instruction; cell 0 promises no intervention | **Not met** (GPT-M1) |
| Digest-verified pinned snapshot | cell 11 | 15/15 fetched and verified, `cuda:0`, `local-snapshot` (Kaggle) | clear | Met (documented) |
| Digest-pinned SciTLDR-A fetch, validation, four refusals | cell 13 | P2: 1,992 / 619 / 618 papers, three digests identical to the record; Kaggle: four refusals with clear messages | "Look for" note matches | Met |
| Split "without leakage" | `build_sample_dataset` from the release's own train/dev/test members + `check_split_disjoint` | 300 / 50 / 100, no shared text | cell 24 advises splitting by collection or author | Met |
| Inference contract in both decoding modes | cell 15 | Kaggle: all eight checks `True`, one recorded rejection, `finished_by` `max_new_tokens` | well explained; "no score" stated | Met (documented) |
| Perplexity beside a unigram floor, frozen model | cell 17 | Kaggle: 1,720.9 / 40.18, equal token counts asserted | well explained, definitions printed | Met (documented) |
| Bounded fine-tune, explicit hyperparameters, validation selection | cell 19 → `adapt` | Kaggle: 28,351,488 of 124,439,808 trainable, val 40.97 → 36.59 → 35.20, best epoch 2 | "Watch" note matches; fine-tune `seed` (0) not exposed or printed | Met (documented) |
| Independent test evaluation, three-way comparison | cell 21 | Kaggle: 1,720.9 / 40.18 / 34.70; per-record spread printed | labelled sample evidence | Met (documented) |
| Before/after continuations | cells 15, 23 | Kaggle: 3/3 changed, printed beside the actual continuation | "read as text, not evidence" | Met (documented); prompt print is a mid-word slice (GPT-m3) |
| Adapter export and fresh reload with parity | cell 23 | Kaggle: 48 tensors, ten-record perplexity 36.1 both ways, 3/3 completions | explained | Met (documented) on the default path; fails after the `TRAINABLE_BLOCKS = 1` experiment (GPT-M3) |
| BYOD through the same cells, "8..20,000 records" | cell 13 BYOD branch; rerun "from that cell" | P3: 8, 10, 11 records refused at the split; 12 accepted. Source: rerun reuses the adapted model as "frozen" | contract states 8 | **Not met as stated** (GPT-m1, GPT-M3) |
| Optional experiments (`TRAINABLE_BLOCKS = 1` "watch the gain shrink", `EPOCHS = 4`, `SEED`) | cell 24 prose only | P4: second `adapt` starts from the adapted weights; parity fails after the blocks change | no rerun scope, no prediction | **Not met** (GPT-M3) |

| Learning objective (cell 0) | Learner activity | Evidence it is exercised |
|---|---|---|
| Install the pinned runtime | run cell 3 | yes, but needs a restart (GPT-M1) |
| Read what the carried modules guarantee | read cells 5, 7, 9 (1,114 lines) | no reading guide or checkpoint (GPT-M2) |
| Stage and digest-verify the snapshot | run cell 11, read the dicts | yes |
| Fetch, validate and split a corpus without leakage | run cell 13, read four refusals | yes |
| Generate in both modes and read `finished_by`, token counts, settings | run cell 15 | yes; no question asks the learner to read them |
| Read perplexity beside a unigram floor | run cell 17 | yes; no prediction or checkpoint |
| Fine-tune with explicit hyperparameters and validation selection | run cell 19 | yes; the experiments that would exercise it are broken on rerun (GPT-M3) |
| Evaluate on an independent test split | run cell 21 | yes |
| Compare continuations before and after | read cell 23 output | display only, no interpretation prompt |
| Export an adapter that reloads with verified parity | run cell 23 | yes on the default path |

## 4. Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Technically precise but not guided. The audience is unstated; there is no how-to-use section, roadmap, glossary, prediction, checkpoint or troubleshooting; and 1,114 lines of infrastructure sit in expanded cells between the install and the first model cell (GPT-M2). The install cell may stop with a restart instruction that the opening never mentions (GPT-M1). |
| Clean default | Documented execution (Kaggle T4, reviewed blob); direct execution of cells 5, 7, 9 and 13 on CPU | Completes only after a manual restart (GPT-M1). The data stage reproduces exactly. No Colab and no hosted CPU run, though CPU is the documented default (GPT-S3). |
| Active learning | Direct execution with a stand-in model + source inspection | The documented experiments re-run `adapt` on the already adapted pipeline. Epoch 0 is labelled "frozen model" but is the adapted model (P4: 42,621.0 = the first run's best validation perplexity). Changing `SEED` and re-running Section 5 overwrites the "before" continuations with adapted ones. `TRAINABLE_BLOCKS = 1` exports one block while four are changed in memory, so the parity assertion fails (P4) (GPT-M3). Not verified with the real model. |
| Reuse and recovery | Direct execution of the BYOD loader and split; source inspection | Wrong-column CSV: clear message. 8–11-record corpora pass the stated contract but fail at the split with a message that does not give the required total (GPT-m1). The upload needs `google.colab` (Jupyter is a supported runtime) and there is no location field (GPT-m1). Re-running from Section 4 measures BYOD against the SciTLDR-adapted model (GPT-M3). A non-improving adaptation stops on a bare `AssertionError` before export (GPT-m2). The real upload dialog and BYOD past the split were not verified. |

## 5. Findings

### Major

#### GPT-M1 — `Run all` needs a manual restart after the in-kernel install

- **Cell/section:** cell 3 (Section 1), generated by `tools/build_notebook.py` (install cell body, stale-module guard at line 69). Status claims are in `STATUS.md`, `README.md`, `tutorials/README.md` and `docs/release-verification.md`.
- **Observed issue:** cell 3 `pip install`s exact pins (`torch==2.14.0`, `numpy==2.5.3`, …) into the running kernel and raises `RuntimeError(... Restart the runtime, then rerun from the top.)` when an already-imported distribution changed. On the recorded Kaggle run it fired (`numpy: loaded=2.0.2, installed=2.5.3`; `cuda-bindings 12.9.4 → 13.4.2`), and the run passed only on a second pass after a restart. `docs/release-verification.md` step 4 states that "an interpreter restart after the install is expected where the runtime's preinstalled torch or numpy differ from the pins", and its own executor table says the in-kernel path "correctly halts". Cell 0 still promises that **Run all** completes with "no configuration edit (NOTEBOOK_SPEC 2.0 §5)", and the repository marks the blob `Release-grade` on that two-pass run.
- **Consequence:** a learner who selects Run all in Colab (or Kaggle) gets an exception at the first code cell. Every recorded "pass" depended on a restart that a learner must discover and perform.
- **Evidence:** documented execution (`executed-pass1.ipynb` error output; `run_summary.json` passes; release-verification rows "1 restart after install cell"); source inspection. Colab behaviour is inferred and not verified, but the repository's own procedure expects the restart.
- **Recommended correction:** adopt the fleet's uv isolated-environment pattern. A carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that environment, so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on `main`. Implement it in `tools/build_notebook.py`, not by hand. Until a one-pass run is recorded, return the registry to `Candidate` in all four status documents.
- **Acceptance check:** a fresh Colab runtime (CPU, the documented default) runs the regenerated notebook top to bottom in one pass with no restart and no error output in any cell. The run is recorded in `docs/release-verification.md` with commit, blob, runtime and outcome, and no status document claims `Release-grade` for a blob whose only record needed a restart.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11 (MUST).

#### GPT-M2 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** whole notebook; template `tools/notebook_template.py` (opening and per-section markdown), carrier cells from `tools/build_notebook.py`.
- **Observed issue:** P0 finds none of: an intended-learner statement, **How to use this notebook**, a roadmap, a glossary, a prediction prompt before a principal result, **What to notice** / **Check your reasoning** with sample answers, a troubleshooting section, or an evidence-based conclusion template. The "Learners" and "predict" text matches are the paper title and "predicted tokens". The carried modules (82 + 738 + 294 lines) sit in three code cells with no `# @title Infrastructure: …` and no `cellView: form`, between the install cell and the first model cell. Sections end without a synthesis. Partial credit goes to the "Look for" (Sections 4, 8) and "Watch" (Section 7) notes, the stated data contract and the careful closing interpretation.
- **Consequence:** a learner new to causal-LM fine-tuning gets no route through the notebook, cannot tell infrastructure from lesson, and is never asked to predict or interpret the perplexity numbers that are the notebook's central result. Expected hosted failures (no GPU, download error, upload cancel, out-of-memory) have no recovery text.
- **Evidence:** source inspection; P0 marker scan (`results.json` → `P0_static.guided_markers_in_markdown`, `cellView_form_cells: []`).
- **Recommended correction:** add the guided layer in the template, following the NOTEBOOK_SPEC 2.2 §25.13 reference notebook. The layer needs:
  - an audience line, a how-to-use section and a roadmap (GDL1–GDL3);
  - an Input → Model → Output contract and a glossary covering perplexity, bits per token, teacher forcing, the unigram floor, greedy vs nucleus decoding and trainable blocks (GDL4, GDL6);
  - a prediction prompt before Sections 6 and 8 ("Will the frozen model be closer to the floor or to 1?", "How much will four blocks and two epochs move the test perplexity?") (GDL7);
  - **What to notice** notes and collapsible **Check your reasoning** answers (GDL8, GDL9);
  - a Predict → Change one thing → Run → Observe → Explain activity, together with GPT-M3 (GDL10);
  - the carrier cells titled `Infrastructure` and collapsed (GDL11), with section tags (GDL12);
  - a troubleshooting section (GDL13) and a conclusion template (GDL14);
  - a one-line synthesis at the end of each section (UX8).
- **Acceptance check:** the regenerated notebook contains each GDL1–GDL14 element (or a one-line justification for an omission recorded in `tutorials/README.md`). Every carried-module and install cell has `cellView: form` and an `Infrastructure` title, and Sections 6 and 8 each have a prediction prompt and a collapsible sample answer.
- **Spec:** GDL1–GDL14, UX8 (SHOULD).

#### GPT-M3 — Documented reruns reuse the adapted model as "frozen", and one experiment breaks reload parity

- **Cell/section:** cell 0 (BYOD: "set `USE_BYOD = True` in Section 4 and re-run from that cell"), cell 24 (optional experiments: `TRAINABLE_BLOCKS = 1`, `EPOCHS = 4`, change `SEED` in Section 5), cells 15, 17, 19, 23; `pipeline.py` `adapt` and `save_artifact`. Template: `tools/notebook_template.py:37`, `:447`; `tools/build_notebook.py:384`.
- **Observed issue:**
  - `GPT2TextGenerationPipeline` is built once, in Section 3. `adapt` trains the model in place, starting from the weights the pipeline currently holds, records them as epoch 0 with `note: "frozen model"`, and keeps the best epoch.
  - `save_artifact` writes only the blocks named by the **latest** `adapt` call. `from_artifact` overlays those onto a fresh base.
  - On every documented rerun, then:
    - **BYOD from Section 4:** Section 5's `before` continuations and Section 6's `frozen_test` are produced by the SciTLDR-adapted model. Section 6 even prints `'adapted': True` for its "frozen model" line (P4: the "frozen" test perplexity on rerun is 40,838.5 vs 49,896.7 on the first run, `adapted: True`). The BYOD adapter is SciTLDR + BYOD training, but its manifest names only the BYOD source.
    - **`EPOCHS = 4`, Section 7 onward:** training continues from the two-epoch model. Epoch 0 "frozen model" is the adapted model (P4: 42,621.0, the first run's best validation perplexity), and the reported delta vs the Section 6 frozen number mixes six epochs of training.
    - **`TRAINABLE_BLOCKS = 1`, Section 7 onward:** the model starts from four adapted blocks and trains one more. The "gain" is larger, not smaller as the notebook says to expect. The exported adapter holds block 11 only, while blocks 8–11 differ from the base in memory, so the Section 9 parity assertion fails (P4: ten-record perplexity 40,278.494 in memory vs 45,388.840 reloaded; assertion would raise).
    - **`SEED`, Section 5:** re-running cell 15 also recomputes `before` with the adapted model, so a later rerun of Section 9 shows "frozen" = "adapted".
  - No cell names the rerun scope, so the learner has no way to know that Section 3 must be re-run first.
- **Consequence:** the learner's main comparison (frozen vs adapted) silently becomes adapted vs adapted. The block-count experiment teaches the opposite of what it promises and then crashes in Section 9, and the BYOD export misdescribes what it contains.
- **Evidence:** direct execution with a **stand-in** 12-block random GPT-2 (P4; control flow only); source inspection of `adapt` and `save_artifact`. Not verified with the real checkpoint.
- **Recommended correction:**
  - Snapshot the base state when the model is loaded, and make `adapt` restore it before training unless the caller opts into continued training (`from_base=True` default). Alternatively, have the generator emit a rebuild of `pipe` from the verified snapshot at the top of Sections 4 and 7.
  - Have Section 6 refuse to label a model "frozen" when `pipe.adapter is not None`.
  - State the exact rerun scope for BYOD and each experiment ("re-run from Section 3", or "Runtime → Run after" on a named cell).
  - Ask for a prediction before each experiment (GDL10).
  - Add regression tests to `tests/test_adaptation.py` (injected tiny model) for: second `adapt` epoch-0 equals the base; reload parity after `trainable_blocks` changes between calls.
- **Acceptance check:** with the regenerated notebook, (a) each documented rerun followed exactly yields a Section 6 "frozen" perplexity equal to the first run's and `adapted: False`; (b) the `TRAINABLE_BLOCKS = 1` experiment passes the Section 9 parity assertion and yields a smaller delta vs frozen than the default; (c) a BYOD rerun's artifact manifest describes only BYOD training. (a) and (b) are reproducible offline with the repository's injected-model test harness.
- **Spec:** DAT13, DAT14 (MUST); GDL10, UX5 (SHOULD).

### Minor

#### GPT-m1 — BYOD: stated minimum is wrong, the upload needs `google.colab`, and there is no location field

- **Cell/section:** cell 1 data contract ("a dataset needs 8..20,000 records"); cell 13 BYOD branch (`from google.colab import files; files.upload()`); `samples.split_dataset`.
- **Observed issue:**
  - P3: corpora of 8, 10 and 11 records pass `validate_dataset` but are refused by `split_dataset` (`split leaves 5 training records; at least 8 are required`); 12 is the smallest accepted. The message gives the training count, not the total the learner must supply.
  - The branch imports `google.colab`, so in Jupyter (a stated supported runtime) `USE_BYOD = True` fails with `ModuleNotFoundError`. There is no `BYOD_PATH` field (EXE2).
  - A cancelled upload makes `next(iter({}))` raise a bare `StopIteration` (source inspection).
- **Consequence:** a learner who sizes a corpus from the stated contract is refused after uploading, without being told how many records to add. Jupyter users cannot use BYOD at all.
- **Evidence:** direct execution (P3); source inspection.
- **Recommended correction:** state the effective minimum (12 records with the default fractions) in the contract and in the `split_dataset` error ("supply at least 12 records"). Add a `BYOD_PATH = ''  # @param` field read before any `google.colab` import. Turn an empty upload into `ValueError("no file was uploaded; …")`. Implement this in the template and `samples.py`.
- **Acceptance check:** the contract text and the split error both name the effective minimum. A 12-record CSV given by `BYOD_PATH` runs in a non-Colab kernel through validation and split, and an empty upload yields a message naming the next action.
- **Spec:** DAT12, DAT19 (MUST); EXE2, UX10 (SHOULD).

#### GPT-m2 — Bare assertions crash a legitimate negative result before export

- **Cell/section:** cell 17 (`frozen < unigram`), cell 21 (`adapted_test['perplexity'] < frozen_test['perplexity']`), cell 23 (reload parity); template `tools/notebook_template.py:271`, `:345`, `:396`.
- **Observed issue:** the three asserts carry no message (P0). On BYOD or after an experiment, an adaptation that does not lower test perplexity is a legitimate result the notebook itself says to read ("read the unigram floor before the adapted number"). It raises an empty `AssertionError` in Section 8 and skips Section 9: no completions, no adapter, no `result.json`.
- **Consequence:** the learner sees a stack trace with no explanation and loses the export path exactly when the result most needs interpreting.
- **Evidence:** source inspection; not executed with the real model.
- **Recommended correction:** keep the cell-21 check as an assertion only on the default sample path (`if not USE_BYOD`). Otherwise print a labelled "adaptation did not lower held-out perplexity" verdict into the report and continue. Give every remaining assert a message naming the failed condition and the next action.
- **Acceptance check:** a BYOD or experiment run whose adapted perplexity is ≥ frozen completes Section 9 and records the negative verdict in `evaluation_report.json`, and every assert in the notebook has a message.
- **Spec:** DAT13 (MUST); RUN9, UX10 (SHOULD).

#### GPT-m3 — Template brace artefacts and a mid-word prompt slice in learner-facing text

- **Cell/section:** cell 0 BYOD paragraph and cell 1 data contract (`tools/notebook_template.py:38`, `:115`); cell 23 (`:375`).
- **Observed issue:**
  - The markdown renders `{{id, text}}` and the id pattern as `[A-Za-z0-9_.:-]{{1,64}}`. These are format-string escapes that were never formatted (P0: cells 0 and 1). As printed, the regex is not the pattern the code enforces.
  - Section 9 prints `head[-60:]`, which cuts the prompt mid-word (Kaggle record: `'tputs that are naturally expressed as sets of entities. This'`).
- **Consequence:** small but misleading. A learner copying the id rule gets a wrong regex, and the before/after table opens on a garbled prompt.
- **Evidence:** source inspection; documented execution output.
- **Recommended correction:** un-escape the braces in the two markdown strings (or route them through the same formatter as the code cells), and print the full prompt or slice at a word boundary.
- **Acceptance check:** the rendered notebook contains no `{{` in markdown, and every printed prompt starts at a word boundary.
- **Spec:** SRC10 (SHOULD).

#### GPT-m4 — Runtime claims name no environment

- **Cell/section:** cell 1 ("the build record measured about 4 s … 8.5 s … about 114 s per training epoch"), cell 0 ("about five minutes of model time" on CPU), cell 16 ("about ten seconds on CPU"), cell 18 ("about 114 s of training … on CPU").
- **Observed issue:** these are measured figures from the local Windows pre-flight harness (`docs/release-verification.md` row `1250eb9`, which itself records 8.0 s and a 248.9 s `adapt`), but the notebook does not say so. The only hosted record is a T4 (`adapt` 31.6 s). No hosted CPU timing exists.
- **Consequence:** a learner on a Colab CPU runtime cannot tell whether a slower run is normal.
- **Evidence:** source inspection; documented execution evidence.
- **Recommended correction:** label each figure with its environment ("measured on a workstation CPU, Python 3.12, torch 2.14") or call it an estimate, and add the hosted CPU timing once GPT-S3 is done.
- **Acceptance check:** every runtime figure in the notebook either names its measuring environment or is marked as an estimate.
- **Spec:** UX12 (MUST).

#### GPT-m5 — Pretraining overlap with the sample is not stated

- **Cell/section:** cells 0, 16, 24.
- **Observed issue:** the notebook says "WebText of 2019 already contains scientific prose". It does not say that GPT-2's pretraining text may contain these very SciTLDR abstracts or close variants. Many SciTLDR papers were public on OpenReview and arXiv before WebText was collected, and this has not been ruled out.
- **Consequence:** the frozen perplexity (≈ 40) may partly reflect memorisation. A learner could read it as the model's general competence on unseen scientific text.
- **Evidence:** source inspection. Overlap itself is not verified either way.
- **Recommended correction:** add one sentence to Section 6 and the limits: overlap between WebText and these abstracts cannot be ruled out. The frozen-vs-adapted delta is measured on the same records and remains a like-for-like comparison, but the absolute frozen number may be optimistic.
- **Acceptance check:** the notebook states the overlap limitation where the frozen perplexity is first interpreted.
- **Spec:** DAT9 (MUST).

### Suggestions

- **GPT-S1 — Regenerate against NOTEBOOK_SPEC 2.2.** Metadata, cell 0, `NOTEBOOK_SOURCE` and `tutorials/README.md` declare 2.0.
- **GPT-S2 — Document `DIMER_NOTEBOOK_CI_PREINSTALLED`** in the notebook (cell 3 reads it; no markdown mentions it) (EXE5).
- **GPT-S3 — Record a hosted CPU run.** CPU is the documented default runtime, but the only hosted record is a Kaggle T4. Record a Colab CPU run of the next blob.
- **GPT-S4 — Show the training curve and explain the warning.** Show the validation-perplexity history and the per-record perplexity spread as a small table or plot. Expose and print the `adapt` seed (fixed at 0, not shown). Say in Section 7 that the `loss_type=None … Using the default loss` transformers warning is harmless.

## 6. Readiness

**Needs revision.**

- **Blockers:** none.
- **Open Majors:**
  - GPT-M1: no one-pass `Run all`; the only hosted record needed a restart.
  - GPT-M2: the guided layer is absent in a `GUIDED` notebook.
  - GPT-M3: documented reruns compare against the wrong model, and one experiment fails reload parity.
- **Unresolved applicable MUSTs:** RUN1, RUN10, ENV6, REL2, REL11, DAT9, DAT12, DAT13, DAT14, DAT19, UX12.
- **Remaining gates after fixes:**
  1. A one-pass Colab run of the new blob, recorded in `docs/release-verification.md`.
  2. A BYOD run (REL12): representative 12+-record corpus accepted, one incompatible input rejected clearly, through export and reload.
  3. One documented experiment run following its stated rerun scope.

The `Release-grade` status in `STATUS.md`, `README.md`, `tutorials/README.md` and `docs/release-verification.md` should return to `Candidate` until gate 1 is met.

## 7. Verified vs inferred

- **Verified by direct execution (CPU, this review):**
  - the carried modules execute;
  - the data stage reproduces the recorded split and digests;
  - BYOD minimum, message text and wrong-column handling;
  - the generator `--check` and the static validator pass;
  - `adapt` / `save_artifact` rerun semantics on a stand-in model.
- **Verified from documented evidence:** the reviewed blob's Kaggle T4 run, including its install-guard failure and second-pass success.
- **Inferred from source:**
  - the Colab restart (expected by the repository's own procedure);
  - the bare-assert crash on a negative BYOD result;
  - the `StopIteration` on a cancelled upload;
  - the BYOD-rerun artifact mislabelling.
- **Only Kurt or a hosted run can confirm:** Colab behaviour of the install cell, real-model timings on a Colab CPU, and the real-model numbers of each experiment.
- **Finding most likely to be wrong:** GPT-M3's parity failure for `TRAINABLE_BLOCKS = 1` is shown on a randomly initialised stand-in. With the real GPT-2 the in-memory/reloaded gap could be small, but it would still exceed the cell's 1e-6 tolerance whenever blocks 8–10 moved during the first `adapt`. The "frozen = adapted" part follows directly from the source and does not depend on the stand-in.

## Probe bundle

`gpt2_text_generation_colab_Review_Probes.zip` contains:

- `run_probes.py`: P0–P5, run as `python run_probes.py <repo> <out>`;
- `results.json`;
- `source_manifest.json`: SHA-256 of the 12 inspected source files at `1d93245`.
