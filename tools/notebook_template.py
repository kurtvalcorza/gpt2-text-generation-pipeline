"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E domain-adaptation workflow: the pinned GPT-2 124M snapshot is
digest-verified and loaded, a digest-pinned real corpus (SciTLDR paper abstracts) is fetched, validated and
split, the inference contract is exercised in both decoding modes, the frozen model's held-out perplexity is
read beside a unigram floor, a bounded fine-tuning of the last transformer blocks adapts the model to the
domain in the kernel, the held-out split is scored again, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "gpt2_text_generation_pipeline",
    "repo_name": "gpt2-text-generation-pipeline",
    "stem": "gpt2_text_generation",
    "notebook_name": "gpt2_text_generation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned GPT-2 snapshot (safetensors, 548 MB), fetches the three digest-pinned SciTLDR-A files from the project "
        "repository (5.5 MB, no credential), reads the paper abstracts and draws 300 / 50 / 100 training, validation and "
        "test documents from the release's own paper-disjoint members, generates greedy and seeded-sampled continuations "
        "of an unseen abstract's opening through the inference contract with an input manifest and a rejection probe, "
        "scores the frozen model on the test abstracts by held-out perplexity beside the add-one unigram floor, runs a "
        "bounded fine-tuning of the last four transformer blocks with validation-perplexity epoch selection, scores the "
        "held-out split again, continues the same unseen abstracts with the adapted model, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify parity. The default path "
        "needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about five minutes of model time after the downloads; a "
        "CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "text corpus as a CSV (columns `id`, `text`), a JSON array or JSONL file of `{{id, text}}` records, or a plain "
        "`.txt` file in which every blank-line-separated paragraph is one document. It passes through the same validation, "
        "seeded text-disjoint split, unigram floor, frozen scoring, fine-tuning, held-out evaluation, generation, artifact "
        "export and reload-parity cells as the SciTLDR sample. The expected schema and the ceilings are stated in the "
        "Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of "
        "the default path."
    ),
    "pipeline_class": "GPT2TextGenerationPipeline",
    "weights_key": "gpt2",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "GPT-2 124M — DIMER E2E domain-adaptation tutorial: perplexity on paper abstracts (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/gpt2-text-generation-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/gpt2-text-generation-pipeline/blob/main/tutorials/gpt2_text_generation_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-openai--community%2Fgpt2-ffcc4d?style=flat",
            "https://huggingface.co/openai-community/gpt2",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-openai%2Fgpt--2-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/openai/gpt-2",
        ),
    ],
    "capability": "causal text generation (continuing one English prompt, greedy by default, seeded nucleus sampling on request) and bounded causal-LM fine-tuning of the last transformer blocks on a text corpus, measured by held-out perplexity, using the pinned `openai-community/gpt2` weights",
    "intro": (
        "At inference the byte-level BPE tokenizer turns the prompt into token ids (no special tokens are added), and "
        "the 12-block decoder-only Transformer predicts one next-token distribution over the 50,257-token vocabulary at "
        "a time, feeding each chosen token back until `max_new_tokens` is reached or the end-of-text token is produced. "
        "**Two decoding modes are demonstrated and must not be confused:** greedy decoding (`do_sample=False`, the "
        "pipeline default) takes the argmax at every step and is deterministic on a fixed device and dtype; nucleus "
        "sampling (`do_sample=True` with `temperature`, `top_p` and a mandatory `seed`) draws from the truncated "
        "distribution, reproducible only for the same seed on the same host. GPT-2 is a **base language model**: no chat "
        "template, no instruction following, no safety tuning, English web text of 2019 vintage. The carried pipeline "
        "module adds manifest verification, input validation and ceilings (prompts are rejected, never truncated), a "
        "settings validator that refuses unseeded sampling, the fixed pad/EOS handling and a fixed output contract. "
        "**A generation carries no score**; the number a language model *does* have is how surprised it is by text it "
        "did not write.\n\n"
        "What this notebook adds to inference is **domain adaptation measured by that number**. The dataset is real: "
        "SciTLDR-A (Cachola et al., 2020; Apache-2.0) ships the abstracts of 3,229 computer-science papers as three "
        "digest-pinned JSON-Lines files fetched from the project repository at a pinned commit; only the abstract text is "
        "used, so every record is one document and no reference is needed. The carried `metrics.py` turns the model's own "
        "teacher-forced per-token negative log-likelihoods into **perplexity** and **bits per token** on the held-out "
        "abstracts, and fits an **add-one unigram model** on the training tokens as the floor a model that ignores "
        "context reaches. The fine-tuning question is whether a bounded adaptation of the last transformer blocks on 300 "
        "abstracts lowers the perplexity of abstracts the model has not seen. Nothing here is a quality claim: a lower "
        "perplexity says the adapted model finds paper abstracts less surprising, not that it writes good ones."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, dataset and metrics modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned referenced corpus and validate and split it "
        "without leakage; generate through the public API in both decoding modes and read `finished_by`, the token counts "
        "and the echoed settings correctly; read a perplexity beside a unigram floor and understand what it does and does "
        "not measure; run a bounded fine-tuning with explicit hyperparameters and validation-based epoch selection; "
        "evaluate on an independent test split; compare continuations before and after; and export a safetensors adapter "
        "that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "chat or instruction following (GPT-2 has neither), batching (one prompt per call), raw logits or hidden states, "
        "beam search, non-English text, full-model or embedding fine-tuning, any quality, fluency or factuality score, "
        "and any claim that a SciTLDR abstract split stands in for your corpus. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate: the build record measured about 4 s to load and digest-verify the 548 MB snapshot, 8.5 s to score the 100-abstract test split (19,779 predicted tokens) and about 114 s per training epoch over 300 abstracts plus a 50-abstract validation pass per epoch. The pinned `torch==2.14.0` install and the 548 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what next-token prediction is; the difference between argmax decoding and sampling from a truncated distribution; what perplexity is (the exponential of the mean per-token negative log-likelihood) and why it is an intrinsic number, not a judgement of the text a model writes.",
        "- **Data contract:** records are `{{id, text}}` — one document of the domain, 1..4,000 characters and at most 1,023 BPE tokens (a longer record is refused, not truncated, when scored), ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a dataset needs 8..20,000 records; texts are de-duplicated case-insensitively before splitting so the same document never sits in two splits; during training only, documents are truncated to 512 tokens and the end-of-text token is appended. BYOD accepts CSV, JSON, JSONL or TXT in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a record belongs to the domain you mean — an off-topic corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal document collection is exactly that. The default path uploads nothing. A base language model can continue any prompt with false, biased or offensive text — read the output before reusing it.",
        "- **External access (data):** besides the Hub, the default path fetches three pinned objects (`train.jsonl` 3,155,015 bytes, `dev.jsonl` 1,124,865 bytes, `test.jsonl` 1,204,107 bytes; SHA-256 `b222771d…` / `3191fa98…` / `fb42dd6c…`) from `raw.githubusercontent.com` at the pinned `allenai/scitldr` commit over HTTPS, each refused on any mismatch before it is read; SciTLDR is Apache-2.0 (Cachola et al., 2020).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and split\n\n"
                "`fetch_corpus` downloads the three pinned SciTLDR-A files (or reads them from the cache), refuses a "
                "byte-size or SHA-256 mismatch per file before it is parsed, and `read_corpus` flattens each JSON-Lines "
                "member into records whose `text` is the abstract's sentences joined by a space — titles and TLDRs are "
                "left unread. `build_sample_dataset` keeps abstracts of 200..2,400 characters, drops repeated texts, and "
                "draws 300 training documents from the `train` member, 50 validation documents from `dev` and 100 test "
                "documents from `test` by a seeded shuffle — the release's own paper-disjoint partition. `validate_dataset` "
                "then checks every record against the contract, `check_split_disjoint` asserts no text appears in two "
                "splits, and the training split is written to `outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 1,992 + 619 + 618 raw papers, three digests, splits 300 / 50 / 100, and four refusal probes — "
                "a duplicate id, an empty text, a missing field and a dataset too small to split — each rejected before "
                "`torch` does anything."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_papers = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/scitldr'))\n"
                "    raw_papers = {{name: len(part) for name, part in corpus.items()}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_papers': raw_papers, 'splits': disjoint, 'file_sha256': {{k: v[2][:12] + '...' for k, v in CORPUS_FILES.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_texts': manifest['unique_texts'], 'text_chars': manifest['text_chars'], 'total_chars': manifest['total_chars'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'example': {{'id': train_records[0]['id'], 'text': train_records[0]['text'][:200] + '...'}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty text': [{{**train_records[0], 'text': '   '}}, *train_records[1:8]],\n"
                "    'missing field': [{{'id': r['id']}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Generate through the inference contract in both decoding modes\n\n"
                "Before any adaptation, the inference contract is exercised as it always was. The prompt is the opening "
                "of one test abstract (about 120 characters); `validate_inputs` applies exactly the checks `generate` "
                "applies (text type and character ceiling, `max_new_tokens` and the sampling settings within their "
                "ceilings) and returns an input manifest; the prompt-token ceilings need the real tokenizer and are "
                "enforced inside `generate`, which **rejects, never truncates**. An unseeded sampling request is validated "
                "too and its rejection recorded as a finding. `generate` returns `completion`, `text`, `prompt_tokens`, "
                "`new_tokens`, `finished_by` (`eos` when the model emitted token 50256 — GPT-2 ships no pad token, so the "
                "pipeline fixes `pad_token_id = eos_token_id = 50256` — or `max_new_tokens`) and echoes the settings. The "
                "greedy call is made twice and must repeat byte-identically (the reproducibility contract, on one device and "
                "dtype); the seeded sample is made twice and must repeat for the same seed. **Score semantics:** the "
                "pipeline emits **no probability, confidence or score of any kind** with a generation. Three further "
                "unseen-abstract openings are continued greedily here and kept as the *before* column for Section 9."
            ),
            "code": (
                "import time\n\n"
                "GREEDY_MAX_NEW_TOKENS = 32  # @param {{type:\"integer\"}}\n"
                "SAMPLE_MAX_NEW_TOKENS = 32  # @param {{type:\"integer\"}}\n"
                "TEMPERATURE = 0.8  # @param {{type:\"number\"}}\n"
                "TOP_P = 0.9  # @param {{type:\"number\"}}\n"
                "SEED = 7  # @param {{type:\"integer\"}}\n\n"
                "def opening(record, chars=120):\n"
                "    head = record['text'][:chars]\n"
                "    return head.rsplit(' ', 1)[0] if ' ' in head else head\n\n"
                "prompt = opening(test_records[0])\n"
                "ceilings = {{'CONTEXT_LENGTH': CONTEXT_LENGTH, 'MAX_PROMPT_TOKENS': MAX_PROMPT_TOKENS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'VOCAB_SIZE': VOCAB_SIZE, 'EOS_TOKEN_ID': EOS_TOKEN_ID, 'PAD_TOKEN_ID': PAD_TOKEN_ID, 'MAX_TRAIN_TOKENS': MAX_TRAIN_TOKENS}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(prompt, max_new_tokens=GREEDY_MAX_NEW_TOKENS, names=['test-opening'])\n"
                "sampling_settings = validate_settings(SAMPLE_MAX_NEW_TOKENS, True, TEMPERATURE, TOP_P, SEED)\n"
                "input_manifest['sampling_settings'] = sampling_settings\n"
                "try:\n"
                "    validate_inputs(prompt, max_new_tokens=SAMPLE_MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'unseeded-sampling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "started = time.perf_counter()\n"
                "greedy = pipe.generate(prompt, max_new_tokens=GREEDY_MAX_NEW_TOKENS)\n"
                "greedy_seconds = round(time.perf_counter() - started, 3)\n"
                "greedy_repeat = pipe.generate(prompt, max_new_tokens=GREEDY_MAX_NEW_TOKENS)\n"
                "sampled = pipe.generate(prompt, max_new_tokens=SAMPLE_MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, seed=SEED)\n"
                "sampled_repeat = pipe.generate(prompt, max_new_tokens=SAMPLE_MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, seed=SEED)\n"
                "checks = {{\n"
                "    'greedy_settings_echoed': greedy['settings'] == input_manifest['settings'] and greedy['settings']['decoding'] == 'greedy',\n"
                "    'sampled_settings_echoed': sampled['settings'] == sampling_settings and sampled['settings']['seed'] == SEED,\n"
                "    'new_tokens_within_budget': greedy['new_tokens'] <= GREEDY_MAX_NEW_TOKENS and sampled['new_tokens'] <= SAMPLE_MAX_NEW_TOKENS,\n"
                "    'prompt_tokens_within_ceiling': 1 <= greedy['prompt_tokens'] <= MAX_PROMPT_TOKENS,\n"
                "    'text_is_prompt_plus_completion': greedy['text'] == greedy['prompt'] + greedy['completion'],\n"
                "    'finished_by_is_known': greedy['finished_by'] in ('eos', 'max_new_tokens'),\n"
                "    'greedy_repeat_is_identical': greedy_repeat['completion'] == greedy['completion'],\n"
                "    'same_seed_reproduces': sampled_repeat['completion'] == sampled['completion'],\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'generate output failed a sanity check: {{checks}}')\n"
                "print({{'prompt_tokens': greedy['prompt_tokens'], 'greedy': {{'new_tokens': greedy['new_tokens'], 'finished_by': greedy['finished_by'], 'seconds': greedy_seconds}}, 'sampled': {{'new_tokens': sampled['new_tokens'], 'finished_by': sampled['finished_by'], 'decoding': sampled['settings']['decoding']}}, 'checks': checks, 'findings': len(input_manifest['findings']), 'no_score': 'the pipeline emits no probability or quality score'}})\n"
                "print(f'prompt:  {{prompt!r}}')\n"
                "print(f\"greedy:  {{greedy['completion']!r}}\")\n"
                "print(f\"sampled: {{sampled['completion']!r}}\")\n"
                "if USE_BYOD:\n"
                "    unseen_records = [{{**r, 'id': f'unseen-{{i:02d}}'}} for i, r in enumerate(test_records[1:4])]\n"
                "else:\n"
                "    used = {{r['text'].lower() for part in splits.values() for r in part}}\n"
                "    unseen_records = [{{**r, 'id': f'unseen-{{i:02d}}'}} for i, r in enumerate([r for r in filter_records(corpus['dev']) if r['text'].lower() not in used][:3])]\n"
                "before = {{r['id']: pipe.generate(opening(r), max_new_tokens=GREEDY_MAX_NEW_TOKENS)['completion'] for r in unseen_records}}\n"
                "print({{'unseen_openings_continued_by_the_frozen_model': len(before)}})"
            ),
        },
        {
            "md": (
                "## 6. The unigram floor and the frozen model's perplexity on the test split\n\n"
                "Two numbers frame the adaptation. `pipe.unigram_baseline` fits an add-one-smoothed unigram model over the "
                "50,257-token vocabulary on the training tokens and scores the test tokens with it: the perplexity a "
                "model that knows the domain's word frequencies but ignores every context reaches — expect a number in "
                "the thousands. `pipe.evaluate` scores the same test abstracts under teacher forcing with the frozen model: "
                "every token after a record's first is predicted from the tokens before it, the per-token negative "
                "log-likelihoods are averaged token-weighted and exponentiated (`perplexity`) or divided by ln 2 "
                "(`bits_per_token`); `record_perplexity` gives the spread across documents. Records over "
                "`MAX_PROMPT_TOKENS` are refused, never truncated. The build record saw the frozen model near 40 on "
                "these abstracts (WebText of 2019 already contains scientific prose); about ten seconds on CPU."
            ),
            "code": (
                "t0 = time.perf_counter()\n"
                "unigram = pipe.unigram_baseline(train_records, test_records)\n"
                "print({{'unigram_floor': {{'perplexity': round(unigram['perplexity'], 1), 'bits_per_token': round(unigram['bits_per_token'], 3), 'n_tokens': unigram['n_tokens'], 'baseline': unigram['baseline']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_model_test': {{'perplexity': round(frozen_test['perplexity'], 2), 'bits_per_token': round(frozen_test['bits_per_token'], 3), 'n_records': frozen_test['n_records'], 'n_tokens': frozen_test['n_tokens'], 'record_perplexity': {{k: round(v, 1) for k, v in frozen_test['record_perplexity'].items()}}, 'verdict': frozen_test['verdict'], 'adapted': frozen_test['adapted']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "assert frozen_test['n_tokens'] == unigram['n_tokens'] and frozen_test['perplexity'] < unigram['perplexity']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning on the training abstracts\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_BLOCKS` transformer blocks — four by default, "
                "28,351,488 of 124,439,808 parameters; the token and position embeddings, the tied output projection, the "
                "final layer norm and the earlier blocks stay frozen — with next-token cross-entropy on every token of "
                "every training abstract (the end-of-text token is appended so the model also learns where a document "
                "ends), AdamW at a fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler. "
                "Documents are truncated to `MAX_TRAIN_TOKENS` (512) **during training only**. Epoch 0 records the frozen "
                "model's validation perplexity; every epoch is scored the same way, and the epoch with the lowest "
                "validation perplexity is kept.\n\n"
                "Watch validation perplexity fall from about 41 by a few points over two epochs (about 114 s of "
                "training plus a validation pass per epoch on CPU). The build record's sweep on this sample: two blocks at 5e-5 reached 37.39, two blocks at 1e-4 reached 36.32, four blocks at 1e-4 reached 34.73 in about the same time — the default."
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_BLOCKS = 4  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_perplexity'] = round(entry['val']['perplexity'], 2)\n"
                "        row['val_bits_per_token'] = round(entry['val']['bits_per_token'], 3)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_blocks=TRAINABLE_BLOCKS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'train_tokens': adapt_result['n_train_tokens'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and no abstract in it appears in the "
                "training or validation splits. The adapted model is scored exactly as the frozen model was in Section 6, "
                "and the three numbers are put side by side. Look for a perplexity a few points below the frozen one and "
                "far below the unigram floor; the cell asserts the adapted perplexity is lower than the frozen. One "
                "hundred abstracts from one seeded split of one corpus give no dispersion estimate; the delta is "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a lower perplexity on "
                "paper abstracts says nothing about your corpus until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{\n"
                "    metric: {{'unigram_floor': round(unigram[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}}\n"
                "    for metric in ('perplexity', 'bits_per_token', 'mean_nll')\n"
                "}}\n"
                "comparison['record_perplexity'] = {{'frozen': {{k: round(v, 1) for k, v in frozen_test['record_perplexity'].items()}}, 'adapted': {{k: round(v, 1) for k, v in adapted_test['record_perplexity'].items()}}}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in ('perplexity', 'bits_per_token')}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'unigram_floor': unigram}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['perplexity'] < frozen_test['perplexity']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Continue unseen abstracts before and after, export the adapter and reload it\n\n"
                "The three abstract openings continued by the frozen model in Section 5 are continued again by the "
                "adapted model through the same `generate` contract, and the two completions are printed side by side "
                "with the abstract's actual continuation. Read them as text, not as evidence: a perplexity gain is "
                "measured on the model's likelihoods, and nothing here scores a completion. The single-prompt "
                "`evaluation_report` helper — the inference-stage helper — is written for the first of them and stays "
                "`not-measurable`, because a continuation has no ground truth.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last four transformer blocks, about 113 MB — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "training configuration and the epoch history (OUT8). `GPT2TextGenerationPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest and digest **before** deserialising, refuses "
                "any tensor that is not a transformer-block tensor of the base, and overlays the tensors onto a freshly "
                "loaded base — a new object from files, not the in-memory model (VER2). The cell asserts an identical "
                "test perplexity on ten records and identical greedy completions (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "rows = []\n"
                "for record in unseen_records:\n"
                "    head = opening(record)\n"
                "    after = pipe.generate(head, max_new_tokens=GREEDY_MAX_NEW_TOKENS)\n"
                "    rows.append({{'id': record['id'], 'prompt': head, 'frozen_completion': before[record['id']], 'adapted_completion': after['completion'], 'actual_continuation': record['text'][len(head):len(head) + 160], 'prompt_tokens': after['prompt_tokens'], 'new_tokens': after['new_tokens'], 'finished_by': after['finished_by']}})\n"
                "    print({{'id': record['id'], 'prompt': head[-60:]}})\n"
                "    print({{'frozen': rows[-1]['frozen_completion']}})\n"
                "    print({{'adapted': rows[-1]['adapted_completion']}})\n"
                "    print({{'actual': rows[-1]['actual_continuation']}})\n"
                "single_report = evaluation_report(pipe.generate(opening(unseen_records[0]), max_new_tokens=GREEDY_MAX_NEW_TOKENS), sample_kind='one unseen SciTLDR abstract opening' if not USE_BYOD else 'one BYOD test record')\n"
                "print({{'single_prompt_report_verdict': single_report['verdict'], 'adapted_differs_from_frozen': sum(r['frozen_completion'] != r['adapted_completion'] for r in rows), 'of': len(rows)}})\n"
                "with open('outputs/{stem}_completions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(rows)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = GPT2TextGenerationPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "parity_records = test_records[:10]\n"
                "ppl_pair = (pipe.evaluate(parity_records)['perplexity'], reloaded.evaluate(parity_records)['perplexity'])\n"
                "gen_pair = [(r['adapted_completion'], reloaded.generate(r['prompt'], max_new_tokens=GREEDY_MAX_NEW_TOKENS)['completion']) for r in rows]\n"
                "parity = {{'perplexity_in_memory': round(ppl_pair[0], 6), 'perplexity_reloaded': round(ppl_pair[1], 6), 'identical_completions': sum(a == b for a, b in gen_pair), 'of': len(gen_pair)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert abs(ppl_pair[0] - ppl_pair[1]) < 1e-6 and parity['identical_completions'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'files': {{k: {{'name': v[0], 'bytes': v[1], 'sha256': v[2]}} for k, v in CORPUS_FILES.items()}}, 'license': CORPUS_LICENSE}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'prompt': prompt, 'greedy': {{k: greedy[k] for k in ('completion', 'prompt_tokens', 'new_tokens', 'finished_by', 'settings')}}, 'sampled': {{k: sampled[k] for k in ('completion', 'prompt_tokens', 'new_tokens', 'finished_by', 'settings')}}, 'seconds_greedy_first_call': greedy_seconds}},\n"
                "    'comparison': comparison,\n"
                "    'completions_before_after': rows,\n"
                "    'single_prompt_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen 2019 model already finds paper abstracts far less surprising than a unigram model does (about 40 "
        "against a floor in the thousands — WebText contains scientific prose), and a bounded fine-tuning of the last "
        "four transformer blocks on 300 abstracts lowers held-out perplexity by a few points in a few minutes on CPU, "
        "with a 113 MB adapter that reloads to identical likelihoods and completions. That is the claim: the adaptation "
        "contract can adapt the model to a domain end to end on a real corpus, and the number it produces is read against "
        "the frozen model and a context-free floor rather than in isolation.\n\n"
        "Perplexity is intrinsic: it says how well the model predicts text it did not write, token-weighted, on one seeded "
        "split of one corpus with no dispersion estimate. It is not fluency, factuality, usefulness or safety, and a lower "
        "perplexity does not make the completions in Section 9 better — they are unscored continuations from a base "
        "language model and can be false, repetitive, biased or offensive. The adapter changes the last blocks, which every "
        "prompt shares, so the model's behaviour on other text shifts too; nothing here measures that. Greedy decoding "
        "repeats itself; seeded sampling is reproducible only on the same host.\n\n"
        "Three things to carry to real data. **Floors first:** the unigram floor and the frozen perplexity on *your* "
        "held-out documents are the numbers to read before any adapted one. **Leakage:** de-duplicate texts across splits "
        "(the contract does this case-insensitively) and split by document collection or author when your documents come "
        "from one. **Ceilings:** documents over `MAX_PROMPT_TOKENS` are refused when scored and truncated to 512 tokens "
        "only during training — long documents need chunking that this pipeline does not provide.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real corpus, validate "
        "the demonstrated dataset contract without leakage, execute the inference contract in both decoding modes and a "
        "bounded fine-tuning, evaluate by perplexity against a trivial floor and the frozen model on an independent split, "
        "and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish "
        "benchmark superiority, text quality or factual reliability on any domain, a usable acceptance threshold, or "
        "production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_BLOCKS = 1` and watch the gain "
        "shrink; set `EPOCHS = 4` and watch whether validation perplexity keeps falling or turns (the best epoch is kept "
        "either way); change `SEED` in Section 5 for a different sampled continuation; or bring your own documents through "
        "BYOD and read the unigram floor before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/gpt2-text-generation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/gpt2-text-generation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/gpt2-text-generation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/openai/gpt-2\n"
        "- Language Models are Unsupervised Multitask Learners (Radford et al., 2019; OpenAI technical report, no arXiv identifier): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf\n"
        "- TLDR: Extreme Summarization of Scientific Documents (Cachola et al., EMNLP Findings 2020; SciTLDR, Apache-2.0): https://arxiv.org/abs/2004.15011\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
