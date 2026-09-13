"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "gpt2_text_generation_pipeline",
    "repo_name": "gpt2-text-generation-pipeline",
    "stem": "gpt2_text_generation",
    "notebook_name": "gpt2_text_generation_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "GPT2TextGenerationPipeline",
    "weights_key": "gpt2",
    "runtime_imports": ["torch", "transformers"],
    "title": "GPT-2 124M — DIMER text-generation tutorial (standalone)",
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
    "capability": "causal text generation (continuing one English prompt) using the pinned GPT-2 124M weights, with greedy decoding by default and explicit, seeded nucleus sampling on request",
    "intro": (
        "At inference the byte-level BPE tokenizer turns the prompt into token ids (no special tokens are added), and "
        "the 12-layer decoder-only Transformer predicts one next-token distribution over the 50,257-token vocabulary at "
        "a time, feeding each chosen token back until `max_new_tokens` is reached or the end-of-text token is produced. "
        "**Two decoding modes are demonstrated and must not be confused (INF8):** greedy decoding (`do_sample=False`, "
        "the pipeline default) takes the argmax at every step and is deterministic on a fixed device and dtype — it is "
        "the mode for reproducibility checks and tends to repeat itself; nucleus sampling (`do_sample=True` with "
        "`temperature`, `top_p` and a mandatory `seed`) draws from the truncated distribution and is the mode usually "
        "preferred for actual use, reproducible only for the same seed on the same host. **No adaptation occurs:** no "
        "training, fine-tuning, in-context conditioning, or preprocessing fitting — the pinned checkpoint is used as "
        "published. GPT-2 is a **base language model**: no chat template, no instruction following, no safety tuning, "
        "English web text of 2019 vintage. What the upstream checkpoint supplies is the model and tokenizer; what the "
        "carried pipeline module adds is manifest verification, input validation and ceilings (prompts are rejected, "
        "never truncated), a settings validator that refuses unseeded sampling, the fixed pad/EOS handling, a fixed "
        "output contract and the `validate_inputs` and `evaluation_report` stage helpers. **No quality metric exists** "
        "for a free-text continuation without a reference corpus; the pipeline ships no metric helper."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author a synthetic prompt (or "
        "upload your own), stage and digest-verify the immutable upstream snapshot, surface the pipeline's ceilings and "
        "validate the prompt and both decoding settings into an input manifest before the model runs, generate a greedy "
        "continuation and confirm it is deterministic, generate a seeded sampled continuation with every setting echoed "
        "and confirm the seed reproduces it, read `finished_by` and the pad/EOS quirk correctly, read from the "
        "machine-readable evaluation report why no metric is reported and what corpus a perplexity number would need, "
        "and export machine-readable results plus provenance."
    ),
    "exclusions": (
        "chat or instruction following (GPT-2 has neither), batching (one prompt per call), raw logits or hidden states, "
        "fine-tuning, beam search, non-English text, or any content filtering. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (also float32). The model card's CPU smoke verified the 15-file snapshot in 0.32 s, loaded in 4.25 s, produced 32 greedy tokens from a 7-token prompt in 0.67 s and two seeded 16-token samples in 0.59 s together, so the default runs in seconds on a hosted CPU runtime. The pinned `torch==2.14.0` install and the 548 MB `model.safetensors` are the largest downloads of the run.",
        "- **Knowledge:** basic Python; what next-token prediction is; the difference between argmax decoding and sampling from a truncated distribution.",
        "- **Data:** the default sample is one synthetic English prompt authored in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 text file whose whole content (whitespace stripped) is the prompt. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API. A base language model can continue any prompt with false, biased or offensive text — read the output before reusing it.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic prompt or optional BYOD\n\n"
                "The default sample is **synthetic**: one short English prompt written in this cell — the same prompt the "
                "model card's CPU smoke used — so it needs no download and contains no personal data. It ships **no "
                "reference continuation**, so whatever the model produces is smoke/sanity evidence that the code path "
                "works, never a quality measurement and never benchmark evidence. The decoding settings are Colab form "
                "parameters: `GREEDY_MAX_NEW_TOKENS` for the deterministic default, and `SAMPLE_MAX_NEW_TOKENS`, "
                "`TEMPERATURE`, `TOP_P`, `SEED` for the sampling demonstration; they are validated against the carried "
                "module in Section 5 and echoed back by the pipeline in Sections 6 and 7.\n\n"
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 text file whose whole content "
                "(leading and trailing whitespace stripped) is the prompt — at most `MAX_TEXT_CHARS` characters and at "
                "most `MAX_PROMPT_TOKENS` BPE tokens, with prompt plus new tokens inside `CONTEXT_LENGTH` (over-long "
                "prompts are rejected by the pipeline, not truncated). The upload stays inside this runtime."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "GREEDY_MAX_NEW_TOKENS = 32  # @param {{type:\"integer\"}}\n"
                "SAMPLE_MAX_NEW_TOKENS = 16  # @param {{type:\"integer\"}}\n"
                "TEMPERATURE = 0.8  # @param {{type:\"number\"}}\n"
                "TOP_P = 0.9  # @param {{type:\"number\"}}\n"
                "SEED = 7  # @param {{type:\"integer\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    prompt = io.TextIOWrapper(io.BytesIO(uploaded[sample_name]), encoding='utf-8').read().strip()\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    prompt = 'The weather in the mountains is usually'\n"
                "    sample_name = 'synthetic_weather_prompt'\n"
                "    sample_kind = 'synthetic (authored in this cell; the model card smoke prompt)'\n"
                "prompt_sha256 = hashlib.sha256(prompt.encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'chars': len(prompt), 'prompt_sha256': prompt_sha256}})\n"
                "print(repr(prompt[:200]))"
            ),
        },
        {
            "md": (
                "## 5. Validate the prompt and both decoding settings → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it routes the prompt through the same "
                "private `_check_prompt` that `generate` uses and the decoding settings through the same public "
                "`validate_settings`, so a bad `temperature`, `top_p`, `max_new_tokens` or a missing `seed` fails "
                "**before** any model work, naming the condition, and a rejection here is a rejection there. It returns "
                "an **input manifest** naming the schema and ceilings, the prompt's character count, and the canonical "
                "settings the pipeline will echo back — for greedy decoding `temperature`, `top_p` and `seed` are "
                "recorded as `None` because they play no role. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`, and the sampling configuration is validated alongside it so both "
                "decoding modes are covered. `CONTEXT_LENGTH` (1024) is the model's positional window and bounds prompt "
                "tokens plus new tokens; `MAX_PROMPT_TOKENS` (1023) leaves room for at least one generated token; "
                "`MAX_NEW_TOKENS` (256) bounds one call's cost; `MAX_TEXT_CHARS` is the character guard applied before "
                "tokenisation; `VOCAB_SIZE` is the output vocabulary; `EOS_TOKEN_ID` and `PAD_TOKEN_ID` are equal by "
                "construction — **GPT-2 ships no pad token, so the pipeline fixes `pad_token_id = eos_token_id = 50256` "
                "in code** and passes an all-ones attention mask, which is why the single-prompt path raises no padding "
                "warning and why a generated `50256` means \"end of text\", after which the completion is cut. The token "
                "count of the prompt can only be checked after tokenisation, so those two ceilings are enforced inside "
                "`generate` in Section 6 (the pipeline rejects, never truncates). To show what rejection looks like, the "
                "cell also validates an unseeded sampling request and records the pipeline's own error message as a "
                "finding."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'CONTEXT_LENGTH': CONTEXT_LENGTH, 'MAX_PROMPT_TOKENS': MAX_PROMPT_TOKENS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'VOCAB_SIZE': VOCAB_SIZE, 'EOS_TOKEN_ID': EOS_TOKEN_ID, 'PAD_TOKEN_ID': PAD_TOKEN_ID}}\n"
                "print(ceilings)\n"
                "print({{'pad_eos_quirk': f'GPT-2 has no pad token; PAD_TOKEN_ID == EOS_TOKEN_ID == {{PAD_TOKEN_ID}}: {{PAD_TOKEN_ID == EOS_TOKEN_ID}}'}})\n"
                "input_manifest = validate_inputs(prompt, max_new_tokens=GREEDY_MAX_NEW_TOKENS, names=[sample_name])\n"
                "greedy_settings = input_manifest['settings']\n"
                "sampling_settings = validate_settings(SAMPLE_MAX_NEW_TOKENS, True, TEMPERATURE, TOP_P, SEED)\n"
                "input_manifest['sampling_settings'] = sampling_settings\n"
                "# Demonstrate the unseeded-sampling refusal; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(prompt, max_new_tokens=SAMPLE_MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'unseeded-sampling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "print({{'token_ceiling': f'prompt tokens <= MAX_PROMPT_TOKENS={{MAX_PROMPT_TOKENS}} and prompt + new tokens <= CONTEXT_LENGTH={{CONTEXT_LENGTH}} are checked by the pipeline after tokenisation; it rejects, never truncates'}})"
            ),
        },
        {
            "md": (
                "## 6. Generate with the greedy default and confirm it is deterministic\n\n"
                "`generate(prompt, max_new_tokens=...)` with the default `do_sample=False` takes the argmax at every "
                "step. It returns `completion` (the new text only), `text` (prompt plus completion), `prompt_tokens`, "
                "`new_tokens`, `finished_by` (`'eos'` when the model produced the end-of-text token 50256 — the "
                "completion is cut there — or `'max_new_tokens'` when the budget ran out), the echoed `settings` "
                "(`decoding: 'greedy'`, with `temperature`/`top_p`/`seed` as `None`), the device and the model identity. "
                "**Greedy decoding is deterministic on a fixed device and dtype:** the cell calls `generate` twice with "
                "the same settings and checks the two completions are byte-identical — a falsifiable check of the "
                "reproducibility contract, which does not extend across devices, PyTorch builds or dtypes. Greedy output "
                "is also the mode that repeats itself and drifts into generic text; it is the reference mode, not the "
                "recommended one for actual use. The model card's smoke observation on this prompt (32 greedy tokens, "
                "`finished_by = 'max_new_tokens'`, completion beginning `\" good, but the snow is not.\"`) is one "
                "measurement on that host, not an expected value — near-tied logits can flip a token between CPU and "
                "CUDA kernels and change everything after it."
            ),
            "code": (
                "import time\n\n"
                "started = time.perf_counter()\n"
                "greedy = pipe.generate(prompt, max_new_tokens=GREEDY_MAX_NEW_TOKENS)\n"
                "greedy_elapsed = time.perf_counter() - started\n"
                "greedy_repeat = pipe.generate(prompt, max_new_tokens=GREEDY_MAX_NEW_TOKENS)\n"
                "greedy_checks = {{\n"
                "    'settings_echoed_as_validated': greedy['settings'] == greedy_settings,\n"
                "    'decoding_is_greedy': greedy['settings']['decoding'] == 'greedy',\n"
                "    'new_tokens_within_budget': 0 <= greedy['new_tokens'] <= GREEDY_MAX_NEW_TOKENS,\n"
                "    'prompt_tokens_within_ceiling': 1 <= greedy['prompt_tokens'] <= MAX_PROMPT_TOKENS,\n"
                "    'text_is_prompt_plus_completion': greedy['text'] == greedy['prompt'] + greedy['completion'],\n"
                "    'finished_by_is_known': greedy['finished_by'] in ('eos', 'max_new_tokens'),\n"
                "    'greedy_repeat_is_identical': greedy_repeat['completion'] == greedy['completion'],\n"
                "}}\n"
                "if not all(greedy_checks.values()):\n"
                "    raise RuntimeError(f'greedy generate output failed a sanity check: {{greedy_checks}}')\n"
                "print({{key: value for key, value in greedy.items() if key not in ('prompt', 'completion', 'text')}})\n"
                "print({{'seconds_first_call': round(greedy_elapsed, 3), 'checks': greedy_checks}})\n"
                "print(f'prompt:     {{prompt!r}}')\n"
                "print(f\"completion: {{greedy['completion']!r}}\")"
            ),
        },
        {
            "md": (
                "## 7. Generate with explicit, seeded nucleus sampling\n\n"
                "Sampling is opt-in: `do_sample=True` with `temperature` (rescales the logits; below 1 sharpens, above 1 "
                "flattens), `top_p` (nucleus truncation: only the smallest set of tokens whose cumulative probability "
                "reaches `top_p` is sampled from) and a **mandatory `seed`** — `validate_settings` refuses unseeded "
                "sampling so a sampled result is always reproducible for the same seed on the same host. The pipeline "
                "seeds PyTorch's generator immediately before the model call. Every setting is echoed in `settings` "
                "(`decoding: 'nucleus-sampling'`) so an exported result records exactly how it was produced (INF9). This "
                "cell samples twice with the same seed and checks the completions are identical, then contrasts the "
                "sampled completion with the greedy one from Section 6: a different completion is expected (the model "
                "card observed `\" good, but you need to be careful with your gear and don't go to\"` for seed 7 on its "
                "host — an observation, not an expected value, because the sampled path depends on the exact "
                "floating-point logits of that host). Sampling is the mode usually preferred for actual use because it "
                "avoids greedy repetition; it is not \"better\" in any measured sense here, and a different seed gives a "
                "different continuation."
            ),
            "code": (
                "started = time.perf_counter()\n"
                "sampled = pipe.generate(prompt, max_new_tokens=SAMPLE_MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, seed=SEED)\n"
                "sampled_elapsed = time.perf_counter() - started\n"
                "sampled_repeat = pipe.generate(prompt, max_new_tokens=SAMPLE_MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, seed=SEED)\n"
                "sampled_checks = {{\n"
                "    'settings_echoed_as_validated': sampled['settings'] == sampling_settings,\n"
                "    'decoding_is_nucleus_sampling': sampled['settings']['decoding'] == 'nucleus-sampling',\n"
                "    'seed_echoed': sampled['settings']['seed'] == SEED,\n"
                "    'new_tokens_within_budget': 0 <= sampled['new_tokens'] <= SAMPLE_MAX_NEW_TOKENS,\n"
                "    'text_is_prompt_plus_completion': sampled['text'] == sampled['prompt'] + sampled['completion'],\n"
                "    'same_seed_reproduces': sampled_repeat['completion'] == sampled['completion'],\n"
                "}}\n"
                "if not all(sampled_checks.values()):\n"
                "    raise RuntimeError(f'sampled generate output failed a sanity check: {{sampled_checks}}')\n"
                "print({{key: value for key, value in sampled.items() if key not in ('prompt', 'completion', 'text')}})\n"
                "print({{'seconds_first_call': round(sampled_elapsed, 3), 'checks': sampled_checks}})\n"
                "print(f\"greedy:  {{greedy['completion'][:120]!r}}\")\n"
                "print(f\"sampled: {{sampled['completion'][:120]!r}}\")\n"
                "print({{'sampled_differs_from_greedy': sampled['completion'] != greedy['completion'][: len(sampled['completion'])], 'note': 'expected but not asserted; both are unscored continuations'}})"
            ),
        },
        {
            "md": (
                "## 8. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report — even, as "
                "here, when nothing is measurable. The repository ships **no metric helper and reports no performance "
                "measure**: a continuation has no ground truth, so the verdict is always `not-measurable` and the report "
                "states what would make the task measurable — a held-out reference corpus from the deployment domain "
                "scored for perplexity with the caller's own code for an intrinsic number, or human raters or a labelled "
                "downstream task for any quality or factuality claim. Supplying a reference string does not change the "
                "verdict, because a reference string is not a corpus and no metric helper exists to score it; the helper "
                "records that in `reason` rather than inventing a number. The report is written for the greedy result "
                "and carries the decoding mode in `score_semantics`, so a reader can see which configuration produced "
                "the unscored text. It lands at `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(greedy, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: a continuation has no ground truth, the sample has no reference corpus, and the repository ships no metric helper; perplexity needs a held-out corpus you supply.')"
            ),
        },
        {
            "md": (
                "## 9. Export outputs and provenance\n\n"
                "Two further files are written under `outputs/` beside the input manifest and the evaluation report. The "
                "two completions go to CSV (`outputs/{stem}_completions.csv`) with explicit `mode`, `decoding`, "
                "`prompt_tokens`, `new_tokens`, `finished_by`, `seed` and `completion` columns, so each completion stays "
                "attached to the configuration that produced it. One JSON record (`outputs/{stem}_result.json`) "
                "preserves the prompt, the greedy result and the sampled result (each with completion, token counts, "
                "`finished_by`, and the echoed settings), the determinism and sanity checks, the ceilings in force "
                "including the pad/EOS ids, the input manifest, the evaluation report, the sample identity and digest, "
                "the notebook's source (repository, revision, embedded module digest, generator), the model identifier, "
                "the immutable model revision, the model licence, the verified snapshot summary, and the runtime "
                "identity (Python, `torch`, `transformers`, device, dtype). No credentials are involved in any step, so "
                "none can reach the export."
            ),
            "code": (
                "import csv\n\n"
                "with open('outputs/{stem}_completions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['mode', 'decoding', 'prompt_tokens', 'new_tokens', 'finished_by', 'seed', 'completion'])\n"
                "    for mode, item in (('greedy', greedy), ('sampled', sampled)):\n"
                "        writer.writerow([mode, item['settings']['decoding'], item['prompt_tokens'], item['new_tokens'], item['finished_by'], item['settings']['seed'], item['completion']])\n"
                "payload = {{\n"
                "    'prompt': prompt,\n"
                "    'greedy': {{key: greedy[key] for key in ('completion', 'text', 'prompt_tokens', 'new_tokens', 'finished_by', 'settings')}},\n"
                "    'sampled': {{key: sampled[key] for key in ('completion', 'text', 'prompt_tokens', 'new_tokens', 'finished_by', 'settings')}},\n"
                "    'sanity_checks': {{'greedy': greedy_checks, 'sampled': sampled_checks}},\n"
                "    'seconds': {{'greedy_first_call': round(greedy_elapsed, 3), 'sampled_first_call': round(sampled_elapsed, 3)}},\n"
                "    'ceilings': ceilings,\n"
                "    'completions_file': 'outputs/{stem}_completions.csv',\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'prompt_sha256': prompt_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': snapshot['path'], 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes')}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Both completions are unscored continuations from a 2019 base language model: they can be false, repetitive, "
        "biased or offensive, they carry no confidence or probability, and nothing in the pipeline filters them. Greedy "
        "decoding is the deterministic reference mode (identical on repeat, on the same device and dtype); seeded nucleus "
        "sampling is the mode usually preferred for use (identical on repeat for the same seed on the same host, "
        "different for a different seed or host). Neither is \"better\" in any measured sense here: the evaluation report "
        "is `not-measurable` because none can be computed without a reference corpus (perplexity) or human judgements, "
        "and the model card's smoke completions are observations from one host, not expected values. Prompts are "
        "rejected above `MAX_PROMPT_TOKENS` or when prompt plus new tokens would exceed the 1024-token window, never "
        "truncated; `finished_by = 'eos'` means the model emitted token 50256, which doubles as the pad id because GPT-2 "
        "ships no pad token; the pipeline exposes one prompt per call, no chat format, no batching, no logits and no "
        "fine-tuning.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot, validate the demonstrated prompt and both decoding "
        "settings against the enforced ceilings, execute the public pipeline path in both decoding modes with the stated "
        "determinism properties, and emit the shown machine-readable outputs in the tested runtime — without the "
        "repository being reachable. It does **not** establish benchmark superiority, text quality on any domain, "
        "factual reliability, safety for high-consequence use, or production fitness on an unseen domain.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: "
        "the pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the "
        "top. `FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/{MODEL_KEY}/` and rerun Section 3. A `ValueError`/`TypeError` "
        "from `validate_inputs` or `validate_settings` in Section 5: a form parameter is out of range "
        "(`max_new_tokens` 1..256, `temperature` > 0, `top_p` in (0, 1], integer `seed` >= 0) — fix it and rerun from "
        "Section 4. A `ValueError` naming `MAX_PROMPT_TOKENS` or `CONTEXT_LENGTH` in Section 6: the BYOD prompt "
        "tokenises too long for the requested budget — shorten it or lower `GREEDY_MAX_NEW_TOKENS`. A "
        "`greedy_repeat_is_identical` failure would indicate non-deterministic kernels on the host and should be "
        "reported with the runtime identity.\n\n"
        "**Next experiments.** Change `SEED` and rerun Section 7 to see a different sampled continuation; set "
        "`TEMPERATURE` to 0.3 and 1.5 and compare how conservative or erratic the samples become; raise "
        "`GREEDY_MAX_NEW_TOKENS` to 128 and watch greedy decoding repeat itself; upload a paragraph via `USE_BYOD` and "
        "inspect `prompt_tokens`; compute perplexity on a small held-out text of your own with your own code as the "
        "first step towards the intrinsic number the evaluation report asks for. None of these turns the sample result "
        "into evidence of production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/gpt2-text-generation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/gpt2-text-generation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/gpt2-text-generation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/openai/gpt-2\n"
        "- Language Models are Unsupervised Multitask Learners (Radford et al., 2019; OpenAI technical report, no arXiv identifier): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
    ),
}
