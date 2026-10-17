"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.5 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E domain-adaptation workflow: the pinned GPT-2 124M snapshot is
digest-verified and loaded, a digest-pinned real corpus (SciTLDR paper abstracts) is fetched, validated and
split, the inference contract is exercised in both decoding modes, the pretrained model's held-out perplexity is
read beside a unigram floor, a bounded fine-tuning of the last transformer blocks adapts the model to the
domain in the kernel, the held-out split is scored again, and the adapter is exported and reloaded.

Review fixes (Notebook Review Framework v1, review PR #8, GPT-M1..M3 / GPT-m1..m5): the runtime is the fleet's uv
isolated environment (no in-kernel install, no restart); every section that measures or trains the pretrained
model first calls `pipe.reset_to_pretrained()`, `adapt` refuses an already adapted pipeline, and the export is
checked to describe the model in memory, so every documented rerun compares against the pretrained model and
reload parity holds after `TRAINABLE_BLOCKS` changes; BYOD has a `BYOD_PATH` field, the stated minimum is the one
the split applies (12 records) and an empty upload gets an actionable message; the perplexity comparisons are
recorded outcomes instead of bare assertions; the possible overlap of the abstracts with GPT-2's pretraining text
is stated; every time names the run that measured it; and the guided layer (who it is for, how to use, roadmap,
predictions, worked answers, a change-one-thing activity, troubleshooting, glossary, conclusion) is added.

Review fixes (row 31, 2026-10-17, T31-1..T31-8): the adaptation defaults are re-selected on validation by a
pre-registered three-configuration grid (docs/reselection-2026-10-17.md) and the selected configuration is scored
once on a pinned test draw that no selection step reads, with the old test abstracts reported once as the exposed development
test; BYOD refusals are messages, not tracebacks; the notebook declares spec 2.5, its kind and the use notice;
run-status wording points at docs/release-verification.md.

Code cells are written with single braces and escaped by ``_py`` for the generator's ``str.format`` pass; ``@STEM@``
becomes the output stem. Stage markdown and the closing are formatted too, so they contain no literal braces; the
header strings (``run_all``, ``byod``, ``intro``, ``prerequisites``) are inserted as written.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "gpt2-text-generation-pipeline"


def _py(code: str) -> str:
    """Escape a code cell for the generator's ``str.format`` pass; ``@STEM@`` stands for ``{stem}``."""
    return code.replace("{", "{{").replace("}", "}}").replace("@STEM@", "{stem}")


_DATA_CODE = _py(
    """import hashlib
import io
import json
import os
from pathlib import Path

USE_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ''  # @param {type:"string"}
SPLIT_SEED = 42  # @param {type:"integer"}

os.makedirs('outputs', exist_ok=True)
# A new dataset needs a model that was trained on nothing else: drop any adaptation an earlier run made.
pipe.reset_to_pretrained()
# A new dataset also ends the active run: its splits and results are dropped, so a refusal leaves nothing stale.
for stale in ('splits', 'train_records', 'val_records', 'test_records', 'exposed_test_records', 'dataset_manifests', 'disjoint', 'reselection', 'selected_config', 'adapt_result', 'frozen_test', 'frozen_val', 'frozen_exposed', 'adapted_test', 'adapted_val', 'frozen_reference', 'preregistered', 'differs'):
    globals().pop(stale, None)
if USE_BYOD:
    # Expected refusals (no file, an empty upload, a file that breaks the contract) print one message and stop;
    # files that an earlier run wrote to outputs/ are kept. Anything unexpected keeps its traceback.
    try:
        if BYOD_PATH:
            byod_path = Path(BYOD_PATH)
            file_name = byod_path.name
        else:
            try:
                from google.colab import files
            except ImportError:
                raise DatasetRefused('USE_BYOD = True needs a file: this runtime has no upload dialog, so set BYOD_PATH to your .csv, .json, .jsonl or .txt file.') from None
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise DatasetRefused(f'Upload exactly one .csv, .json, .jsonl or .txt file (got {len(uploaded)}; an empty or cancelled upload gives 0). Run this cell again, or set BYOD_PATH.')
            file_name, payload = next(iter(uploaded.items()))
            byod_path = Path('work') / file_name
            byod_path.parent.mkdir(parents=True, exist_ok=True)
            byod_path.write_bytes(payload)
        records = load_byod_dataset(byod_path)
        byod_splits = split_dataset(records, seed=SPLIT_SEED)
        validate_splits(byod_splits)
        check_split_disjoint(byod_splits)
    except (OSError, TypeError, ValueError) as exc:
        raise DatasetRefused(f'BYOD refused: {str(exc).rstrip(".")}. Nothing from this file is used and the run is reset; files an earlier run wrote to outputs/ are kept. Fix the file (or BYOD_PATH) and run this cell again.') from None
    splits = byod_splits
    data_source = 'BYOD (' + file_name + ')'
    raw_papers = {'byod': len(records)}
else:
    corpus = read_corpus(fetch_corpus(cache_dir='weights/scitldr'))
    raw_papers = {name: len(part) for name, part in corpus.items()}
    # train and validation from the seeded draw, test = the pinned draw no selection step reads, exposed_test = the old test draw
    splits = build_tutorial_dataset(corpus, seed=SPLIT_SEED)
    data_source = f'{CORPUS_NAME} ({CORPUS_RELEASE}; {CORPUS_LICENSE})'
train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']
exposed_test_records = splits.get('exposed_test', [])
# Each split with the minimum the next stage applies (training 8, validation and test 1); a refusal names the split.
dataset_manifests = validate_splits(splits)
disjoint = check_split_disjoint(splits)
papers = check_paper_disjoint(splits)
write_dataset_csv(train_records, 'outputs/@STEM@_train.csv')
print({'data_source': data_source, 'raw_papers': raw_papers, 'splits': disjoint, 'no_text_or_paper_in_two_splits': True, 'file_sha256': {k: v[2][:12] + '...' for k, v in CORPUS_FILES.items()}})
for name, manifest in dataset_manifests.items():
    print({name: {'n': manifest['n_records'], 'unique_texts': manifest['unique_texts'], 'text_chars': manifest['text_chars'], 'total_chars': manifest['total_chars'], 'digest': manifest['digest'][:16] + '...'}})
if not USE_BYOD:
    print({'test': 'pinned draw that no selection or tuning step reads', 'digest_is_pinned': dataset_manifests['test']['digest'] == FRESH_TEST_DIGEST, 'exposed_test': 'the 100 abstracts the 2026-09 sweep scored; reported once in Section 8, chooses nothing'})
print({'example': {'id': train_records[0]['id'], 'text': train_records[0]['text'][:200] + '...'}})

probes = {
    'duplicate id': [{**r, 'id': 'same'} for r in train_records[:8]],
    'empty text': [{**train_records[0], 'text': '   '}, *train_records[1:8]],
    'missing field': [{'id': r['id']} for r in train_records[:8]],
    'too small': train_records[:3],
}
for name, probe in probes.items():
    try:
        validate_dataset(probe)
        print({'probe': name, 'verdict': 'accepted'})
    except (TypeError, ValueError) as exc:
        print({'probe': name, 'rejected': str(exc)[:110]})"""
)

_GENERATE_CODE = _py(
    """import math
import time

GREEDY_MAX_NEW_TOKENS = 32  # @param {type:"integer"}
SAMPLE_MAX_NEW_TOKENS = 32  # @param {type:"integer"}
TEMPERATURE = 0.8  # @param {type:"number"}
TOP_P = 0.9  # @param {type:"number"}
SEED = 7  # @param {type:"integer"}

# The "before" continuations must come from the pretrained model, also when this cell is re-run after Section 7.
pipe.reset_to_pretrained()

def opening(record, chars=120):
    head = record['text'][:chars]
    return head.rsplit(' ', 1)[0] if ' ' in head else head

prompt = opening(val_records[0])
ceilings = {'CONTEXT_LENGTH': CONTEXT_LENGTH, 'MAX_PROMPT_TOKENS': MAX_PROMPT_TOKENS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'VOCAB_SIZE': VOCAB_SIZE, 'EOS_TOKEN_ID': EOS_TOKEN_ID, 'PAD_TOKEN_ID': PAD_TOKEN_ID, 'MAX_TRAIN_TOKENS': MAX_TRAIN_TOKENS}
print(ceilings)
input_manifest = validate_inputs(prompt, max_new_tokens=GREEDY_MAX_NEW_TOKENS, names=['validation-opening'])
sampling_settings = validate_settings(SAMPLE_MAX_NEW_TOKENS, True, TEMPERATURE, TOP_P, SEED)
input_manifest['sampling_settings'] = sampling_settings
try:
    validate_inputs(prompt, max_new_tokens=SAMPLE_MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
except ValueError as exc:
    input_manifest['findings'].append({'input': 'unseeded-sampling-probe', 'verdict': 'rejected', 'message': str(exc)})
with open('outputs/@STEM@_input_manifest.json', 'w', encoding='utf-8') as handle:
    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)
started = time.perf_counter()
greedy = pipe.generate(prompt, max_new_tokens=GREEDY_MAX_NEW_TOKENS)
greedy_seconds = round(time.perf_counter() - started, 3)
greedy_repeat = pipe.generate(prompt, max_new_tokens=GREEDY_MAX_NEW_TOKENS)
sampled = pipe.generate(prompt, max_new_tokens=SAMPLE_MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, seed=SEED)
sampled_repeat = pipe.generate(prompt, max_new_tokens=SAMPLE_MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, seed=SEED)
checks = {
    'greedy_settings_echoed': greedy['settings'] == input_manifest['settings'] and greedy['settings']['decoding'] == 'greedy',
    'sampled_settings_echoed': sampled['settings'] == sampling_settings and sampled['settings']['seed'] == SEED,
    'new_tokens_within_budget': greedy['new_tokens'] <= GREEDY_MAX_NEW_TOKENS and sampled['new_tokens'] <= SAMPLE_MAX_NEW_TOKENS,
    'prompt_tokens_within_ceiling': 1 <= greedy['prompt_tokens'] <= MAX_PROMPT_TOKENS,
    'text_is_prompt_plus_completion': greedy['text'] == greedy['prompt'] + greedy['completion'],
    'finished_by_is_known': greedy['finished_by'] in ('eos', 'max_new_tokens'),
    'greedy_repeat_is_identical': greedy_repeat['completion'] == greedy['completion'],
    'same_seed_reproduces': sampled_repeat['completion'] == sampled['completion'],
}
if not all(checks.values()):
    raise RuntimeError(f'generate output failed a sanity check: {checks}')
print({'prompt_tokens': greedy['prompt_tokens'], 'greedy': {'new_tokens': greedy['new_tokens'], 'finished_by': greedy['finished_by'], 'seconds': greedy_seconds}, 'sampled': {'new_tokens': sampled['new_tokens'], 'finished_by': sampled['finished_by'], 'decoding': sampled['settings']['decoding']}, 'checks': checks, 'findings': len(input_manifest['findings']), 'no_score': 'the pipeline emits no probability or quality score'})
print(f'prompt:  {prompt!r}')
print(f"greedy:  {greedy['completion']!r}")
print(f"sampled: {sampled['completion']!r}")
if USE_BYOD:
    # A small BYOD test split may hold only two records; the first one is the prompt above.
    unseen_records = [{**r, 'id': f'unseen-{i:02d}'} for i, r in enumerate(test_records[1:4] or test_records[:1])]
else:
    used = {r['text'].lower() for part in splits.values() for r in part}
    unseen_records = [{**r, 'id': f'unseen-{i:02d}'} for i, r in enumerate([r for r in filter_records(corpus['dev']) if r['text'].lower() not in used][:3])]
before = {r['id']: pipe.generate(opening(r), max_new_tokens=GREEDY_MAX_NEW_TOKENS)['completion'] for r in unseen_records}
print({'unseen_openings_continued_by_the_pretrained_model': len(before)})"""
)

_FLOOR_CODE = _py(
    """# Section 6 measures the pretrained model: undo any adaptation a re-run left behind, and refuse otherwise.
pipe.reset_to_pretrained()
if pipe.adapter is not None:
    raise RuntimeError('the pipeline still holds an adaptation, so its numbers are not the pretrained model; run this cell again')
# Score-once rule: the pinned test split is read only under the pre-registered data settings (the sample with
# SPLIT_SEED = 42). Your own data or another draw is exploratory, and every number here is then on validation.
data_preregistered = not USE_BYOD and SPLIT_SEED == 42
reference_records = test_records if data_preregistered else val_records
reference_split = 'test' if data_preregistered else 'validation (exploratory data settings: the test split is not read)'
if not data_preregistered:
    print('Exploratory data settings (your own data, or SPLIT_SEED other than 42): Sections 6 and 8 read the validation split; the test split is not scored.')
t0 = time.perf_counter()
unigram = pipe.unigram_baseline(train_records, reference_records)
unigram_light = pipe.unigram_baseline(train_records, reference_records, alpha=0.1)
print({'unigram_floor': {'split': reference_split, 'perplexity': round(unigram['perplexity'], 1), 'bits_per_token': round(unigram['bits_per_token'], 3), 'n_tokens': unigram['n_tokens'], 'baseline': unigram['baseline']}, 'unigram_add_0.1_reference': round(unigram_light['perplexity'], 1), 'seconds': round(time.perf_counter() - t0, 1)})
t0 = time.perf_counter()
frozen_test = pipe.evaluate(test_records) if data_preregistered else None
frozen_val = pipe.evaluate(val_records)
frozen_reference = frozen_test if data_preregistered else frozen_val
# The exposed development test chooses nothing; it is scored here on the pretrained model and printed in Section 8.
frozen_exposed = pipe.evaluate(exposed_test_records) if exposed_test_records else None
print({'pretrained_model': {'split': reference_split, 'perplexity': round(frozen_reference['perplexity'], 2), 'bits_per_token': round(frozen_reference['bits_per_token'], 3), 'n_records': frozen_reference['n_records'], 'n_tokens': frozen_reference['n_tokens'], 'record_perplexity': {k: round(v, 1) for k, v in frozen_reference['record_perplexity'].items()}, 'verdict': frozen_reference['verdict'], 'adapted': frozen_reference['adapted']}, 'validation_perplexity': round(frozen_val['perplexity'], 2), 'seconds': round(time.perf_counter() - t0, 1)})
print({'definitions': frozen_reference['definitions']})
if frozen_reference['n_tokens'] != unigram['n_tokens']:
    raise RuntimeError('the floor and the model were scored on different tokens, so they cannot be compared; run Section 4 again')
# A recorded outcome, not an assertion: on your own corpus the pretrained model may not beat the floor.
floor_outcome = 'pretrained model below the unigram floor' if frozen_reference['perplexity'] < unigram['perplexity'] else 'pretrained model NOT below the unigram floor: it does not use context on this corpus'
print({'floor_outcome': floor_outcome})"""
)

_GRID_CODE = _py(
    """RESELECT_ON_VALIDATION = True  # @param {type:"boolean"}
# The selection the CPU build record made on the sample (docs/reselection-2026-10-17.md); the grid replays it.
RECORDED_SELECTION = 'C0'

recorded = next(g for g in RESELECTION_GRID if g['name'] == RECORDED_SELECTION)
data_key = [[name, manifest['digest']] for name, manifest in sorted(dataset_manifests.items())]
if not RESELECT_ON_VALIDATION:
    reselection = None
    print({'reselection': 'skipped (RESELECT_ON_VALIDATION = False)', 'recorded_selection': recorded})
elif globals().get('reselection') and reselection['data_key'] == data_key:
    print({'reselection': 'already run on these splits in this session; not repeated'})
else:
    def grid_report(name, entry):
        if entry.get('val'):
            print({'config': name, 'epoch': entry['epoch'], 'val_perplexity': round(entry['val']['perplexity'], 2)})

    t0 = time.perf_counter()
    # Validation only: every configuration starts from the pretrained weights, and the pipeline is returned to them.
    reselection = reselect_on_validation(pipe, train_records, val_records, epochs=2, batch_size=8, seed=0, progress=grid_report)
    reselection.update(data_key=data_key, seconds=round(time.perf_counter() - t0, 1))
    with open('outputs/@STEM@_reselection.json', 'w', encoding='utf-8') as handle:
        json.dump(reselection, handle, indent=2, ensure_ascii=False)
if reselection is not None:
    for row in reselection['grid']:
        print({'config': row['name'], 'trainable_blocks': row['trainable_blocks'], 'learning_rate': row['lr'], 'best_epoch': row['best_epoch'], 'best_val_perplexity': round(row['best_val_perplexity'], 2), 'seconds': row['seconds']})
    print({'selected': reselection['selection']['selected'], 'reason': reselection['selection']['reason'], 'improvement_vs_C0': reselection['selection']['improvement_vs_incumbent']})
# The selected configuration: on the default sample draw (SPLIT_SEED = 42) the pre-registered build record's choice;
# on another draw or on your data, the replay's (the 2 % margin was justified on SciTLDR only).
replayed = reselection is not None and (USE_BYOD or SPLIT_SEED != 42)
selected_config = reselection['selection']['config'] if replayed else recorded
if reselection is not None:
    for name, pair in reselection['paired_vs_incumbent_validation'].items():
        print({'paired_validation_vs_C0': name, 'token_weighted_delta_bits': round(pair['token_weighted_delta_bits'], 3), 'bootstrap_95ci': [round(x, 3) for x in pair['bootstrap_95ci']], 'records_worse_than_C0': pair['n'] - pair['n_improved'], 'of': pair['n']})
if not replayed and reselection is not None:
    print({'reproduces_recorded_selection': reselection['selection']['selected'] == RECORDED_SELECTION, 'recorded_selection': RECORDED_SELECTION})
print({'selected_configuration': selected_config['name'], 'trainable_blocks': selected_config['trainable_blocks'], 'learning_rate': selected_config['lr']})"""
)

_ADAPT_CODE = _py(
    """EPOCHS = 2  # @param {type:"integer"}
LEARNING_RATE = 1e-4  # @param {type:"number"}
BATCH_SIZE = 8  # @param {type:"integer"}
TRAINABLE_BLOCKS = 4  # @param {type:"integer"}
ADAPT_SEED = 0  # @param {type:"integer"}

# The form defaults are the pre-registered selection (C0). If the replay selected another configuration (your data,
# or a SPLIT_SEED other than 42) and these two fields are untouched, the selected configuration is trained.
if (TRAINABLE_BLOCKS, LEARNING_RATE) == (4, 1e-4) and (selected_config['trainable_blocks'], selected_config['lr']) != (4, 1e-4):
    TRAINABLE_BLOCKS, LEARNING_RATE = selected_config['trainable_blocks'], selected_config['lr']
    print({'training_the_selected_configuration': selected_config['name'], 'TRAINABLE_BLOCKS': TRAINABLE_BLOCKS, 'LEARNING_RATE': LEARNING_RATE})

def report(entry):
    row = {'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}
    if entry.get('val'):
        row['val_perplexity'] = round(entry['val']['perplexity'], 2)
        row['val_bits_per_token'] = round(entry['val']['bits_per_token'], 3)
    if 'note' in entry:
        row['note'] = entry['note']
    print(row)

# Every run starts from the pretrained weights: a re-run with other settings never builds on an earlier one,
# and the exported adapter then holds every block that differs from the base.
pipe.reset_to_pretrained()
t0 = time.perf_counter()
adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_blocks=TRAINABLE_BLOCKS, seed=ADAPT_SEED, progress=report)
adapt_seconds = round(time.perf_counter() - t0, 1)
epoch0 = adapt_result['history'][0]['val']
start_check = None if epoch0 is None else abs(epoch0['perplexity'] - frozen_val['perplexity']) / frozen_val['perplexity']
if start_check is not None and start_check > 1e-4:
    raise RuntimeError(f'epoch 0 ({epoch0["perplexity"]:.3f}) is not the pretrained model of Section 6 ({frozen_val["perplexity"]:.3f}); run Section 6, then this cell again')
print({'starts_from': 'pretrained weights', 'epoch0_vs_section6_relative_difference': start_check, 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'train_tokens': adapt_result['n_train_tokens'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seed': adapt_result['seed'], 'seconds': adapt_seconds})
# Score-once rule: the test split is scored only if EVERY pre-registered setting holds (docs/reselection-2026-10-17.md).
settings = {'data_source': 'BYOD' if USE_BYOD else 'sample', 'split_seed': SPLIT_SEED, 'dataset_digests': {k: dataset_manifests[k]['digest'] for k in ('train', 'validation', 'test')}, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'trainable_blocks': TRAINABLE_BLOCKS, 'adapt_seed': ADAPT_SEED, 'max_train_tokens': MAX_TRAIN_TOKENS, 'configuration': selected_config['name']}
differs = preregistration_diff(settings)
preregistered = not differs
grid_row = next((r for r in reselection['grid'] if r['name'] == selected_config['name']), None) if reselection is not None and preregistered else None
curve = [None if h['val'] is None else h['val']['perplexity'] for h in adapt_result['history']]
print({'preregistered': preregistered, 'differs_from_preregistration': differs, 'repeats_the_grid_run': None if grid_row is None else curve == grid_row['val_perplexity_by_epoch']})
if not preregistered:
    print(f'Exploratory run (differs in {differs}): Section 8 compares on validation only, the test split is not scored, and its outputs go to separate _exploratory_ files.')"""
)

_EVAL_CODE = _py(
    """adapted_val = pipe.evaluate(val_records)
# Score-once rule: the test split is read only under the pre-registered settings (Section 7 printed which differ).
earlier = [r for r in globals().get('run_history', [])]
look = test_look(differs, sum(r.get('test_look') in ('pre-registered evaluation', 'reproduction') for r in earlier))
run_tag = '' if preregistered else f"_exploratory_{sum(r.get('test_look') == 'not scored (exploratory)' for r in earlier) + 1}"
adapted_test = pipe.evaluate(test_records) if preregistered else None
held_out = 'test' if adapted_test is not None else 'validation (exploratory: the test split is not scored)'
floor_matches = adapted_test is not None or not data_preregistered
print({'test_look': look, 'differs_from_preregistration': differs})
before_m, after_m = (frozen_test, adapted_test) if adapted_test is not None else (frozen_val, adapted_val)
comparison = {
    metric: {'unigram_floor': round(unigram[metric], 3) if floor_matches else None, 'pretrained': round(before_m[metric], 3), 'adapted': round(after_m[metric], 3)}
    for metric in ('perplexity', 'bits_per_token', 'mean_nll')
}
comparison['split'] = held_out
comparison['record_perplexity'] = {'pretrained': {k: round(v, 1) for k, v in before_m['record_perplexity'].items()}, 'adapted': {k: round(v, 1) for k, v in after_m['record_perplexity'].items()}}
comparison['delta_vs_pretrained'] = {metric: round(after_m[metric] - before_m[metric], 3) for metric in ('perplexity', 'bits_per_token')}
for metric, row in comparison.items():
    print({metric: row})
# Paired: each abstract compared with itself, so the per-record change is read directly (seed 0 bootstrap).
paired = paired_delta(before_m['records'], after_m['records'])
comparison['paired_per_record'] = paired
print({'paired_per_record': {'split': held_out, 'n_improved': paired['n_improved'], 'of': paired['n'], 'delta_bits_per_token_median': round(paired['delta_bits_per_token']['median'], 3), 'quartiles': [round(paired['delta_bits_per_token']['q1'], 3), round(paired['delta_bits_per_token']['q3'], 3)], 'token_weighted_delta_bits': round(paired['token_weighted_delta_bits'], 3), 'bootstrap_95ci': [round(x, 3) for x in paired['bootstrap_95ci']]}})
if adapted_test is not None and not USE_BYOD:
    # A heuristic contamination probe (Section 6): 9-character OpenReview ids are ICLR-2018-era papers.
    for label, keep in (('id_len_9', True), ('id_len_other', False)):
        ids = [r['id'] for r in test_records if (len(r['paper_id']) == 9) == keep]
        tokens = sum(frozen_test['records'][i]['n_tokens'] for i in ids)
        ppl = [math.exp(sum(scored['records'][i]['mean_nll'] * scored['records'][i]['n_tokens'] for i in ids) / tokens) for scored in (frozen_test, adapted_test)]
        print({'era_probe': label, 'n': len(ids), 'pretrained': round(ppl[0], 2), 'adapted': round(ppl[1], 2)})
exposed_report = None
if adapted_test is not None and frozen_exposed is not None:
    adapted_exposed = pipe.evaluate(exposed_test_records)
    exposed_paired = paired_delta(frozen_exposed['records'], adapted_exposed['records'])
    exposed_report = {'label': 'exposed development test: the 100 abstracts the 2026-09 sweep scored when the earlier defaults were chosen; reported once for continuity, not independent evidence', 'n_tokens': adapted_exposed['n_tokens'], 'pretrained': round(frozen_exposed['perplexity'], 2), 'adapted': round(adapted_exposed['perplexity'], 2), 'delta': round(adapted_exposed['perplexity'] - frozen_exposed['perplexity'], 2), 'n_improved': exposed_paired['n_improved'], 'bootstrap_95ci_bits': [round(x, 3) for x in exposed_paired['bootstrap_95ci']]}
    print({'exposed_development_test': exposed_report})
# A recorded outcome, not an assertion: an adaptation that does not help is a result to read, not a crash.
adaptation_outcome = 'lowered held-out perplexity' if after_m['perplexity'] < before_m['perplexity'] else 'did NOT lower held-out perplexity'
print({'adaptation_outcome': adaptation_outcome, 'read_on': held_out, 'floor_outcome': floor_outcome})
evaluation_report_payload = {
    'model': {'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY},
    'data_source': data_source,
    'dataset_digests': {name: manifest['digest'] for name, manifest in dataset_manifests.items()},
    'splits': disjoint,
    'test_look': look,
    'differs_from_preregistration': differs,
    'evidence': 'one seeded split of one corpus; the configuration was re-selected on validation by a pre-registered rule (docs/reselection-2026-10-17.md) and the pinned test split, which no selection step reads, is scored only under the pre-registered settings (the recorded evaluation and its reproductions are listed in docs/release-verification.md); paired bootstrap interval over records, no seed-to-seed estimate',
    'selected_configuration': selected_config,
    'reselection': None if reselection is None else {k: v for k, v in reselection.items() if k != 'data_key'},
    'baselines': {'unigram_floor': unigram},
    'pretrained_test': frozen_test,
    'pretrained_validation': frozen_val,
    'validation_metrics': adapted_val,
    'test_metrics': adapted_test,
    'exposed_development_test': exposed_report,
    'comparison': comparison,
    'adaptation_outcome': adaptation_outcome,
    'floor_outcome': floor_outcome,
    'adaptation': {k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')},
    'history': adapt_result['history'],
    'adaptation_seconds': adapt_seconds,
}
with open(f'outputs/@STEM@_evaluation_report{run_tag}.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)
print({'report': f'outputs/@STEM@_evaluation_report{run_tag}.json'})

# One row per Section 7 training run in this session, so a changed setting is read next to the default run (Section 10).
run_history = globals().get('run_history', [])
run_history.append({'run': len(run_history) + 1, 'data': 'BYOD' if USE_BYOD else 'sample', 'trainable_blocks': TRAINABLE_BLOCKS, 'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'trainable_parameters': adapt_result['n_trainable'], 'best_epoch': adapt_result['best_epoch'], 'pretrained_val_perplexity': round(frozen_val['perplexity'], 2), 'adapted_val_perplexity': round(adapted_val['perplexity'], 2), 'delta_val_perplexity': round(adapted_val['perplexity'] - frozen_val['perplexity'], 2), 'test_look': look, 'test_perplexity': 'not scored (exploratory)' if adapted_test is None else f"{frozen_test['perplexity']:.2f} -> {adapted_test['perplexity']:.2f}", 'val_records': adapted_val['records']})"""
)

_EXPORT_CODE = _py(
    """import csv
import shutil

rows = []
for record in unseen_records:
    head = opening(record)
    after = pipe.generate(head, max_new_tokens=GREEDY_MAX_NEW_TOKENS)
    rows.append({'id': record['id'], 'prompt': head, 'pretrained_completion': before[record['id']], 'adapted_completion': after['completion'], 'actual_continuation': record['text'][len(head):len(head) + 160], 'prompt_tokens': after['prompt_tokens'], 'new_tokens': after['new_tokens'], 'finished_by': after['finished_by']})
    print({'id': record['id'], 'prompt': head})
    print({'pretrained': rows[-1]['pretrained_completion']})
    print({'adapted': rows[-1]['adapted_completion']})
    print({'actual': rows[-1]['actual_continuation']})
single_report = evaluation_report(pipe.generate(opening(unseen_records[0]), max_new_tokens=GREEDY_MAX_NEW_TOKENS), sample_kind='one unseen SciTLDR abstract opening' if not USE_BYOD else 'one BYOD test record')
print({'single_prompt_report_verdict': single_report['verdict'], 'adapted_differs_from_pretrained': sum(r['pretrained_completion'] != r['adapted_completion'] for r in rows), 'of': len(rows)})
with open(f'outputs/@STEM@_completions{run_tag}.csv', 'w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

artifact_dir = Path(f'outputs/@STEM@_adapter{run_tag}')
shutil.rmtree(artifact_dir, ignore_errors=True)
pipe.save_artifact(artifact_dir, metadata={'tutorial': '@STEM@', 'data_source': data_source})
artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))
print({'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'trainable_blocks': artifact_manifest['adapter']['trainable_blocks'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'})

reloaded = GPT2TextGenerationPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)
parity_records = val_records[:10]
ppl_pair = (pipe.evaluate(parity_records)['perplexity'], reloaded.evaluate(parity_records)['perplexity'])
gen_pair = [(r['adapted_completion'], reloaded.generate(r['prompt'], max_new_tokens=GREEDY_MAX_NEW_TOKENS)['completion']) for r in rows]
parity = {'perplexity_in_memory': round(ppl_pair[0], 6), 'perplexity_reloaded': round(ppl_pair[1], 6), 'identical_completions': sum(a == b for a, b in gen_pair), 'of': len(gen_pair)}
print({'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']})
assert abs(ppl_pair[0] - ppl_pair[1]) < 1e-6 and parity['identical_completions'] == parity['of'], f'reload parity failed ({parity}): do not use this adapter; select Section 7 and choose Run after'
del reloaded

weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)
result_payload = {
    'notebook_source': NOTEBOOK_SOURCE,
    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],
    'model_id': MODEL_ID,
    'model_revision': MODEL_REVISION,
    'model_license': MODEL_LICENSE,
    'snapshot': {'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']},
    'data_source': data_source,
    'corpus': {'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'files': {k: {'name': v[0], 'bytes': v[1], 'sha256': v[2]} for k, v in CORPUS_FILES.items()}, 'license': CORPUS_LICENSE},
    'inference_contract': {'input_manifest': input_manifest, 'sanity_checks': checks, 'prompt': prompt, 'greedy': {k: greedy[k] for k in ('completion', 'prompt_tokens', 'new_tokens', 'finished_by', 'settings')}, 'sampled': {k: sampled[k] for k in ('completion', 'prompt_tokens', 'new_tokens', 'finished_by', 'settings')}, 'seconds_greedy_first_call': greedy_seconds},
    'comparison': comparison,
    'adaptation_outcome': adaptation_outcome,
    'floor_outcome': floor_outcome,
    'completions_before_after': rows,
    'single_prompt_report': single_report,
    'artifact': {'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors']), 'trainable_blocks': artifact_manifest['adapter']['trainable_blocks']},
    'reload_parity': parity,
    'test_look': look,
    'differs_from_preregistration': differs,
    'evaluation_protocol': {'preregistration': 'docs/reselection-2026-10-17.md', 'selected_configuration': selected_config, 'test_look': look, 'test_digest': dataset_manifests['test']['digest']},
    'exposed_development_test': exposed_report,
    'run_history': run_history,
    'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source},
}
with open(f'outputs/@STEM@_result{run_tag}.json', 'w', encoding='utf-8') as handle:
    json.dump(result_payload, handle, indent=2, ensure_ascii=False)
# Every export of this run describes one model: the report, the adapter and this file.
assert evaluation_report_payload['adaptation']['trainable_blocks'] == artifact_manifest['adapter']['trainable_blocks'] == TRAINABLE_BLOCKS, 'the report, the adapter and the form disagree on TRAINABLE_BLOCKS: select Section 7 and choose Run after'
print(sorted(os.listdir('outputs')))"""
)

_ACTIVITY_CODE = _py(
    """# Section 8 adds one row per Section 7 training run in this session; this cell only prints them.
columns = ['run', 'data', 'trainable_blocks', 'epochs', 'learning_rate', 'trainable_parameters', 'best_epoch', 'pretrained_val_perplexity', 'adapted_val_perplexity', 'delta_val_perplexity', 'test_perplexity']
print(' | '.join(columns))
for row in run_history:
    print(' | '.join(str(row.get(column)) for column in columns))
# Settings are compared pairwise: the same validation abstracts, scored after each run (bootstrap over records, seed 0).
for row in run_history[1:]:
    pair = paired_delta(run_history[0]['val_records'], row['val_records'])
    print({'paired_validation': f"run {row['run']} vs run 1", 'token_weighted_delta_bits': round(pair['token_weighted_delta_bits'], 3), 'bootstrap_95ci': [round(x, 3) for x in pair['bootstrap_95ci']], 'records_better_in_this_run': pair['n_improved'], 'of': pair['n']})
if len(run_history) == 1:
    print('One run so far. Set TRAINABLE_BLOCKS = 1 in the Section 7 training cell, select that cell and choose Runtime > Run after; this table then shows both runs.')"""
)

TEMPLATE = {
    "package": "gpt2_text_generation_pipeline",
    "repo_name": REPO,
    "stem": "gpt2_text_generation",
    "notebook_name": "gpt2_text_generation_colab.ipynb",
    "profile": "E2E",
    "notebook_kind": "TUTORIAL",
    "use_notice": '> **⚠️ Notice.** For research, training and evaluation only. Not validated for production, clinical, diagnostic or safety-critical use. Running this notebook downloads third-party model weights and datasets under their own licenses (see the model card) and uses compute on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are responsible for having the right to use it. Results here are tutorial evidence, not benchmarks. Everything is provided "as is", without warranty of any kind. To the maximum extent permitted by law, the maintainers and the DIMER platform accept no liability for its use. GPT-2 is a base language model: what it generates can be false, biased or offensive.',
    "mode": "GUIDED",
    "infrastructure_labels": True,
    "collapse_model_cell": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline):
    # managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`. The pins equal ast-audio-classification-pipeline's
    # plus `tokenizers==0.22.2`, which that lock already holds at that version, so the lock is ast-audio 16eee39's
    # (as reused by esm2-protein-pipeline) with the requesting project renamed and tokenizers marked as a direct pin.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (the "
        "kernel's own packages are left alone, so no restart is needed), stages and digest-verifies the pinned GPT-2 "
        "snapshot (safetensors, 548 MB), fetches the three digest-pinned SciTLDR-A files from the project repository "
        "(5.5 MB, no credential), reads the paper abstracts and draws 300 training and 50 validation documents and the "
        "pinned 100-document test draw that no selection step reads, from the release's own paper-disjoint members, generates greedy and seeded-sampled continuations of an "
        "unseen abstract's opening through the inference contract with an input manifest and a rejection probe, scores "
        "the pretrained model on the test abstracts by held-out perplexity beside the add-one unigram floor, replays "
        "the pre-registered three-configuration re-selection on the validation split, runs a bounded fine-tuning of the "
        "selected configuration with validation-perplexity epoch selection, scores the test split for it under the pre-registered settings, continues the same unseen abstracts with the adapted model, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify parity. The default path "
        "needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.5 §5). Every hosted run of every notebook version, with its "
        "measured times, is listed in `docs/release-verification.md`; the numbers quoted in this notebook's worked "
        "answers are the CPU build record of this version (2026-10-17, `tools/reselect_on_validation.py`, CPython 3.12.14, PyTorch 2.14.0+cpu; not a hosted run), pending a hosted run (REL13). Building the isolated environment and downloading the "
        "checkpoint usually take a few minutes; the Section 7 re-selection grid adds three short training runs "
        "(about 1.4 min on a T4, an estimate scaled from the recorded 32.7 s four-block T4 run; 17 min "
        "on the CPU build record's workstation CPU, measured)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4, select that cell and choose "
        "**Runtime → Run after** to supply your own text corpus as a CSV (columns `id`, `text`), a JSON array or JSONL "
        "file of `{id, text}` records, or a plain `.txt` file in which every blank-line-separated paragraph is one "
        "document — through the upload dialog on Colab, or as a path in `BYOD_PATH` on any runtime. Section 4 first "
        "returns the model to its pretrained weights, so your corpus passes through the same validation, seeded "
        "text-disjoint split, unigram floor, pretrained scoring, re-selection grid, fine-tuning, held-out evaluation, generation, artifact "
        "export and reload-parity cells as the SciTLDR sample (as an **exploratory** run: Sections 6 and 8 read your "
        "validation split and your test split is not scored), and the adapter it exports was trained on your corpus "
        "only. The expected schema, the minimum size (12 unique records) and the ceilings are stated in the "
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
            f"https://github.com/kurtvalcorza/{REPO}",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/gpt2_text_generation_colab.ipynb",
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
        "perplexity says the adapted model finds paper abstracts less surprising, not that it writes good ones.\n\n"
        "**Who this is for.** A learner who knows basic Python and has met the idea of a train/validation/test split, and "
        "wants to see how a pretrained language model is measured and adapted honestly: a trivial floor and the pretrained "
        "model first, a bounded fine-tune, one look at the held-out split, an exported adapter that reloads. No prior "
        "experience with GPT-2, transformers or fine-tuning is assumed; each term is explained where it is first used and "
        "again in the **Glossary** at the end. No GPU is required (CPU works; a GPU is used automatically).\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Generation (Sections 5 and 9) | Perplexity and adaptation (Sections 6–9) |\n"
        "|---|---|---|\n"
        "| Input | one English prompt of 1..1,023 tokens and the decoding settings | text records `id`, `text` split into training, validation and test |\n"
        "| Model | GPT-2 124M, pretrained or adapted, one next token at a time | the same model scored under teacher forcing; the last `TRAINABLE_BLOCKS` blocks trained |\n"
        "| Output | a continuation with `finished_by` and token counts; no score | held-out perplexity and bits per token beside the unigram floor and the pretrained model, and a safetensors adapter that reloads to the same numbers |\n\n"
        "**How to use this notebook.** Choose a runtime (Colab, Kaggle or Linux Jupyter; a GPU is faster, CPU works), then "
        "**Runtime → Run all**. Sections 1–3 are **infrastructure** — the isolated environment, the carried code and the "
        "model verification — and can be run without study; their code is collapsed. The learning path starts in Section "
        "4. Form fields (`# @param`) are the only values meant to be edited; the defaults reproduce the default path. Each "
        "learner section asks you to **predict** before it runs; the next section opens with **What to notice** and a "
        "collapsible **Check your reasoning** block with a worked answer. Each optional experiment names the field to "
        "change and the cell to re-run from (**Runtime → Run after**); every section that measures or trains the pretrained "
        "model first returns it to its pretrained weights, so a re-run never compares an adapted model with itself. "
        "Section 10 is a **change-one-thing activity**. **Troubleshooting**, a **Glossary** and a **Conclusion** template "
        "are at the end.\n\n"
        "**Roadmap:** *core concepts* — 4 a real corpus and a leakage-free split → 5 generation and its contract → 6 "
        "perplexity, a floor and the pretrained model; *evaluation practice* — 7 pre-registered re-selection and bounded "
        "fine-tuning with validation selection → 8 the held-out comparison; *engineering* — 9 before/after continuations, export and reload → 10 "
        "**change one thing: train one block** → conclude."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) say what a leakage-free split of a text corpus needs and check it (Section 4); "
        "(2) tell greedy decoding from seeded sampling and read `finished_by`, the token counts and the echoed settings "
        "(Section 5); (3) read a perplexity beside a unigram floor and say what it does and does not measure (Section 6); "
        "(4) state which parameters a bounded fine-tuning trains and how validation picks the configuration and the epoch (Section 7); (5) judge "
        "what a perplexity delta on 100 abstracts does and does not show (Section 8); (6) check that an exported adapter "
        "reproduces the evaluated model (Section 9); and (7) predict, measure and explain what changes when only one block "
        "is trained (Section 10)."
    ),
    "exclusions": (
        "chat or instruction following (GPT-2 has neither), batching (one prompt per call), raw logits or hidden states, "
        "beam search, non-English text, full-model or embedding fine-tuning, any quality, fluency or factuality score, "
        "and any claim that a SciTLDR abstract split stands in for your corpus. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python; what next-token prediction is; what a train/validation/test split protects against. The notebook explains greedy decoding and nucleus sampling, perplexity (the exponential of the mean per-token negative log-likelihood), bits per token, teacher forcing, the unigram floor, transformer blocks, epochs, the adapter and reload parity where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab, Kaggle or Linux Jupyter. Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter, and a Windows or macOS kernel is not supported (Section 1 stops with that message). The default path runs on CPU (float32) and uses CUDA automatically when available. Measured times, each with its run: the Colab CLI T4 run of the previous notebook version (2026-10-04, blob d88cc658) took 153.8 s for the whole session, 56 s of it building the environment and 32.7 s training four blocks; the CPU build record of this version (2026-10-17, `tools/reselect_on_validation.py`, CPython 3.12.14, PyTorch 2.14.0+cpu; not a hosted run) took 17 min for the three-configuration grid and 346 s for the selected configuration's two epochs. Every recorded run is listed in `docs/release-verification.md`. A Colab CPU runtime may be slower than a workstation (no hosted CPU run is recorded). The locked install (PyTorch 2.14.0 with its CUDA libraries) and the 548 MB checkpoint are the large downloads.",
        "- **Data contract:** records are `{id, text}` — one document of the domain, 1..4,000 characters and at most 1,023 BPE tokens (a longer record is refused, not truncated, when scored), ids matching `^[A-Za-z0-9_.:-]{1,64}$` and unique; at most 20,000 records. **Minimum size:** at least **12 unique records** with the default split fractions (15 % validation, 20 % test), because the training split must keep 8; a refusal names the number to supply. Texts are de-duplicated case-insensitively before splitting so the same document never sits in two splits; during training only, documents are truncated to 512 tokens and the end-of-text token is appended.",
        "- **BYOD file (Section 4):** CSV with an `id,text` header, a JSON array or JSONL of `{id, text}` objects, or a `.txt` file in which every blank-line-separated paragraph is one document, saved as UTF-8 (a spreadsheet's 'CSV UTF-8'). On Colab the upload dialog opens; on any runtime set `BYOD_PATH` to the file instead. Your training split is written to `outputs/gpt2_text_generation_train.csv`, the same file the sample writes, which is also a template of the expected shape.",
        "- **Validation is structural, not semantic:** nothing checks that a record belongs to the domain you mean — an off-topic corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal document collection is exactly that. The default path uploads nothing. A base language model can continue any prompt with false, biased or offensive text — read the output before reusing it.",
        "- **External access (data):** besides the Hub, the default path fetches three pinned objects (`train.jsonl` 3,155,015 bytes, `dev.jsonl` 1,124,865 bytes, `test.jsonl` 1,204,107 bytes; SHA-256 `b222771d…` / `3191fa98…` / `fb42dd6c…`) from `raw.githubusercontent.com` at the pinned `allenai/scitldr` commit over HTTPS, each refused on any mismatch before it is read; SciTLDR is Apache-2.0 (Cachola et al., 2020).",
        "- **Expected log lines:** during training, transformers may print that `loss_type=None` was set and the default loss is used — that is the standard next-token cross-entropy this notebook means, not a defect.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and split\n\n"
                "`fetch_corpus` downloads the three pinned SciTLDR-A files (or reads them from the cache), refuses a "
                "byte-size or SHA-256 mismatch per file before it is parsed, and `read_corpus` flattens each JSON-Lines "
                "member into records whose `text` is the abstract's sentences joined by a space — titles and TLDRs are "
                "left unread. `build_tutorial_dataset` keeps abstracts of 200..2,400 characters, drops repeated texts, and "
                "draws 300 training documents from the `train` member and 50 validation documents from `dev` by a seeded "
                "shuffle — the release's own paper-disjoint partition. The **test** split is 100 abstracts of the `test` "
                "member that no recorded selection or tuning step has read, pinned by paper id and digest (`FRESH_TEST_DIGEST`); the 100 `test` "
                "abstracts earlier versions scored are kept apart as the **exposed development test**. `validate_dataset` "
                "then checks every record against the contract, `check_split_disjoint` and `check_paper_disjoint` assert "
                "that no text and no paper appears in two splits, `check_near_duplicates` refuses abstracts that share 50 % or more of their word 8-grams across splits, and the training split is written to `outputs/{stem}_train.csv` in the shape BYOD expects. The "
                "cell first calls `pipe.reset_to_pretrained()`: a new dataset needs a model trained on nothing else, so "
                "any adaptation an earlier run made is undone.\n\n"
                "With `USE_BYOD = True`, your file (upload dialog on Colab, or `BYOD_PATH` on any runtime) goes through "
                "`load_byod_dataset` and `split_dataset` instead; see the Prerequisites for the format and the minimum "
                "size. A refusal of your file prints one message (no traceback), resets the run and keeps the files an "
                "earlier run wrote to `outputs/`.\n\n"
                "**Pre-registered evaluation protocol (2026-10-17).** The adaptation defaults of earlier versions (four "
                "blocks, learning rate 1e-4) came from a 2026-09 sweep that read the scores of the old 100 test abstracts, "
                "so those abstracts are no longer an independent test. Before anything in this version was trained or "
                "scored, this protocol was committed to the repository (`docs/reselection-2026-10-17.md`): (1) Section 7 "
                "re-selects on the **validation split only** among C0 = 4 blocks at 1e-4 (the incumbent), C1 = 2 blocks "
                "at 1e-4 and C2 = 2 blocks at 5e-5, with everything else fixed (2 epochs with the best epoch chosen on "
                "validation, batch 8, AdamW weight decay 0.01, clipping 1.0, `ADAPT_SEED = 0`, 512 training tokens, "
                "float32); (2) a configuration's score is its best-epoch validation perplexity, and a challenger replaces "
                "C0 only if it is at least 2 % better, with ties going to the smaller configuration; (3) the adapted model "
                "of the selected configuration is scored on the pinned test split (Section 8), next to the pretrained model on the "
                "same records (Section 6), **only under the exact pre-registered settings**: the sample with `SPLIT_SEED = "
                "42`, C0, and every fixed setting above. Any change (a form field, your own data, a smaller batch for "
                "memory) makes the run **exploratory**: it is compared on validation only, never scores the test split, "
                "and writes separate `_exploratory_<n>` output files. A later run with identical settings is a "
                "**reproduction** of the pre-registered evaluation; the recorded evaluation and its reproductions are "
                "listed in `docs/release-verification.md`; (4) the old 100 abstracts are "
                "reported once, labelled as the exposed development test, and choose nothing. Two limits of this design: "
                "the 2 % margin is a **predeclared practical tolerance**, not a noise threshold (nothing here measures "
                "seed-to-seed variation); and the grid holds only smaller or gentler challengers, while C0, the incumbent "
                "the margin protects, was first chosen in 2026-09 with the old test scores in view. The clean evidence is "
                "the fresh test split, not the re-selection.\n\n"
                "**Predict before running:** the sample's training, validation and test documents come from different "
                "papers. Why does that matter more for a language model than for, say, an image classifier?"
            ),
            "code": _DATA_CODE,
        },
        {
            "md": (
                "**What to notice:** 1,992 + 619 + 618 raw papers, four split digests, splits 300 / 50 / 100 plus the 100 "
                "exposed abstracts with no shared text or paper, `digest_is_pinned: True`, the written training CSV, and four refusal probes — a duplicate id, an empty text, a missing field "
                "and a dataset too small — each rejected before `torch` does anything.\n\n"
                "<details><summary>Check your reasoning</summary>A language model is scored on how well it predicts the "
                "exact words of a document. If the same or a near-identical abstract sat in both training and test, the "
                "adapted model would partly be recalling text it was trained on, and the test perplexity would look "
                "better than it is. Splitting by paper (and de-duplicating texts) keeps the test split unseen. On your "
                "own documents, split by collection or author when they come from one.</details>\n\n"
                "## 5. Generate through the inference contract in both decoding modes\n\n"
                "Before any adaptation, the inference contract is exercised as it always was. The prompt is the opening "
                "of one test abstract (about 120 characters, cut at a word boundary); `validate_inputs` applies exactly "
                "the checks `generate` applies (text type and character ceiling, `max_new_tokens` and the sampling settings "
                "within their ceilings) and returns an input manifest; the prompt-token ceilings need the real tokenizer and "
                "are enforced inside `generate`, which **rejects, never truncates**. An unseeded sampling request is "
                "validated too and its rejection recorded as a finding. `generate` returns `completion`, `text`, "
                "`prompt_tokens`, `new_tokens`, `finished_by` (`eos` when the model emitted token 50256 — GPT-2 ships no pad "
                "token, so the pipeline fixes `pad_token_id = eos_token_id = 50256` — or `max_new_tokens`) and echoes the "
                "settings. The greedy call is made twice and must repeat byte-identically (the reproducibility contract, "
                "on one device and dtype); the seeded sample is made twice and must repeat for the same seed. **Score "
                "semantics:** the pipeline emits **no probability, confidence or score of any kind** with a generation. "
                "Three further unseen-abstract openings are continued greedily here and kept as the *before* column for "
                "Section 9. The cell first returns the model to its pretrained weights, so the *before* column is the "
                "pretrained model also when you re-run from here.\n\n"
                "**Predict before running:** the greedy continuation is 32 new tokens. Will `finished_by` be `eos` or "
                "`max_new_tokens`, and will the two sampled continuations differ from the greedy one?"
            ),
            "code": _GENERATE_CODE,
        },
        {
            "md": (
                "**What to notice:** the ceilings, all eight sanity checks `True`, one recorded rejection, and the three "
                "printed strings.\n\n"
                "<details><summary>Check your reasoning</summary>Almost always `max_new_tokens`: a base model rarely "
                "ends a document within 32 tokens of an abstract's opening, so the output is cut, not finished — read it "
                "as a fragment. The sampled continuation usually differs from the greedy one (it draws from the "
                "distribution instead of taking the argmax), but the same seed repeats it exactly on the same host. "
                "Neither carries a score.</details>\n\n"
                "## 6. The unigram floor and the pretrained model's perplexity on the test split\n\n"
                "Two numbers frame the adaptation. `pipe.unigram_baseline` fits an add-one-smoothed unigram model over the "
                "50,257-token vocabulary on the training tokens and scores the test tokens with it: the perplexity a "
                "model that knows the domain's word frequencies but ignores every context reaches. `pipe.evaluate` scores "
                "the same test abstracts under **teacher forcing** with the pretrained model: every token after a "
                "record's first is predicted from the true tokens before it, the per-token negative log-likelihoods are "
                "averaged token-weighted and exponentiated (`perplexity`) or divided by ln 2 (`bits_per_token`); "
                "`record_perplexity` gives the spread across documents. Records over `MAX_PROMPT_TOKENS` are refused, "
                "never truncated. Under exploratory data settings (your own data, or a `SPLIT_SEED` other than 42) the cell reads the validation split instead of the test split and says so. The validation split is scored too, so Section 7 can check that its epoch 0 is this "
                "model. The cell first returns the model to its pretrained weights and refuses to label anything else "
                "pretrained.\n\n"
                "**Pretraining overlap.** GPT-2 was trained on WebText, pages linked from Reddit up to December 2017. Some "
                "SciTLDR papers were public on arXiv or OpenReview before then, so GPT-2 may have seen some of these very "
                "abstracts, or close variants, during pretraining. A rough check exists but is unverified: by a heuristic not "
                "confirmed against OpenReview's venue metadata, 9-character OpenReview forum ids belong to ICLR 2018 "
                "submissions, which were public from 27 October 2017, before the cutoff. 41 of the 100 test abstracts carry "
                "such ids (48 of the 100 exposed ones, 10 of the 50 validation ones), so a large share could have been in "
                "WebText if Reddit linked to them; it cannot be ruled out. The pretrained "
                "perplexity printed here may therefore be optimistic about unseen scientific text. The pretrained-versus-"
                "adapted comparison stays like-for-like (same records, same tokens), but read the absolute pretrained "
                "number with that caveat.\n\n"
                "**Scoring convention.** No end-of-text token is prefixed, each record's first token is not scored, and the "
                "end-of-text token appended during training is never scored. These perplexities are therefore not "
                "comparable to published GPT-2 figures, which usually prefix the end-of-text token and score every token. "
                "The add-one unigram floor is a deliberately weak, context-free reference (its 50,257 pseudo-counts "
                "outweigh the 62,583 training tokens); an add-0.1 reference is printed beside it.\n\n"
                "**Predict before running:** perplexity 1 means the model predicts every token with certainty; the "
                "unigram floor ignores context. Will the pretrained model be closer to the floor or to 1?"
            ),
            "code": _FLOOR_CODE,
        },
        {
            "md": (
                "**What to notice:** the unigram floor, the pretrained test perplexity with `'adapted': False`, the "
                "per-record spread, the validation perplexity and `floor_outcome`.\n\n"
                "<details><summary>Check your reasoning</summary>Far closer to 1 than to the floor. The CPU build record of this version (2026-10-17, `tools/reselect_on_validation.py`, CPython 3.12.14, PyTorch 2.14.0+cpu; not a hosted run) "
                "recorded an add-one floor of 1,674.5 (add-0.1: 1,204.7) and a pretrained test perplexity of 39.43 (5.30 bits "
                "per token) on the test split's 21,764 tokens: context cuts the model's uncertainty by a factor of about "
                "31 against the add-0.1 reference, or about 42 against the deliberately weak add-one floor. Part of "
                "that may be familiarity with this kind of text, or with these abstracts themselves (see the overlap note). "
                "The per-record spread (about 14 to 98) shows that some abstracts are much harder than others. On your own "
                "corpus `floor_outcome` is recorded, not asserted: a model that does not beat the floor there is a finding, "
                "not a crash.</details>\n\n"
                "## 7. Bounded fine-tuning on the training abstracts\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_BLOCKS` **transformer blocks** — four by default, "
                "28,351,488 of 124,439,808 parameters; the token and position embeddings, the tied output projection, the "
                "final layer norm and the earlier blocks stay frozen — with next-token cross-entropy on every token of "
                "every training abstract (the end-of-text token is appended so the model also learns where a document "
                "ends), AdamW at a fixed learning rate, gradient clipping at 1.0, shuffling seeded by `ADAPT_SEED` and no "
                "scheduler. Documents are truncated to `MAX_TRAIN_TOKENS` (512) **during training only**. An **epoch** is "
                "one pass over the training split. Epoch 0 records the pretrained model's validation perplexity, and the "
                "cell checks it equals the Section 6 validation number; every epoch is scored the same way, and the "
                "epoch with the lowest validation perplexity is kept.\n\n"
                "The cell first calls `pipe.reset_to_pretrained()`, and `adapt` refuses a pipeline that already holds an "
                "adaptation: every run starts from the pretrained weights, so a re-run with other settings never builds "
                "on an earlier one, and the exported adapter holds every block that differs from the base.\n\n"
                "**Re-selection on validation (the Section 4 protocol).** The first cell below replays the pre-registered "
                "grid: C0, C1 and C2 each train for two epochs from the pretrained weights and are scored on the validation "
                "split only, and `select_configuration` applies the margin rule and prints the branch that decided. "
                "The CPU build record of this version (2026-10-17, `tools/reselect_on_validation.py`, CPython 3.12.14, PyTorch 2.14.0+cpu; not a hosted run) selected **C0** (no challenger beat C0 by 2% or more; C0 stays) at the fixed two-epoch budget: every configuration's "
                "best epoch was its last, so validation was still falling and the choice holds at this budget only "
                "(`EPOCHS = 4` in the Next experiments probes it). C0 was also the lowest outright, so the margin did not "
                "decide. The training cell's form defaults "
                "are C0's settings (4 blocks, learning rate 1e-4). `RESELECT_ON_VALIDATION = False` "
                "skips the grid, which costs three short training runs (about 1.4 min on a T4, an estimate; "
                "17 min on the CPU build record's CPU). On your own data, or on another `SPLIT_SEED` draw, the replay's winner is the "
                "selected configuration and the training cell trains it unless you change its two fields; the 2 % margin "
                "was justified on SciTLDR only.\n\n"
                "**Predict before running:** will either two-block configuration beat four blocks by 2 % on validation?"
            ),
            "code": _GRID_CODE,
        },
        {
            "md": (
                "**What to notice:** each configuration's validation curve and best epoch, the `selected` line with its "
                "`reason`, the `paired_validation_vs_C0` lines and `reproduces_recorded_selection`.\n\n"
                "<details><summary>Check your reasoning</summary>"
                "The CPU build record of this version (2026-10-17, `tools/reselect_on_validation.py`, CPython 3.12.14, PyTorch 2.14.0+cpu; not a hosted run) recorded best-epoch validation perplexities of C0 35.19 (epoch 2, 393 s); C1 36.88 (epoch 2, 313 s); C2 37.98 (epoch 2, 322 s). Training more "
                "blocks gives the model more room to fit the genre in two epochs; the margin rule keeps the incumbent "
                "unless a smaller configuration is at least 2 % better, a predeclared practical tolerance, not a measured "
                "noise level. Paired over the same 50 validation abstracts, the build record put C1 at "
                "+0.068 bits per token against C0 (bootstrap 95 % interval +0.056 to +0.079; 47 of 50 abstracts "
                "worse) and C2 at +0.110 (+0.092 to +0.128; 48 of 50 worse), on one seed. On a "
                "GPU the numbers can differ slightly from the CPU record, which is why the selection is stated in advance "
                "and only replayed here.</details>\n\n"
                "**Train the selected configuration.** The next cell trains the form settings, by default the selected "
                "configuration (the replay's choice on your data or another draw), once more from the pretrained weights (same seed, so on one device it repeats the grid's "
                "run), checks epoch 0 against Section 6, and records whether these are the selected settings: the test "
                "split is scored in Section 8 only if they are.\n\n"
                "**Predict before running:** validation perplexity starts near the Section 6 value. After two epochs on "
                "300 abstracts, will it fall by a few tenths, a few points, or by half?"
            ),
            "code": _ADAPT_CODE,
        },
        {
            "md": (
                "**What to notice:** epoch 0 with its relative difference to Section 6 (0 on one device), the trainable "
                "and total parameter counts, the validation perplexity per epoch, `best_epoch` and `these_are_the_selected_settings`.\n\n"
                "<details><summary>Check your reasoning</summary>A few points. The CPU build record recorded validation "
                "perplexity 40.97 → 36.71 → 35.19 (best epoch 2) for C0; the Colab CLI T4 run of the previous version "
                "(2026-10-04, same default settings) recorded 40.97 → 36.59 → 35.20. A fall of about five points is large for two "
                "epochs, but these are 300 abstracts of one narrow genre; validation chose the configuration and the "
                "epoch, so the test split is still unused.</details>\n\n"
                "## 8. Held-out evaluation\n\n"
                "The test split is used for no training, configuration or epoch choice, no recorded selection or tuning step "
                "has read it, and no abstract or paper in it appears in another split. It is scored only when every pre-registered "
                "setting holds; any other run (the Section 10 activity, your own data, another `SPLIT_SEED`) is "
                "exploratory and is read on validation instead. `test_look` labels each run: `pre-registered evaluation`, "
                "`reproduction` (an identical run after one), or `not scored (exploratory)`. The adapted model is scored exactly as the pretrained model was in Section "
                "6, and the three numbers are put side by side with `delta_vs_pretrained`. `paired_per_record` compares each "
                "abstract with itself: how many got easier, the median and quartiles of the per-record change in bits per "
                "token, and a bootstrap 95 % interval of the token-weighted change (records resampled, seed 0). The "
                "marginal min / median / max cannot show this: the hardest abstract before and after need not be the same one. "
                "`adaptation_outcome` records whether the adaptation lowered held-out perplexity — a recorded outcome, "
                "not an assertion, because on your own corpus or with other settings an adaptation that does not help is "
                "a legitimate result to read. One hundred abstracts from one seeded split of one corpus give a paired "
                "bootstrap interval over records but no seed-to-seed estimate; the delta is sample-sanity evidence that the adaptation contract works, not a benchmark, and a lower perplexity on "
                "paper abstracts says nothing about your corpus until you measure it there. The cell also adds this run "
                "to `run_history`, which Section 10 prints. The **exposed development test**, the old 100 abstracts that the "
                "2026-09 sweep read when it chose the earlier defaults, is printed once beside the test split, labelled, "
                "for continuity with earlier records; it is not independent evidence.\n\n"
                "**Predict before running:** how much will the selected configuration move the test perplexity from the "
                "pretrained value, and will every abstract improve?"
            ),
            "code": _EVAL_CODE,
        },
        {
            "md": (
                "**What to notice:** the three-way perplexity row, `delta_vs_pretrained`, `paired_per_record` (with "
                "`n_improved` and the interval), the `era_probe` lines, `adaptation_outcome` with `read_on`, and the labelled `exposed_development_test` line.\n\n"
                "<details><summary>Check your reasoning</summary>The CPU build record of this version (2026-10-17, `tools/reselect_on_validation.py`, CPython 3.12.14, PyTorch 2.14.0+cpu; not a hosted run) recorded 39.43 → "
                "34.25 on the test split (Δ −5.18 perplexity, −0.20 bits per token). Paired per abstract, 99 of 100 got "
                "easier and 1 got harder: not every abstract improves. The median change was −0.204 bits per token "
                "(quartiles −0.285 to −0.138), and the bootstrap 95 % interval of the token-weighted change, −0.224 to "
                "−0.182, is well away from 0 for this seed and sample. The marginal maximum fell from 98 to 90, but that "
                "alone could not have shown any of this. By the heuristic era split, the 9-character-id abstracts moved "
                "36.63 → 32.03 (41 abstracts) and the others 41.69 → 36.04 (59). The exposed "
                "development test moved 40.17 → 34.73 (Δ −5.44; 99 of 100 easier, interval −0.230 to −0.190 bits); the Colab CLI T4 run of the previous version "
                "recorded 40.18 → 34.70 on those abstracts, with defaults chosen on their scores. About "
                "0.20 bits per token is the gain on abstracts that no recorded choice read, on one seed and one small "
                "sample of one corpus.</details>\n\n"
                "## 9. Continue unseen abstracts before and after, export the adapter and reload it\n\n"
                "The three abstract openings continued by the pretrained model in Section 5 are continued again by the "
                "adapted model through the same `generate` contract, and the two completions are printed side by side "
                "with the full prompt and the abstract's actual continuation. Read them as text, not as evidence: a "
                "perplexity gain is measured on the model's likelihoods, and nothing here scores a completion. The "
                "single-prompt `evaluation_report` helper — the inference-stage helper — is written for the first of "
                "them and stays `not-measurable`, because a continuation has no ground truth.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last `TRAINABLE_BLOCKS` transformer blocks, about "
                "113 MB for four — as `adapter.safetensors` (**safetensors** stores tensors without executable code), "
                "with a `manifest.json` recording the artifact format, the base model id and revision, the digest of the "
                "base `model.safetensors`, the tensor names, the file size and SHA-256, the training configuration and the "
                "epoch history (OUT8). It refuses to export if any block outside the adapter differs from the pretrained "
                "base, because base + adapter would then not be the model in memory. "
                "`GPT2TextGenerationPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest "
                "and digest **before** deserialising, refuses any tensor that is not a transformer-block tensor of the "
                "base, and overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory "
                "model (VER2). The cell asserts that the perplexity of ten validation records agrees within 1e-6 "
                "and that the greedy completions are identical (**reload parity**, VER4), and that the report, the adapter and the form agree on "
                "`TRAINABLE_BLOCKS`.\n\n"
                "**Predict before running:** will the adapted continuations read more like an abstract than the "
                "pretrained ones, and will the reloaded model's perplexity be identical or only close?"
            ),
            "code": _EXPORT_CODE,
        },
        {
            "md": (
                "**What to notice:** each prompt in full beside the pretrained, adapted and actual text; "
                "`adapted_differs_from_pretrained`; the adapter's `trainable_blocks`, tensor count and size; and "
                "`reload_parity`.\n\n"
                "<details><summary>Check your reasoning</summary>Often a little more abstract-like (\"we propose\", "
                "\"our method\"), but three greedy fragments are anecdotes, not evidence; the CPU build record changed "
                "3/3. Reload parity holds within 1e-6, the assertion's tolerance: the CPU build record "
                "printed 36.054856 both ways for the ten validation records and 3/3 identical "
                "completions, from an adapter of 48 tensors (113,410,784 bytes). "
                "Loading a file is not reproducing a result; the parity check is what shows the export is the model that "
                "was evaluated.</details>\n\n"
                "## 10. Your turn — change one thing: train one block\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.**\n\n"
                "1. **Predict:** with `TRAINABLE_BLOCKS = 1` only the last transformer block trains (7,087,872 parameters "
                "instead of 28,351,488). Will the validation perplexity gain be larger, smaller or the same? Write your guess "
                "down.\n"
                "2. **Change one thing:** in the Section 7 training cell set `TRAINABLE_BLOCKS = 1` and nothing else.\n"
                "3. **Run:** select the Section 7 training cell and choose **Runtime → Run after** (it re-runs the rest of Sections 7–10, so "
                "the report, the completions, the adapter and the result file are all rewritten for this run and "
                "describe one model). `adapt` starts again from the pretrained weights, and Section 6's pretrained "
                "numbers stay valid.\n"
                "4. **Observe:** this cell prints one row per Section 7 run in this session — the setting, the trained "
                "parameter count, the best epoch, the pretrained and adapted validation perplexity with their delta, and the test "
                "perplexity only for a pre-registered run (`not scored (exploratory)` otherwise, so the test split is "
                "never used to compare settings).\n"
                "5. **Explain:** what did training three more blocks buy? Read the `paired_validation` line under the "
                "table: the same validation abstracts scored after both runs, with a bootstrap 95 % interval of the "
                "token-weighted difference. Is that interval clearly away from zero? (The spread across records in "
                "Section 8 is the wrong yardstick: it mixes easy and hard abstracts, while both runs see the same ones.)\n\n"
                "<details><summary>Check your reasoning</summary>Usually smaller: fewer trainable "
                "parameters can adapt less of the model's behaviour to the genre in two epochs. No run of this activity "
                "on the 300-abstract sample is recorded yet, so compare your two rows. The only measurement so far is a "
                "local Windows CPU check of the previous notebook version (091ba0c) on a 12-abstract BYOD corpus (8 training records, 2026-10-04): "
                "four blocks lowered the test perplexity of its two test abstracts by 0.32 (35.87 → 35.55), one block by "
                "0.20 (→ 35.67) — the same direction, on far too little data to be more than a sanity check. The pretrained number is the same in both rows because each run starts from the "
                "pretrained weights; before this notebook reset them, a second run silently continued from the first and "
                "its adapter did not reload to the same model. Other experiments in the same pattern: `EPOCHS = 4`, "
                "`LEARNING_RATE = 5e-5`, `TRAINABLE_BLOCKS = 12` — each adds a row.</details>"
            ),
            "code": _ACTIVITY_CODE,
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The pretrained 2019 model already finds paper abstracts far less surprising than a unigram model does (about 40 "
        "against a floor in the thousands — WebText contains scientific prose, and it may contain some of these abstracts "
        "or close variants, which has not been ruled out, so the absolute pretrained number may be optimistic). A bounded "
        "fine-tuning of the selected configuration (the last four transformer blocks by default) on 300 abstracts lowers held-out perplexity by a few points in a "
        "few minutes on CPU, with an adapter that reloads to the same likelihoods (within 1e-6) and identical completions. That is the "
        "claim: the adaptation contract can adapt the model to a domain end to end on a real corpus, and the number it "
        "produces is read against the pretrained model and a context-free floor rather than in isolation. The pretrained "
        "and adapted numbers are measured on the same records and tokens, so their difference is like-for-like; the "
        "configuration was re-selected on validation by a pre-registered rule, at the fixed two-epoch budget only, and the "
        "test split is 100 abstracts that no recorded selection or tuning step has read; each run scores it once, for the "
        "pre-registered configuration only. The exposed development test is reported for continuity only.\n\n"
        "Perplexity is intrinsic: it says how well the model predicts text it did not write, token-weighted, on one seeded "
        "split of one corpus with a paired bootstrap over records but no seed-to-seed dispersion estimate. It is not fluency, factuality, usefulness or safety, and a lower "
        "perplexity does not make the completions in Section 9 better — they are unscored continuations from a base "
        "language model and can be false, repetitive, biased or offensive. The adapter changes the last blocks, which every "
        "prompt shares, so the model's behaviour on other text shifts too; nothing here measures that. Greedy decoding "
        "repeats itself; seeded sampling is reproducible only on the same host.\n\n"
        "Three things to carry to real data. **Floors first:** the unigram floor and the pretrained perplexity on *your* "
        "held-out documents are the numbers to read before any adapted one. **Leakage:** de-duplicate texts across splits "
        "(the contract does this case-insensitively) and split by document collection or author when your documents come "
        "from one. **Ceilings:** documents over `MAX_PROMPT_TOKENS` are refused when scored and truncated to 512 tokens "
        "only during training — long documents need chunking that this pipeline does not provide.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real corpus, validate "
        "the demonstrated dataset contract without leakage, execute the inference contract in both decoding modes and a "
        "bounded fine-tuning, evaluate by perplexity against a trivial floor and the pretrained model on a pinned "
        "test split that no selection step reads, for the configuration selected on validation, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** "
        "establish benchmark superiority, text quality or factual reliability on any domain, a usable acceptance threshold, "
        "or production fitness.\n\n"
        "**Next experiments** (each starts after the default Run all; every adaptation starts again from the pretrained "
        "weights):\n\n"
        "1. **Trainable blocks:** the Section 10 activity (`TRAINABLE_BLOCKS = 1`, or 12), **Run after** from Section 7.\n"
        "2. **More epochs:** in Section 7 set `EPOCHS = 4`, then **Run after** from Section 7, and watch whether validation "
        "perplexity keeps falling or turns (the best epoch is kept either way); a new Section 10 row records it on validation.\n"
        "3. **Another sampled continuation:** in Section 5 change `SEED`, then **Run after** from Section 5 (Section 5 "
        "returns the model to its pretrained weights, so the whole adaptation runs again and the *before* column stays the "
        "pretrained model).\n"
        "4. **Your own documents:** in Section 4 set `USE_BYOD = True` (and upload, or set `BYOD_PATH`), then **Run after** "
        "from Section 4. Read the unigram floor and the pretrained number before the adapted one.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment is built from manylinux "
        "wheels; use Colab, Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 fails to download `uv`, Python or a package.** The runtime needs `pypi.org`, "
        "`files.pythonhosted.org` and the python-build-standalone release host. Re-run the cell; a size or SHA-256 mismatch "
        "is refused on purpose.\n"
        "- **\"The isolated environment's Python process exited\".** Usually out of memory. Restart the session and choose "
        "**Run all** again; on a small runtime lower `BATCH_SIZE` in Section 7.\n"
        "- **Section 3 reports a size or SHA-256 mismatch.** A snapshot file was altered or truncated; delete "
        "`weights/gpt2/model.safetensors` and run Section 3 again.\n"
        "- **Section 4 cannot fetch or refuses a SciTLDR file.** The runtime needs `raw.githubusercontent.com`; a size or "
        "SHA-256 mismatch is refused on purpose. Delete `weights/scitldr/` and run Section 4 again.\n"
        "- **Section 4: \"USE_BYOD = True needs a file\".** This runtime has no upload dialog; set `BYOD_PATH` to your "
        "file.\n"
        "- **Section 4: \"Upload exactly one … file (got 0 …)\".** The upload was cancelled or empty. Run Section 4 again "
        "and pick one file, or set `BYOD_PATH`.\n"
        "- **Section 4: \"BYOD refused: …\".** The message names what is wrong with the file; nothing from it is used, the "
        "run is reset, and files an earlier run wrote to `outputs/` are kept. Fix the file and run Section 4 again, then "
        "**Run after**.\n"
        "- **Section 4: \"… is not UTF-8 text\".** Save the file as UTF-8 (a spreadsheet's 'CSV UTF-8') and run Section "
        "4 again.\n"
        "- **Section 4 refuses a BYOD dataset.** The message names the record and the rule (missing column, empty text, "
        "duplicate id, id pattern, too long), or says how many unique records to supply (12 with the default fractions). "
        "Fix the file and run Section 4 again (**Run after**).\n"
        "- **\"this pipeline already holds an adaptation\".** `adapt` was called without resetting; select Section 7 and "
        "choose **Run after** (its first line resets the model).\n"
        "- **Section 9: \"nothing to save: call adapt() first\".** A Section 4–6 cell was re-run after Section 7 and "
        "returned the model to its pretrained weights; select Section 7 and choose **Run after**.\n"
        "- **Section 9's parity assertion fails.** The export or reload is broken; do not use that adapter. Select Section "
        "7 and choose **Run after**.\n"
        "- **Slow on CPU.** Each epoch trains on 300 abstracts through the 12-block model; a GPU runtime is several times "
        "faster. Lower `EPOCHS` for a quicker experiment.\n"
        "- **CUDA out of memory.** Lower `BATCH_SIZE` in Section 7, or switch the runtime to CPU.\n\n"
        "## Glossary\n\n"
        "- **Token / byte-level BPE:** the pieces GPT-2 reads and writes; common words are one token, rare words several. "
        "The end-of-text token (50256) marks where a document ends.\n"
        "- **Base language model:** a model trained only to predict the next token of web text; no instruction following "
        "or safety tuning.\n"
        "- **Greedy decoding / nucleus sampling:** take the most likely next token every time / draw from the smallest set "
        "of tokens whose probabilities add up to `top_p`, after `temperature` rescaling; a `seed` makes the draw "
        "repeatable.\n"
        "- **Teacher forcing:** scoring each token given the true tokens before it, not the model's own output.\n"
        "- **Negative log-likelihood (NLL) / perplexity / bits per token:** how surprised the model is by each true token "
        "(in nats) / the exponential of the mean NLL, an effective number of equally likely choices per token / the mean "
        "NLL divided by ln 2.\n"
        "- **Unigram floor (add-one smoothing):** a model that predicts each token from its frequency in the training "
        "split alone, with one added to every count so no token has probability zero.\n"
        "- **Transformer block / trainable blocks:** one of GPT-2's 12 stacked attention-plus-feed-forward layers; only the "
        "last `TRAINABLE_BLOCKS` are trained here.\n"
        "- **Epoch / validation selection:** one pass over the training split / keeping the epoch with the lowest "
        "validation perplexity, so the test split stays unused until the end.\n"
        "- **Pre-registration / exposed test:** writing down the configurations, the selection rule and the test set "
        "before any of them is scored / a test set whose scores already influenced a choice, so it is no longer "
        "independent evidence.\n"
        "- **Pretrained weights / reset:** the downloaded GPT-2 weights; `reset_to_pretrained()` restores them after an "
        "adaptation.\n"
        "- **Adapter / safetensors / reload parity:** the saved trained blocks / a code-free tensor file format / the check "
        "that base + adapter reproduces the in-memory model's perplexity (within 1e-6) and completions.\n"
        "- **Pretraining overlap (contamination):** test text that the model may already have seen during pretraining, "
        "which makes its numbers look better than on truly new text.\n\n"
        "## Conclusion (your notes)\n\n"
        "Fill in from the numbers this run printed; keep each claim to what the evidence shows.\n\n"
        "- **Task:** which corpus, which split sizes, how many test tokens?\n"
        "- **Principal result:** the adapted test perplexity and bits per token, with `n_tokens`.\n"
        "- **Baselines / reference:** the unigram floor and the pretrained model on the same tokens, and "
        "`delta_vs_pretrained`.\n"
        "- **Uncertainty or failure mode:** the per-record spread; what did your Section 10 run change?\n"
        "- **Limitations:** what does this run *not* show (pretraining overlap, one small test sample read only under the pre-registered settings, the "
        "exposed development test's history, one corpus, no quality score)?\n\n"
        "## References\n\n"
        "[1] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, \"Language models are unsupervised multitask learners,\" OpenAI, San Francisco, CA, USA, Tech. Rep., 2019. [Online]. Available: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf\n"
        "[2] I. Cachola, K. Lo, A. Cohan, and D. S. Weld, \"TLDR: Extreme summarization of scientific documents,\" in *Findings of the Association for Computational Linguistics: EMNLP 2020*, 2020. [Online]. Available: https://arxiv.org/abs/2004.15011\n"
        "[3] OpenAI, \"gpt-2: Code for the paper 'Language models are unsupervised multitask learners',\" GitHub repository. [Online]. Available: https://github.com/openai/gpt-2\n"
        "[4] OpenAI, \"openai-community/gpt2,\" Hugging Face model repository, revision {MODEL_REVISION}. [Online]. Available: https://huggingface.co/{MODEL_ID}\n"
        "[5] K. Valcorza, \"gpt2-text-generation-pipeline: README, model card and weight provenance,\" GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/gpt2-text-generation-pipeline (README.md, MODEL_CARD.md, docs/WEIGHTS.md)\n"
        "[6] K. Valcorza, \"Re-selection of the adaptation defaults on validation (2026-10-17),\" pre-registration and record, docs/reselection-2026-10-17.md in [5]. [Online]. Available: https://github.com/kurtvalcorza/gpt2-text-generation-pipeline/blob/main/docs/reselection-2026-10-17.md\n"
        "[7] DIMER platform, \"DIMER Notebook Specification 2.5\" and \"Model Card Specification 1.1,\" fleet specifications in the ml-worker repository, 2026."
    ),
}
