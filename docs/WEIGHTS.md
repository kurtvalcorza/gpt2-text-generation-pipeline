# Weight provenance and DIMER hosting

- Upstream: `openai-community/gpt2`
- Immutable revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Weight format: SafeTensors (`model.safetensors`, 548,105,171 bytes)
- Manifest: `weights/gpt2/dimer-base-manifest.json` (15 files, 554,331,411 bytes total, per-file SHA-256). The snapshot also carries an `onnx/` subdirectory of 7 config/tokenizer files from the upstream ONNX export; they are verified by digest and unused by this package. The upstream repository's TensorFlow, Flax, ONNX and TFLite weight files are not part of the manifest and are not staged.
- Upstream weight license: MIT
- DIMER hosting: the MIT license permits use, modification, distribution, sublicensing, and commercial use subject to preservation of the copyright and license notice. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision, into the snapshot directory; `verify_snapshot()` then checks every file before any load.
- Loader trust boundary: Transformers `GPT2LMHeadModel` / `GPT2TokenizerFast` with `trust_remote_code=False`, `local_files_only=True` from the verified directory, float32.

## Adaptation corpus (tutorial data, not weights)

- Corpus: SciTLDR-A paper abstracts (Cachola et al., EMNLP Findings 2020), `allenai/scitldr` at commit `5ccad9c00a60ad75c9e04abf7f27d0f53f983b20`, licence Apache-2.0.
- Files: `SciTLDR-Data/SciTLDR-A/train.jsonl` (3,155,015 bytes, SHA-256 `b222771d387be585cfdf5ae957b36757138415a352e0a3e3b23f73f87c3b1119`), `dev.jsonl` (1,124,865 bytes, `3191fa98ccc09521332b7a1cd63b1930be4e8df125a235ccd31e40329709525e`), `test.jsonl` (1,204,107 bytes, `fb42dd6cd4f4a1928ae8a01a189456fbfe994a07e938bd49f68653933f6503c9`), fetched by `samples.fetch_corpus` from `raw.githubusercontent.com` over HTTPS at run time into the git-ignored `weights/scitldr/` cache and refused on any byte or SHA-256 mismatch. Only the `source` (abstract sentences) field is read; titles and TLDRs are ignored.
- Sample: `build_sample_dataset(seed=42)` draws 300 / 50 / 100 `{id, text}` records from the release's own train/dev/test members (abstracts of 200..2,400 characters, de-duplicated). The repository redistributes none of the corpus; DIMER hosting of the weights is unaffected.
- Adapter artifacts written by the tutorial (`outputs/gpt2_text_generation_adapter/`, `org.valcorza.gpt2.adapter.v1`, about 113 MB) carry only the trained transformer-block tensors and a manifest naming the base `model.safetensors` digest; they are outputs, not hosted weights.
