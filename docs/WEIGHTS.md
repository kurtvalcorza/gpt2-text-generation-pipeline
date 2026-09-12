# Weight provenance and DIMER hosting

- Upstream: `openai-community/gpt2`
- Immutable revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Weight format: SafeTensors (`model.safetensors`, 548,105,171 bytes)
- Manifest: `weights/gpt2/dimer-base-manifest.json` (15 files, 554,331,411 bytes total, per-file SHA-256). The snapshot also carries an `onnx/` subdirectory of 7 config/tokenizer files from the upstream ONNX export; they are verified by digest and unused by this package. The upstream repository's TensorFlow, Flax, ONNX and TFLite weight files are not part of the manifest and are not staged.
- Upstream weight license: MIT
- DIMER hosting: the MIT license permits use, modification, distribution, sublicensing, and commercial use subject to preservation of the copyright and license notice. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision, into the snapshot directory; `verify_snapshot()` then checks every file before any load.
- Loader trust boundary: Transformers `GPT2LMHeadModel` / `GPT2TokenizerFast` with `trust_remote_code=False`, `local_files_only=True` from the verified directory, float32.
