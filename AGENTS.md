# AGENTS.md

## Cursor Cloud specific instructions

SAM-Audio is a **Python ML library** (not a web/service app) for isolating sounds in audio
from text / visual / temporal prompts. Public API: `from sam_audio import SAMAudio, SAMAudioProcessor`.
There is no server to run and no automated test suite in this repo — "running the app" means
importing the package and running the separation/codec pipeline in Python.

### Environment facts
- Python 3.12; dependencies are installed into the **user site** (`~/.local/...`) and the package
  is installed via `pip install --user .`. The install script in `.cursor/environment.json` keeps this current.
- **CPU-only** VM (no GPU). Everything runs on CPU; expect the benign
  `WARNING[XFORMERS] ... can't load C++/CUDA extensions` and a `pynvml deprecated` warning — both
  are harmless on CPU.
- `~/.local/bin` is added to `PATH` via `~/.bashrc`. In a non-login shell it may not be present; if
  a console script (e.g. `ruff`, `hf`) is "not found", invoke via module instead (e.g. `python3 -m ruff`).

### Dependency version pin (important, non-obvious)
`pyproject.toml` leaves `transformers` unpinned (`>=4.54.0`) and does not pin `huggingface_hub`.
A fresh `pip install .` resolves to `transformers 5.x` + `huggingface_hub 1.x`, which **breaks
`SAMAudio.from_pretrained` / `SAMAudioProcessor.from_pretrained`** with
`TypeError: _from_pretrained() missing ... 'proxies' and 'resume_download'` (hub 1.x changed the
`ModelHubMixin._from_pretrained` calling convention). The install script in
`.cursor/environment.json` pins `transformers>=4.54,<5` (pulls a compatible
`huggingface_hub<1.0`, currently 4.57.6 / 0.36.2) to fix this. Do not "upgrade"
transformers to 5.x.

### Lint (this is the CI check — see `.github/workflows/ci.yaml`)
Run from repo root: `python3 -m ruff format --check .` and `python3 -m ruff check .`.
Match the pre-commit-pinned tool version `ruff==0.12.0` (installed by the environment
install script); newer ruff reformats Markdown code blocks in `README.md` and will fail
`format --check`.

### Running the model (gated checkpoints)
The pretrained checkpoints (`facebook/sam-audio-*`, `facebook/sam-audio-judge`, PE-AV span
predictor) are **gated on Hugging Face**. Without auth, `from_pretrained` raises
`GatedRepoError` (401). To run the real product you must (1) request/receive access to the gated
repos on your HF account and (2) provide an `HF_TOKEN` (then `hf auth login` or set the env var).
Downloads are multi-GB and inference is slow on CPU.

### Verifying the env without gated weights
The DACVAE audio codec + `SAMAudioProcessor` run fully offline with random weights and exercise the
core audio pipeline (preprocess -> encode to latents -> decode). This is the recommended smoke test
when no HF token/GPU is available.
