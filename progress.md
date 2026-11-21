# Project Progress

**Project:** Car Service Manager with Voice Agent  
**Repository:** https://github.com/AadCode7/Car_Service_Manager_TTS  
**Working branch:** `main`  
**Last updated:** 2026-11-21

## Project objective and confirmed constraints

Build a local employee-facing car service management application with a voice agent. The application frontend/backend/database run locally. Speech-model training and inference use remote GPU resources because the laptop's GTX 1650 is not the target for model training or inference.

The main research objective is to build our own English speech recognition (ASR) and speech synthesis (TTS) models with weights trained from random initialisation. Pretrained speech checkpoints and fine-tuning a pretrained model do not satisfy the main objective. Groq is reserved for conversational LLM reasoning, not ASR/TTS.

The application reads accurate customer, vehicle and service records held by the company. The user supplies optimal/reference vehicle values. AI-generated customer-request details are editable drafts and require supervisor approval before official records/work orders change. Payment scope is estimates and payment breakdowns only, with no payment gateway or transactions. Voice interaction is a browser/laptop microphone and speaker simulation; Vapi is optional for comparison, not a core dependency.

## Current repository state

- [x] Confirmed repository exists and default branch is `main`.
- [x] Created `architecture.md` and `progress.md`.
- [x] Added a `.gitignore` for datasets, recordings, model weights, tokenizers, artifacts, secrets and runtime files.
- [x] Added the ASR training package under `training/asr/`.
- [x] Added a notebook-specific ASR README with Kaggle/Colab instructions.
- [x] Python syntax compilation passed in GitHub Actions.
- [ ] Run the smoke test in a CUDA notebook.
- [ ] Run a short real-data training experiment.
- [ ] Run full LibriSpeech train/validation/test experiments.
- [ ] Review measured performance before choosing a larger/next architecture.

## Technology and model decisions

| Area | Decision | Status |
|---|---|---|
| Frontend/backend/database | Local React + TypeScript, FastAPI, SQLite | Planned |
| Conversation LLM | Groq API behind a backend adapter | Planned |
| ASR | Random-initialised English FastConformer-inspired Conformer-CTC | Implemented as first baseline |
| ASR feature frontend | 16 kHz, 80 log-Mel features, per-utterance CMVN | Implemented |
| Tokenizer | SentencePiece unigram/BPE; ID 0 reserved for CTC blank | Implemented |
| Encoder | 8x depthwise-separable convolutional subsampling; 9-frame Conformer kernels; Macaron FFNs; rotary self-attention; GLU/depthwise convolution; PyTorch SDPA | Implemented |
| Training | CTC loss, SpecAugment, speed perturbation, duration buckets, AdamW, warmup/cosine schedule, AMP, gradient clipping, EMA | Implemented |
| Evaluation | Validation/test loss, corpus WER/CER, utterance predictions and optional CTC prefix beam search | Implemented |
| Charts | Loss, WER/CER, learning rate, epoch time, per-utterance WER distribution and WER vs duration | Implemented |
| Model storage | Best checkpoint, last/resume checkpoint, tokenizer and model/audio config | Implemented |
| Temporary inference | API-key-protected FastAPI endpoint; can be exposed through a temporary tunnel | Implemented |
| TTS | Random-initialised English TTS model | Planned after ASR baseline |
| Vapi | Optional browser-based comparison | Optional |

### Research positioning

The current code is a **research-informed FastConformer-inspired Conformer-CTC baseline**, not a faithful Zipformer/FastConformer reproduction and not a claim of state-of-the-art benchmark performance. It uses recent, practical techniques in a compact implementation suitable for an initial free-notebook experiment. The architecture roadmap documents a Zipformer-style multi-resolution encoder and RNN-T as later research upgrades.

Training from random initialisation is materially harder than using a large pretrained speech foundation model. The initial LibriSpeech read-speech corpus is a baseline; automotive domain accuracy needs additional permissioned English speech and human-corrected transcripts.

## ASR files committed

- `training/asr/config.yaml` — audio, model, optimisation and augmentation configuration.
- `training/asr/requirements.txt` — Python dependencies.
- `training/asr/prepare_librispeech.py` — download/extract LibriSpeech and write JSONL manifests.
- `training/asr/tokenizer.py` — text normalisation and SentencePiece wrapper.
- `training/asr/train_tokenizer.py` — train vocabulary from training transcripts only.
- `training/asr/data.py` — audio loading, resampling, log-Mel features, duration bucketing, collate and SpecAugment.
- `training/asr/model.py` — random-initialised FastConformer-inspired Conformer-CTC network.
- `training/asr/metrics.py` — greedy decoding, CTC prefix beam decoding, WER and CER.
- `training/asr/train.py` — mixed-precision training, validation, EMA, scheduler, periodic mid-epoch checkpoint and resume.
- `training/asr/evaluate.py` — held-out evaluation, test metrics and per-utterance CSV.
- `training/asr/plot_metrics.py` — training and test charts.
- `training/asr/inference.py` — load checkpoint and transcribe audio arrays.
- `training/asr/server.py` — temporary authenticated FastAPI inference service.
- `training/asr/smoke_test.py` — no-download forward/loss/backward smoke test.
- `training/asr/README.md` — Kaggle/Colab setup, commands and Cloudflare Tunnel instructions.

## How to run the first experiment

Run from the repository root in Kaggle or Colab with GPU enabled:

1. Install `training/asr/requirements.txt` after checking the notebook's CUDA-enabled PyTorch installation.
2. Prepare `train-clean-100`, `dev-clean` and `test-clean` manifests with `python -m training.asr.prepare_librispeech`.
3. Train SentencePiece from the training manifest with `python -m training.asr.train_tokenizer`.
4. Run `python -m training.asr.smoke_test`.
5. Run a small real-data check with `python -m training.asr.train --smoke`.
6. Run full training with `python -m training.asr.train`.
7. Evaluate with `python -m training.asr.evaluate`, then create charts with `python -m training.asr.plot_metrics`.
8. Start `python -m training.asr.server` with `ASR_API_KEY` and optionally expose it using the temporary Cloudflare Tunnel documented in the ASR README.

## Progress log

### 2026-11-21 — ASR-from-scratch implementation added

- Added a LibriSpeech downloader/extractor and manifest generator for train-clean-100, dev-clean and test-clean.
- Added SentencePiece tokenizer training from training transcripts only, reserving token ID 0 for the CTC blank symbol.
- Added 16 kHz mono loading, resampling, 80-bin log-Mel extraction, per-utterance normalisation, speed perturbation, duration-bucketed batches and SpecAugment.
- Added a random-initialised Conformer-CTC encoder using FastConformer-inspired 8x depthwise-separable subsampling, 9-frame convolution kernels, Macaron feed-forward blocks, rotary attention and PyTorch scaled-dot-product attention.
- Added configurable mid-epoch checkpoints and resume from the next saved batch to reduce lost work when free notebook sessions disconnect.
- Added CTC training with AdamW, warmup/cosine schedule, mixed precision, gradient clipping, EMA, best/last checkpoints and resume support.
- Added validation loss, WER/CER, held-out predictions, optional prefix beam decoding and charts.
- Added single-audio inference plus API-key-protected FastAPI endpoint for temporary remote hosting in a notebook runtime.
- Added notebook setup/instructions and ignored local datasets, recordings, checkpoints, tokenizers and experiment artifacts in Git.
- CI syntax compilation passed. The first CPU smoke-test attempt exposed a missing SentencePiece dependency in the workflow; that dependency has been added, and the rerun is pending. Code has **not yet been trained or benchmarked on a GPU in this session**. The first action is to run the smoke test and short real-data experiment in Kaggle/Colab and fix any runtime issues found.

## Next implementation and validation checklist

### Immediate: verify the ASR baseline
- [ ] Run the CPU model/data smoke test in GitHub Actions (rerun after adding the missing SentencePiece dependency).\n- [ ] Run the Python smoke test in a CUDA notebook.
- [ ] Prepare a small/short manifest and verify audio feature extraction.
- [ ] Train for one epoch on a limited sample.
- [ ] Confirm best.pt, last.pt, tokenizer.model and history.csv are created.
- [ ] Run held-out validation and inspect WER/CER and a sample of transcripts.
- [ ] Generate charts and review train/validation loss behaviour.
- [ ] Start the temporary inference server and verify authenticated upload transcription.
- [ ] Test the tunnel from the local laptop backend.

### ASR research improvements after the first baseline
- [ ] Establish full LibriSpeech 100-hour baseline metrics.
- [ ] Compare vocab size and BPE versus unigram tokenisation.
- [ ] Compare SpecAugment settings, speed perturbation and audio normalisation.
- [ ] Add a text-only n-gram language model trained from licensed text and evaluate rescoring separately.
- [ ] Reproduce or implement a Zipformer-style multi-resolution encoder from random initialisation.
- [ ] Add an RNN-T objective for streaming recognition after the CTC baseline is stable.
- [ ] Add consented company-domain English speech and measure automotive entity/number accuracy.
- [ ] Add speaker/accent/noise breakdowns and latency profiling.

### Later: car service application and TTS
- [ ] Build company-data import and local relational database.
- [ ] Add user-supplied reference value management and deterministic above/below/within comparison.
- [ ] Add customer/vehicle history, service entitlements, appointments and job cards.
- [ ] Add AI-generated structured request form with transcript evidence and supervisor approval.
- [ ] Train a separate English TTS model from random initialisation.
- [ ] Integrate the custom ASR/TTS endpoints with the local voice console.
- [ ] Compare our own voice pipeline with Vapi on matched test conversations.

## Risks and guardrails

- Free notebook GPU runtimes are limited and can disconnect. Save checkpoints to durable storage during training.
- The 100-hour LibriSpeech download requires several gigabytes of storage. More data and longer training will likely be needed for strong generalisation.
- Read audiobook speech is not representative of automotive conversations. Do not claim domain accuracy from LibriSpeech WER alone.
- WER/CER do not directly measure critical slot correctness. Add exact-match tests for registration numbers, vehicle/model names, dates, mileage, amounts and appointment times.
- The first model is a compact FastConformer-inspired Conformer-CTC baseline. A faithful Zipformer/RNN-T reproduction is a subsequent research milestone.
- Temporary inference tunnels are publicly reachable. Protect the endpoint with a strong API key, keep it out of frontend code, and never upload customer audio without permission.
- Model checkpoints, tokenizer, data, recordings, secrets and output charts stay outside Git unless synthetic/public data and licence rules explicitly permit sharing.
- Keep company vehicle data and user-supplied reference values as the authoritative source. The LLM must not invent reference limits or approve its own customer-request forms.

## Working rules for future changes

1. Inspect current files before changing them.
2. Preserve existing working code and features.
3. Make changes directly on `main`.
4. Add/update tests for each implemented feature.
5. Update this file after each meaningful milestone.
6. Never commit real customer data, recordings, secrets, model weights or downloaded datasets.
7. Do not mark a task complete unless it has actually been implemented and checked.
