# Car Service Manager with Speech AI

A local car-servicing operations application with a voice assistant. The longer-term product includes customer and vehicle history, service scheduling, job cards, diagnostic comparisons, AI-generated customer-request forms and supervisor approvals.

The primary research objective is to build our own English speech models from random initialisation. We are not using pretrained Whisper/wav2vec checkpoints or fine-tuning an existing speech model as a substitute.

## Current milestone: English ASR from scratch

The first ASR training pipeline is committed under [training/asr/](training/asr/README.md).

It currently contains:
- LibriSpeech download, safe extraction and JSONL manifest creation.
- SentencePiece tokenizer training using training transcripts only.
- 16 kHz audio resampling and 80-bin log-Mel feature extraction.
- Duration-bucketed batches, speed perturbation and SpecAugment.
- Random-initialised FastConformer-inspired Conformer-CTC acoustic model.
- CTC training, AdamW, warmup/cosine schedule, mixed precision, gradient clipping and EMA.
- Periodic mid-epoch checkpoints and resume.
- Validation/test loss, WER/CER, greedy and CTC prefix-beam decoding.
- CSV prediction exports and training/evaluation charts.
- Single-file inference and an API-key-protected temporary FastAPI endpoint.

**Architecture note:** the current encoder uses 8x depthwise-separable convolutional subsampling and 9-frame Conformer convolution kernels, drawing on FastConformer efficiency ideas. It is a compact custom model, not an exact FastConformer or Zipformer reproduction and not a claim of state-of-the-art benchmark accuracy. A Zipformer-style multi-rate encoder and RNN-T objective are later research experiments.

## Hardware and execution

- The employee application is intended to run locally on your laptop.
- Do not run ASR/LLM/TTS inference on the GTX 1650.
- Train and serve the custom speech model in a Kaggle or Colab GPU runtime.
- A temporary tunnel can expose the notebook's inference API while that runtime remains active. The tunnel URL stops working when the runtime disconnects.
- Free GPU quotas and always-on free inference hosting are not guaranteed.

## Quick start in Kaggle or Colab

Clone the repository and run commands from its root:

    !git clone https://github.com/AadCode7/Car_Service_Manager_TTS.git
    %cd Car_Service_Manager_TTS

Check that the notebook has a CUDA-enabled PyTorch build before installing dependencies:

    import torch
    print(torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

Install the remaining dependencies. PyTorch is deliberately not pinned in requirements.txt, because Kaggle/Colab images have different CUDA builds. Install a matching PyTorch build only if the notebook does not already have one.

    !pip install -r training/asr/requirements.txt

Prepare LibriSpeech training, validation and test manifests:

    !python -m training.asr.prepare_librispeech \
        --root data/librispeech \
        --manifest-dir data/manifests \
        --splits train-clean-100 dev-clean test-clean

Train the tokenizer on training transcripts only:

    !python -m training.asr.train_tokenizer \
        --manifest data/manifests/train-clean-100.jsonl \
        --output artifacts/asr/tokenizer.model \
        --vocab-size 500

Run a quick pipeline check:

    !python -m training.asr.smoke_test
    !python -m training.asr.train --smoke

The short training smoke run saves under artifacts/asr-smoke/, so it cannot overwrite the full training checkpoint.

Start the configurable full run:

    !python -m training.asr.train

Evaluate the held-out test split and produce charts:

    !python -m training.asr.evaluate --manifest data/manifests/test-clean.jsonl
    !python -m training.asr.plot_metrics

For the complete workflow, options, resume instructions and temporary endpoint setup, read [training/asr/README.md](training/asr/README.md).

## Current validation status

- Python modules compile in GitHub Actions.
- The CPU smoke test passes: feature extraction, duration sampler, model forward pass, CTC loss and backward gradients.
- No CUDA training run or LibriSpeech WER benchmark has been run yet. Run these in Kaggle/Colab before making any accuracy claims.

## Project documentation

- [Architecture and phased implementation plan](architecture.md)
- [Progress tracker and validation checklist](progress.md)
- [ASR implementation and notebook guide](training/asr/README.md)

## Data and security

Downloaded audio, manifests, customer records, recordings, checkpoints, tokenizers, generated charts, credentials and temporary logs are ignored by Git. Do not commit company/customer data or recordings. Obtain permission before using company calls as training data. Keep the temporary inference API key private and never expose it in frontend code.
