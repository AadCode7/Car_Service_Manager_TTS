# English ASR From Scratch

This package trains an English automatic speech recognition model with **randomly initialised weights**. It does not download or fine-tune Whisper, wav2vec 2.0, HuBERT, NeMo checkpoints or any other pretrained speech recogniser.

The current baseline is a compact **FastConformer-inspired Conformer-CTC** model: 8x depthwise-separable convolutional subsampling, 9-frame Conformer convolution kernels, Macaron feed-forward blocks, rotary self-attention, PyTorch scaled-dot-product attention, SpecAugment, speed perturbation, duration-bucketed batches, AdamW, warmup/cosine scheduling, mixed precision, EMA weights and greedy/prefix-beam decoding.

It is research-informed and deliberately small enough to experiment with in Kaggle or Colab. It is **not a claim of matching large pretrained foundation ASR models or absolute state of the art**. Training from random initialisation needs much more labelled audio and compute than using a pretrained checkpoint.

## 1. Notebook preparation

Use a Kaggle notebook with GPU enabled or a Google Colab GPU runtime. The application can stay local on your laptop; the training and temporary inference server run in the notebook.

Clone the repository in a notebook, or upload the project folder. Run commands from the repository root.

    !git clone https://github.com/AadCode7/Car_Service_Manager_TTS.git
    %cd Car_Service_Manager_TTS

Check the accelerator:

    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

Install the ASR dependencies. Notebook images often already contain a CUDA-enabled PyTorch build. Check it before installing or replacing PyTorch.

    !pip install -r training/asr/requirements.txt

If PyTorch is missing or CPU-only, install the CUDA build matching the notebook image from https://pytorch.org/get-started/locally/.

## 2. Download and prepare LibriSpeech

The starter recipe uses:
- train-clean-100 for training (approximately 100 hours)
- dev-clean for validation
- test-clean for final evaluation

The first download is several gigabytes and needs enough notebook disk space. Enable internet access in Kaggle. The dataset is read English, so it is a starting point for acoustic-model training, not a substitute for company-domain car-service speech.

    !python -m training.asr.prepare_librispeech \
        --root data/librispeech \
        --manifest-dir data/manifests \
        --splits train-clean-100 dev-clean test-clean

The script downloads from OpenSLR, safely extracts regular files/directories and creates JSONL manifests containing audio path, transcript and duration. Re-running it skips already extracted splits.

Dataset reference and licence: https://openslr.org/12/ . Check attribution and licence requirements before redistributing the dataset or model.

## 3. Train the tokenizer

SentencePiece is trained only on the training transcripts. It uses a 500-piece unigram vocabulary by default. Token ID 0 is reserved for the CTC blank.

    !python -m training.asr.train_tokenizer \
        --manifest data/manifests/train-clean-100.jsonl \
        --output artifacts/asr/tokenizer.model \
        --vocab-size 500 \
        --model-type unigram

Do not train the tokenizer on dev/test transcripts. That would leak evaluation text into model preparation.

## 4. Run a smoke test before a long training job

First verify that the neural network, CTC objective, data pipeline and gradients work:

    !python -m training.asr.smoke_test

Then run a short real-data experiment. This reads a small subset from the manifest and trains a small two-layer model for one epoch:

    !python -m training.asr.train --smoke

The smoke run writes to artifacts/asr-smoke by default, separate from the full model checkpoint directory. It is only a pipeline test; do not interpret its WER as a meaningful model result.

## 5. Train the full baseline

    !python -m training.asr.train

The current model uses padding-aware LayerNorm in its convolutional frontend and Conformer convolution modules. The smoke test also checks that valid-frame logits do not change when an utterance is padded and batched with a longer recording.

**Important when updating from the earlier BatchNorm model:** the model architecture and checkpoint state dictionary have changed. Start a fresh run without `--resume`. Back up the old `artifacts/asr` directory first if you need to preserve its checkpoints and metrics. Old checkpoints are intentionally rejected with an explicit compatibility error; they cannot be loaded into this architecture.

Main settings are in training/asr/config.yaml:
- model: hidden dimension, attention heads, number of Conformer blocks and convolution kernel
- data: manifests, tokenizer path and worker count
- training: duration budget, AMP mode, SpecAugment, speed perturbation, EMA, learning rate and early stopping

The default model has 10 Conformer blocks, 192 hidden dimensions, 6 attention heads, 128 subsampling channels and an 8x temporal frontend. This is a smaller resource-conscious configuration than NVIDIA's documented 256-channel FastConformer recipe. If the notebook runs out of GPU memory, lower max_batch_seconds, max_batch_size or d_model. If training is stable and GPU memory permits, increase the duration budget or model size gradually.

For a quick controlled subset without changing the YAML:

    !python -m training.asr.train \
        --epochs 2 \
        --max-train-samples 2000 \
        --max-valid-samples 300

Resume from artifacts/asr/last.pt:

    !python -m training.asr.train --resume

Checkpoints are written every 300 optimiser updates (configurable) and at epoch boundaries. Resume continues from the next saved batch when a mid-epoch checkpoint exists:
- artifacts/asr/last.pt: full training state for resume, including mid-epoch batch position
- artifacts/asr/best.pt: best validation-WER weights for inference/evaluation
- artifacts/asr/tokenizer.model: exact tokenizer used by the model
- artifacts/asr/history.csv and metrics.jsonl: training history
- artifacts/asr/model_config.json: model and audio configuration

Save checkpoints to durable notebook storage or download/copy them before the runtime disconnects. Ephemeral notebook files can be lost when a session ends.

## 6. Evaluate the held-out test split

Do not use test-clean to select hyperparameters or stop training. Use dev-clean for validation and only run test evaluation after choosing the checkpoint.

    !python -m training.asr.evaluate \
        --manifest data/manifests/test-clean.jsonl \
        --run-dir artifacts/asr

Greedy decoding is the default. To compare a small CTC prefix beam decoder (without an external language model):

    !python -m training.asr.evaluate \
        --manifest data/manifests/test-clean.jsonl \
        --run-dir artifacts/asr \
        --beam-width 8

Outputs:
- test_metrics.json: aggregate loss, WER, CER and counts
- test_predictions.csv: one row per utterance with reference, hypothesis and utterance-level WER/CER

WER and CER are corpus-level edit-distance rates. They may exceed 1.0 when a hypothesis contains many insertions. The per-utterance charts clip WER at 2 only for visual readability; the CSV preserves the actual value.

## 7. Generate charts

    !python -m training.asr.plot_metrics --run-dir artifacts/asr

Charts are saved in artifacts/asr/charts:
- loss_curves.png
- error_rates.png
- learning_rate.png
- epoch_time.png
- utterance_wer_distribution.png (after test evaluation)
- wer_vs_duration.png (after test evaluation)

## 8. Test one audio file

After training, test a local or notebook audio file:

    import soundfile as sf
    from training.asr.inference import load_asr_model, transcribe_audio_array

    bundle = load_asr_model("artifacts/asr")
    audio, sample_rate = sf.read("sample.wav", dtype="float32", always_2d=True)
    result = transcribe_audio_array(audio, sample_rate, bundle)
    print(result)

The result contains text, audio duration, inference time and the training epoch of the loaded checkpoint.

## 9. Expose a temporary inference endpoint

This starts an authenticated FastAPI endpoint inside the notebook runtime. It does not deploy the main application.

Create a strong temporary key:

    import os, secrets
    ASR_API_KEY = secrets.token_urlsafe(32)
    os.environ["ASR_API_KEY"] = ASR_API_KEY
    os.environ["ASR_ARTIFACT_DIR"] = "artifacts/asr"

Start the server in a notebook background process:

    !nohup python -m training.asr.server > /tmp/asr_server.log 2>&1 &

Install Cloudflare Tunnel for a temporary public HTTPS URL (Colab/Linux x86_64):

    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
    !chmod +x /usr/local/bin/cloudflared
    !nohup cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate > /tmp/cloudflared.log 2>&1 &

Read the generated trycloudflare.com URL:

    import re, time
    time.sleep(5)
    tunnel_log = open("/tmp/cloudflared.log").read()
    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", tunnel_log)
    print(match.group(0) if match else tunnel_log[-2000:])

Kaggle and Colab network policies can change. If a notebook provider blocks inbound tunnel traffic, keep inference inside the notebook or use a temporary endpoint option supported by that runtime. Do not assume the tunnel will survive a notebook restart.

Test the endpoint:

    import requests
    endpoint = "https://YOUR-URL.trycloudflare.com/v1/transcribe"
    with open("sample.wav", "rb") as audio_file:
        response = requests.post(
            endpoint,
            headers={"X-API-Key": ASR_API_KEY},
            files={"audio": ("sample.wav", audio_file, "audio/wav")},
            timeout=180,
        )
    response.raise_for_status()
    print(response.json())

The server also exposes GET /health. The transcription endpoint requires X-API-Key. Keep the generated URL and key private, stop the tunnel when finished, and never send real customer audio to a public test endpoint without the company's permission and an agreed retention policy.

## 10. Connect the local app later

The local FastAPI application will call the temporary endpoint with:
- URL: the generated base URL plus /v1/transcribe
- multipart field: audio
- header: X-API-Key

Keep the endpoint URL and key in the local backend environment, never in React/browser code. When the notebook disconnects, the endpoint disappears. A stable service requires a separately managed remote inference host.

## 11. Research roadmap

This baseline is intentionally built to make later experiments measurable. Candidate upgrades, in order:

1. Compare character/BPE vocabularies, data augmentation and feature normalisation.
2. Train with additional clean and more varied English corpora, keeping speaker-disjoint evaluation.
3. Add a carefully evaluated external n-gram language model trained from text, not a pretrained ASR checkpoint.
4. Implement a Zipformer-style multi-resolution encoder or reproduce the published Zipformer recipe with its training toolkit, still initialising all weights randomly.
5. Add an RNN-T objective for streaming/online recognition once the CTC baseline is stable.
6. Add domain adaptation using company-approved English automotive recordings and human-corrected transcripts.
7. Profile streaming chunking and latency, and evaluate exact-match accuracy for registration numbers, dates, mileage, amounts and model names.

NVIDIA's FastConformer recipe uses 8x depthwise convolutional subsampling and 9-frame Conformer convolution kernels to reduce encoder compute. This implementation now borrows those two efficiency choices, plus Macaron blocks, rotary attention and PyTorch SDPA, but it is a compact custom model rather than a reproduction of NVIDIA's full FastConformer implementation. Zipformer, published at ICLR 2024, uses a different multi-rate U-Net-like encoder, weight-sharing attention, BiasNorm and Swoosh activations; reproducing that architecture is a later research experiment.

Research references:
- Zipformer paper: https://proceedings.iclr.cc/paper_files/paper/2024/hash/c1bb0e3b062f0a443f2cc8a4ec4bb30d-Abstract-Conference.html
- NVIDIA NeMo ASR model architecture guide: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html
- SpeechBrain ASR-from-scratch tutorial: https://speechbrain.readthedocs.io/en/stable/tutorials/tasks/speech-recognition-from-scratch.html
