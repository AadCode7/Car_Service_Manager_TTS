# English ASR with Wav2Vec 2.0 Transfer Learning

The default training backend fine-tunes **torchaudio WAV2VEC2_BASE**, a 95M-parameter wav2vec 2.0 encoder pretrained self-supervised on 960 hours of LibriSpeech audio. It replaces the original randomly initialised Conformer encoder. The transfer path uses a 39-symbol character vocabulary (blank, word separator, A–Z, apostrophe and digits 0–9) and a newly trained CTC head. It is separate from the original 500-piece SentencePiece Conformer baseline. Epoch 1 trains only the new CTC head; later epochs fine-tune the upper four Transformer layers while keeping the feature extractor and lower eight layers frozen. Separate learning rates protect the pretrained representation.

Transfer learning is selected by `model.backend: wav2vec2` in the config. Set it to `conformer` to run the original compact Conformer-CTC baseline.

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

SentencePiece is used by the original Conformer baseline only. It is trained only on training transcripts and uses a 500-piece unigram vocabulary by default. The Wav2Vec2 transfer path creates its own 39-symbol character vocabulary and does not require this step. Token ID 0 is reserved for the CTC blank in both paths.

    !python -m training.asr.train_tokenizer \
        --manifest data/manifests/train-clean-100.jsonl \
        --output artifacts/asr/tokenizer.model \
        --vocab-size 500 \
        --model-type unigram

Do not train the tokenizer on dev/test transcripts. That would leak evaluation text into model preparation.

## 4. Run a smoke test before a long training job

Run a short transfer-learning smoke experiment first. It downloads the same pretrained encoder, trains on a small manifest subset and writes to a separate directory:

    !python -m training.asr.train --smoke

This is a pipeline check, not a meaningful WER benchmark. The older `python -m training.asr.smoke_test` command tests the Conformer baseline, not the transfer-learning backend.

## 5. Train with pretrained wav2vec 2.0

    !python -m training.asr.train

The first run downloads the official torchaudio `WAV2VEC2_BASE` checkpoint (approximately 360 MB) and caches it in the notebook environment. Kaggle internet access must be enabled. Install a torchaudio build matching the PyTorch version already installed in the Kaggle image; do not blindly upgrade PyTorch or install a mismatched torchaudio wheel. Check compatibility first:

    import torch, torchaudio
    print(torch.__version__, torchaudio.__version__)
    print(torchaudio.pipelines.WAV2VEC2_BASE.sample_rate)

The model expects 16 kHz raw waveforms. The transfer-learning dataset therefore reads audio directly rather than using the old log-Mel frontend. The CTC head is newly initialised for the 39-symbol character vocabulary; the acoustic encoder starts from pretrained weights. The default T4 recipe trains the CTC head alone for the first epoch, then fine-tunes the upper four Transformer layers at 1e-5 while keeping the feature extractor and lower eight layers frozen. The CTC head uses a 3e-4 learning rate. It uses smaller duration-based batches to control attention memory.

The original Conformer backend remains in the repository. To select it, change `model.backend` to `conformer`; its checkpoints are not compatible with wav2vec2 checkpoints.

**Important:** transfer checkpoints use a different model and tokenizer from the previous Conformer run. They are written to `artifacts/asr-transfer`, so the old `artifacts/asr` checkpoints are left untouched. The transfer trainer supports `--resume` from `artifacts/asr-transfer/last.pt`. If you ran an earlier transfer experiment before the numeric symbols were added to the vocabulary, move or rename its `artifacts/asr-transfer` directory before starting this version; the trainer checks for a vocabulary-size mismatch when resuming.

Main transfer settings are in training/asr/config.yaml:
- `model.freeze_encoder_layers`: number of lower Transformer layers to keep frozen after head warm-up
- `training.transfer_epochs`: maximum fine-tuning epochs
- `training.transfer_encoder_lr` and `training.transfer_head_lr`: separate learning rates
- `training.transfer_max_batch_seconds` and `training.transfer_max_batch_size`: T4 memory controls
- `training.transfer_patience`: early-stopping patience

The default transfer model is wav2vec2-base (12 Transformer layers, 768-dimensional representations). If the notebook runs out of GPU memory, lower `transfer_max_batch_seconds` or `transfer_max_batch_size`.

For a quick controlled subset without changing the YAML:

    !python -m training.asr.train \
        --epochs 2 \
        --max-train-samples 2000 \
        --max-valid-samples 300

Resume from the most recent completed epoch:

    !python -m training.asr.train --resume

Transfer checkpoints are saved at epoch boundaries:
- `artifacts/asr-transfer/last.pt`: model, optimizer, scaler and training history for resume
- `artifacts/asr-transfer/best.pt`: best validation-WER weights
- `artifacts/asr-transfer/tokenizer.json`: exact character vocabulary
- `artifacts/asr-transfer/history.csv`: epoch metrics
- `artifacts/asr-transfer/model_config.json`: model metadata

Save checkpoints to durable notebook storage or download/copy them before the runtime disconnects. Ephemeral notebook files can be lost when a session ends.

## 6. Evaluate the held-out test split

Do not use test-clean to select hyperparameters or stop training. Use dev-clean for validation and only run test evaluation after choosing the checkpoint.

**Evaluation caveat:** torchaudio's self-supervised `WAV2VEC2_BASE` checkpoint was pretrained on unlabelled audio from LibriSpeech train-clean-100, train-clean-360 and train-other-500. Your custom validation and test splits were carved from train-clean-100, so their waveforms may already have appeared during pretraining even though their transcripts were not used. Treat these scores as development results, not a fully independent benchmark. For an unbiased generalisation estimate, also evaluate on a separate corpus that was not part of LibriSpeech pretraining.

    !python -m training.asr.evaluate \
        --manifest data/manifests/test-clean.jsonl \
        --run-dir artifacts/asr-transfer

Greedy decoding is the default. To compare a small CTC prefix beam decoder (without an external language model):

    !python -m training.asr.evaluate \
        --manifest data/manifests/test-clean.jsonl \
        --run-dir artifacts/asr-transfer \
        --beam-width 8

Outputs:
- test_metrics.json: aggregate loss, WER, CER and counts
- test_predictions.csv: one row per utterance with reference, hypothesis and utterance-level WER/CER

WER and CER are corpus-level edit-distance rates. They may exceed 1.0 when a hypothesis contains many insertions. The per-utterance charts clip WER at 2 only for visual readability; the CSV preserves the actual value.

## 7. Generate charts

    !python -m training.asr.plot_metrics --run-dir artifacts/asr-transfer

Charts are saved in artifacts/asr-transfer/charts:
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

    bundle = load_asr_model("artifacts/asr-transfer")
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
    os.environ["ASR_ARTIFACT_DIR"] = "artifacts/asr-transfer"

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

The transfer-learning baseline is intended to make later experiments measurable. Candidate upgrades, in order:

1. Compare character/BPE vocabularies, data augmentation and feature normalisation.
2. Train with additional clean and more varied English corpora, keeping speaker-disjoint evaluation.
3. Add a carefully evaluated external n-gram language model trained from text, not a pretrained ASR checkpoint.
4. Compare the pretrained Wav2Vec2 encoder with a Zipformer-style multi-resolution encoder using the same data splits and decoding setup.
5. Add an RNN-T objective for streaming/online recognition once the CTC baseline is stable.
6. Add domain adaptation using company-approved English automotive recordings and human-corrected transcripts.
7. Profile streaming chunking and latency, and evaluate exact-match accuracy for registration numbers, dates, mileage, amounts and model names.

NVIDIA's FastConformer recipe uses 8x depthwise convolutional subsampling and 9-frame Conformer convolution kernels to reduce encoder compute. This implementation now borrows those two efficiency choices, plus Macaron blocks, rotary attention and PyTorch SDPA, but it is a compact custom model rather than a reproduction of NVIDIA's full FastConformer implementation. Zipformer, published at ICLR 2024, uses a different multi-rate U-Net-like encoder, weight-sharing attention, BiasNorm and Swoosh activations; reproducing that architecture is a later research experiment.

Research references:
- Zipformer paper: https://proceedings.iclr.cc/paper_files/paper/2024/hash/c1bb0e3b062f0a443f2cc8a4ec4bb30d-Abstract-Conference.html
- NVIDIA NeMo ASR model architecture guide: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html
- SpeechBrain ASR-from-scratch tutorial: https://speechbrain.readthedocs.io/en/stable/tutorials/tasks/speech-recognition-from-scratch.html
