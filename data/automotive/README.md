# Automotive English Speech Dataset

This folder is the collection area for real, consented recordings used to adapt the Wav2Vec2 CTC model to car-service conversations.

## What to collect

Start with a pilot of 5–10 speakers and roughly 50–100 utterances per speaker. Use several speakers with different Indian English accents, genders, speaking rates and levels of familiarity with cars. Record with more than one phone or microphone. Include quiet recordings and realistic workshop ambience, but keep speech intelligible.

Do not record real customer calls or identifiable customer information for this pilot. Use fictional names, registration numbers and vehicle details. Obtain explicit informed consent from every speaker before recording. Store consent records separately from the audio dataset.

## Recording rules

- Record one utterance per WAV file.
- Prefer mono PCM WAV at 16 kHz or 48 kHz. The loader converts audio to 16 kHz.
- Keep utterances between 0.3 and 20 seconds.
- Leave a short silence before and after speech. Do not trim consonants.
- Avoid clipping, music, overlapping speakers and aggressive noise suppression.
- Capture a mix of phone microphones and laptop/headset microphones.
- Record some sessions in a quiet room and some with controlled workshop-like background noise.
- Do not use generated TTS as a substitute for real-speaker recordings.
- Use a stable anonymous speaker ID such as spk_001; do not put a person's name in the ID.
- Have a human verify every transcript against the recording.

## Transcript convention

The model's existing 39-symbol tokenizer supports A–Z, apostrophe, spaces and digits 0–9. It does not emit punctuation or symbols such as hyphen, slash, decimal point, rupee sign or colon.

Write transcripts in uppercase or mixed case; the preparation script normalizes them to uppercase. Use a single space between words. Use apostrophes only when spoken.

For numeric entities, use a consistent canonical transcript:
- Spoken “forty-two thousand five hundred kilometres” → 42500 KILOMETRES
- Spoken “thirty-two PSI” → 32 PSI
- Spoken “M H fifteen A B four two one zero” → MH 15 AB 4210
- Spoken “twenty twenty-four” as a model year → 2024

This is intentional normalization, not strict verbatim transcription. Keep a separate annotation log if you need the exact spoken wording for later research. Do not alternate between FORTY TWO and 42 for the same kind of numeric entity.

## Metadata CSV

Copy metadata_template.csv, fill one row per WAV file, and save it as metadata.csv.

Required columns:
- audio_filepath: absolute path or path relative to the CSV file.
- text: human-verified canonical transcript.
- speaker_id: stable anonymous ID.
- consent: yes only after explicit consent has been obtained.
- transcript_verified: yes only after a human has checked the audio.

Optional columns:
- session_id, scenario, noise_condition, microphone, vehicle_context, notes.

Never set transcript_verified=yes automatically. The script rejects rows without confirmed consent and human transcript verification.

## Generate speaker-disjoint manifests

From the repository root:

    !python -m training.asr.prepare_automotive \
        --csv data/automotive/metadata.csv \
        --output-dir data/automotive/manifests \
        --seed 42

The script validates the WAV files, duration, sample rate, transcript and metadata. It rejects invalid rows and writes:
- train.jsonl
- valid.jsonl
- test.jsonl
- summary.json

A speaker is assigned to exactly one split. This is important: if the same speaker appears in both training and test, the evaluation can overstate generalisation. At least three distinct speakers are required; more speakers are strongly preferred. Inspect summary.json and the rejected-row report before training.

## Fine-tune from the existing LibriSpeech checkpoint

First ensure the existing checkpoint is present:

    artifacts/asr-transfer/best.pt

Then run a small smoke test using the automotive manifests:

    !python -m training.asr.train \
        --config training/asr/automotive_config.yaml \
        --output-dir artifacts/asr-automotive-smoke \
        --init-checkpoint artifacts/asr-transfer/best.pt \
        --smoke \
        --max-train-samples 64 \
        --max-valid-samples 32

If the smoke test completes, start the real domain-adaptation run:

    !python -m training.asr.train \
        --config training/asr/automotive_config.yaml \
        --output-dir artifacts/asr-automotive \
        --init-checkpoint artifacts/asr-transfer/best.pt

The --init-checkpoint option loads the complete Wav2Vec2 encoder and existing CTC head weights, but starts a fresh optimizer, epoch counter and validation history. It is different from --resume, which continues an interrupted run from last.pt. Do not use --resume for the first automotive adaptation run.

The automotive config uses a conservative encoder learning rate, trains the upper four Transformer layers plus CTC head, and keeps the feature extractor and lower eight Transformer layers frozen. It retains the same tokenizer, so the checkpoint remains compatible. The original artifacts/asr-transfer and Conformer artifacts/asr folders are not overwritten.

## Evaluate correctly

Use valid.jsonl to select checkpoints and tune settings. Keep test.jsonl untouched until the pipeline and checkpoint are selected:

    !python -m training.asr.evaluate \
        --manifest data/automotive/manifests/test.jsonl \
        --run-dir artifacts/asr-automotive

Report WER and CER, but also calculate exact match for high-value fields such as registration number, mileage, phone number and amount. A low overall WER does not guarantee that a registration plate or price is correct.

Do not use the LibriSpeech test score as the automotive acceptance criterion. The final product needs a separate, speaker-disjoint automotive test set that reflects the intended microphones and workshop conditions.
