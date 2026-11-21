# Project Progress

**Project:** Car Service Manager with Voice Agent  
**Repository:** https://github.com/AadCode7/Car_Service_Manager_TTS  
**Working branch:** `main`  
**Last updated:** 2026-11-21

## Project objective

Build a local employee-facing car service management application with a voice agent. The frontend, backend and operational database run on the laptop. The speech models are trained and served remotely because the laptop's GTX 1650 is not used for model inference or training.

The main research objective is to build our own English speech recognition (ASR) and speech synthesis (TTS) models with weights trained from random initialisation. Groq is reserved for conversational LLM reasoning, not speech recognition or synthesis.

The application reads accurate customer, vehicle and service records already held by the company. Reference/optimal values are provided by the user and stored in the system. The AI explains records and prepares structured customer-request drafts. A supervisor approves drafts before they change official records or work orders.

## Confirmed scope and constraints

- English only.
- Local React/TypeScript + FastAPI application; no public application deployment required.
- No local ASR, TTS or LLM inference. Use remote/cloud compute for speech model training and inference.
- Do not use pretrained ASR/TTS checkpoints or fine-tune existing speech models as a substitute for the main objective. SpeechBrain/ESPnet may be used as toolkits, with random-initialised model weights.
- Groq API may be used for the conversation LLM and tool selection.
- Local browser microphone and speakers for simulated voice conversations; no real phone calls in the initial project.
- Vapi is optional for comparison; it is not a core dependency.
- Company records are the source of truth for customers, vehicles and service history.
- The user supplies the reference/optimal vehicle values; comparisons must be deterministic and based on stored values.
- AI-generated customer-request details are editable drafts and require supervisor approval before official records change.
- The project calculates estimates and payment breakdowns only. No payment gateway, payment requests or actual transactions.
- Use synthetic test records until privacy, consent, access and backup controls are ready.
- Free cloud GPU quotas and always-on model hosting are not guaranteed. A temporary notebook endpoint is acceptable during experiments.
- Work directly on `main`; do not create a feature branch unless explicitly requested.

## Current repository state

- [x] Confirmed repository exists and is writable.
- [x] Confirmed default branch is `main`.
- [x] Confirmed repository was empty at project start.
- [x] Added `architecture.md`.
- [x] Added `progress.md`.
- [x] Updated the planning documents to reflect the user's confirmed scope.
- [ ] Create the initial application scaffold.

## Architecture decisions

| Area | Decision | Status |
|---|---|---|
| Frontend | React + TypeScript + Vite + Tailwind | Proposed |
| Backend | Python + FastAPI | Proposed |
| Database | SQLite + SQLAlchemy + Alembic | Proposed |
| Conversation LLM | Groq API behind an adapter | Proposed |
| ASR | Train an English model from random initialisation | Confirmed objective |
| TTS | Train an English model from random initialisation | Confirmed objective |
| Training toolkit | PyTorch + SpeechBrain or ESPnet, without pretrained speech checkpoints | Proposed |
| Speech model compute | Remote cloud GPU sessions | Required by hardware constraint |
| Local voice interface | Browser microphone + speakers | Confirmed |
| Vapi | Optional browser voice benchmark; not core | Optional |
| Customer-request workflow | AI draft form followed by supervisor approval | Confirmed |
| Reference values | Supplied by the user and stored in the application | Confirmed |
| Payment | Estimate and line-item breakdown only | Confirmed |
| Deployment | Application local; temporary remote model endpoint for testing | Confirmed direction |

## Progress log

### 2026-11-21 — Initial planning and scope clarification

- Inspected the GitHub repository and confirmed it was empty with `main` as the default branch.
- Created `architecture.md` and `progress.md) directly on `main`.
- Defined the product as an employee operations system with a voice interface, not simply a voice chatbot.
- Proposed a local React + FastAPI + SQLite architecture.
- User confirmed English-only speech and local laptop microphone/speaker testing.
- User confirmed company-held vehicle records are the factual source and will supply the optimal/reference values.
- User clarified that training the speech model itself is the main project objective. The plan now requires random-initialised ASR and TTS weights; using pretrained speech checkpoints or merely fine-tuning them does not meet this objective.
- Reserved Groq for conversational reasoning; it is not the project's ASR/TTS model.
- Defined AI-generated customer-request details as an editable draft requiring supervisor approval before official data changes.
- Restricted payment scope to estimate/payment-breakdown calculations; no payment provider or real transaction flow.
- Researched Vapi's browser voice SDK, custom transcriber/TTS interfaces, proprietary orchestration layer and current usage pricing. Vapi is an optional comparison, not the primary architecture.
- Documented that free cloud GPU quotas and always-on remote inference cannot be assumed.
- Updated the staged delivery plan to prioritise ASR training from scratch, then TTS training from scratch.

## Implementation checklist

### Phase 0 — Data and environment readiness
- [ ] Obtain a synthetic/sample export format for customers, vehicles and service history.
- [ ] Obtain the reference-value table schema, including units, thresholds and vehicle applicability.
- [ ] Confirm the desired English accent/voice for the TTS model.
- [ ] Identify a permitted speech dataset and review its licence/attribution obligations.
- [ ] Confirm available cloud GPU notebook/runtime limits.
- [ ] Confirm local Python and Node.js versions.

### Phase 1 — Local application and structured data
- [ ] Create the project scaffold without overwriting files.
- [ ] Add safe configuration loading and `.env.example`.
- [ ] Add SQLite database and migrations.
- [ ] Add synthetic seed data.
- [ ] Build the employee dashboard.
- [ ] Build customer and vehicle search/profile.
- [ ] Display service history and service-plan entitlements.
- [ ] Store user-supplied reference values and implement deterministic above/below/within comparisons.
- [ ] Add backend and frontend tests.

### Phase 2 — ASR model from scratch
- [ ] Prepare a licensed English speech corpus and data manifests.
- [ ] Keep training, validation and test splits speaker-disjoint.
- [ ] Implement a compact random-initialised ASR architecture and training loop.
- [ ] Train a small baseline on cloud GPU compute.
- [ ] Measure WER/CER and exact-match accuracy for registration, model, date, mileage and price fields.
- [ ] Add domain vocabulary and consented company speech data when available.
- [ ] Serve the trained model through a temporary remote inference endpoint.

### Phase 3 — Voice loop and conversation tools
- [ ] Capture microphone audio in the browser.
- [ ] Stream audio to our custom ASR endpoint.
- [ ] Connect transcripts to the Groq conversation LLM.
- [ ] Add typed backend tools for reading records, service eligibility, diagnostic comparisons and appointment slots.
- [ ] Display the transcript and retrieved context in the voice console.
- [ ] Handle correction, uncertainty, timeout and manual fallback.
- [ ] Test complete service conversations using synthetic customer data.

### Phase 4 — Customer-request form and supervisor approval
- [ ] Extract customer requests into a structured draft form.
- [ ] Link material fields to transcript evidence.
- [ ] Flag uncertain or missing fields.
- [ ] Allow a supervisor to edit, approve or reject.
- [ ] Record approver, timestamp, decision and changes.
- [ ] Prevent unapproved drafts from changing official records or work orders.

### Phase 5 — TTS model from scratch
- [ ] Confirm the English speaker/voice and recording permission.
- [ ] Prepare clean speech recordings with exact transcripts.
- [ ] Implement a random-initialised TTS architecture and training loop.
- [ ] Train on remote GPU compute.
- [ ] Evaluate intelligibility, naturalness, pronunciation and generation latency.
- [ ] Serve the custom model remotely and connect browser playback.

### Phase 6 — Documents and estimate calculations
- [ ] Generate job cards, inspection sheets, estimates and invoices.
- [ ] Calculate labour, parts, taxes, service-plan coverage and totals.
- [ ] Show the payment breakdown without creating a payment request or collecting money.
- [ ] Add document backup and export.

### Phase 7 — End-to-end evaluation
- [ ] Test service reminder conversations.
- [ ] Test service history and free-service explanation.
- [ ] Test customer request extraction and supervisor approval.
- [ ] Test diagnostic value explanations against user-supplied references.
- [ ] Test estimate breakdowns and collection-ready notifications.
- [ ] Track ASR, TTS, latency and workflow regression metrics.

### Phase 8 — Optional Vapi benchmark
- [ ] Configure Vapi Web SDK for browser voice tests.
- [ ] Compare matched scenarios against our own pipeline.
- [ ] Compare latency, interruptions, accuracy, audio quality and cost.
- [ ] Explore custom ASR/TTS integration only if it benefits the benchmark.
- [ ] Keep Vapi optional and outside the core architecture.

## Known risks and open issues

- Training ASR from random initialisation requires substantial labelled English audio. Public datasets can bootstrap research experiments, but read speech alone will not represent real service conversations.
- ASR and TTS are separate research projects. Start with ASR and build TTS after the training/evaluation pipeline is proven.
- TTS from random initialisation requires a clean, consistent speaker recording corpus with exact transcripts.
- A GTX 1650 is not suitable for the intended remote-model approach. Free cloud GPU sessions can be limited, pre-emptible or unavailable; large training jobs and always-on low-latency inference may require paid compute.
- A trained model must be exposed through a reachable inference endpoint for the local application. Temporary notebook tunnels are useful for testing but not reliable production hosting.
- Accuracy for registration numbers, model names, dates, mileage, quantities and prices must be measured separately from overall WER.
- Reference values must be applicable to the exact car/variant, units and test conditions. The application must show unavailable when a matching reference is missing.
- The LLM must not approve its own extracted request. Supervisor approval is mandatory before official workflow changes.
- Local databases, transcripts and any audio files need access control, consent, retention and backup policies.
- Vapi is not unlimited free: current pricing has a usage-based hosting charge and separately billed model costs. Its managed real-time orchestration is proprietary.

## Working rules for future changes

1. Inspect the current files before changing them.
2. Preserve existing working code and features.
3. Make changes directly on `main`.
4. Add or update tests for each implemented feature.
5. Update this file after each meaningful milestone, including what changed and what remains.
6. Do not add real customer data, API keys, recordings or generated customer documents to Git.
7. Do not mark a task complete unless it has actually been implemented and checked.
