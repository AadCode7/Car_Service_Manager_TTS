# Car Service Manager with Voice Agent

**Project status:** Planning / Phase 0 — clarified scope, speech model research is the primary objective  
**Repository:** `AadCode7/Car_Service_Manager_TTS`  
**Target execution:** Local web application on the employee's laptop; AI inference through external APIs  
**Hardware constraint:** NVIDIA GTX 1650 laptop. No local LLM, ASR or TTS inference.

> The application is the system of record. The speech model is an interface to that system, not the source of truth for customer, vehicle, service, diagnostic or payment data.

## 1. Product goal

Build a service-centre operations application with an employee dashboard and a voice agent that can handle customer conversations using the correct customer and vehicle records.

The application should cover the full service journey:

1. Identify a customer and their vehicle using a verified identifier.
2. Check service due dates, mileage-based intervals, free-service entitlements and service history.
3. Contact the customer, explain why service is due, explain what is included and offer appointment slots.
4. Record the customer's choice and book only after confirmation.
5. Check the vehicle in, capture its condition and odometer, and create a job card.
6. Record inspections and diagnostic readings, comparing them with the correct manufacturer or approved reference specifications.
7. Prepare an estimate, explain recommended work and optional add-ons, and obtain customer approval before work is performed.
8. Track the service, quality checks, estimate/invoice breakdown and vehicle collection status.
9. Contact the customer when the vehicle is actually ready and arrange collection.
10. Preserve the full timeline for future visits.

The app must also work when the voice service is unavailable. Employees should be able to complete every operational task manually.

## 2. Important product boundaries

### The model must not invent vehicle facts

Customer and vehicle details come from the local database. Service entitlements come from the customer's recorded plan and its terms. Service history comes from completed job cards. Prices come from the service catalogue. Diagnostic limits come from a versioned reference-specification table with a recorded source.

The LLM may explain retrieved facts in natural language. It must not generate a missing service history, decide that a reading is safe from general knowledge, invent a price, or claim that a payment succeeded.

### Recommendations must be explainable

Every recommended job should have a reason, source and urgency:

- **Required / safety critical:** supported by a technician's finding or an authoritative service rule.
- **Due by interval:** based on the exact vehicle variant, mileage, date and service schedule.
- **Advisory:** a technician has identified a potential issue that needs customer consideration.
- **Optional add-on:** a clearly labelled extra, not represented as mandatory.

A technician confirms findings. The customer approves chargeable work. The agent must never pressure the customer or silently add work to the job card.

### Free servicing needs its own rules

Do not model free servicing as a simple yes/no flag. Store the plan, number of visits, expiry, date/mileage window, included labour, included consumables, exclusions, and whether parts or taxes are chargeable. The employee and agent should be able to explain exactly what is covered.

### A local app is not the same as offline AI

The frontend, API and database run locally. Speech recognition, language-model responses and speech generation use external services, so internet access and provider quotas are still required. Customer information sent to those services must be minimised.

## 3. Recommended technology

Build a modular local application. The frontend, backend and operational database run on the laptop. The trained speech models run on remote GPU infrastructure because the laptop's GTX 1650 is not the target for model training or inference.

| Layer | Initial choice | Responsibility |
|---|---|---|
| Employee frontend | React + TypeScript + Vite | Dashboard, customer/vehicle records, job cards, inspections, approval forms and voice console |
| UI styling | Tailwind CSS | Employee-facing interface |
| Backend | Python + FastAPI | Business rules, APIs, validation, voice orchestration and model endpoints |
| Database | SQLite + SQLAlchemy + Alembic | Local customer, vehicle, service, diagnostic and approval records |
| Background jobs | APScheduler initially | Local reminders and appointment follow-ups while the app is running |
| Conversation LLM | Groq API adapter | Interpret the transcript, explain retrieved records and prepare structured tool calls; not used for ASR or TTS |
| ASR model | **Our own English speech recogniser, trained from random initial weights** | Convert microphone audio into text |
| TTS model | **Our own English speech synthesiser, trained from random initial weights** | Convert approved responses into speech |
| Training framework | PyTorch with SpeechBrain or ESPnet recipes as engineering toolkits | Dataset preparation, model implementation, training and evaluation; do not load pretrained speech-model weights |
| Training compute | Hosted GPU notebooks / remote GPU sessions | Train and evaluate models away from the GTX 1650 |
| Inference | Temporary remote GPU endpoint for the prototype | Serve our trained model to the local application; always-free availability is not guaranteed |
| Local voice transport | Browser microphone/speakers + WebSocket to FastAPI | Simulated voice conversation on the laptop |
| Documents | HTML/PDF generation | Job cards, inspection sheets, estimates and invoices |
| Tests | Pytest + frontend tests | Business rules, speech metrics, approval workflow and end-to-end tests |
| Optional comparison | Vapi Web SDK | Benchmark turn-taking and orchestration; not a core dependency |

**Model ownership rule:** the ASR and TTS models in the intended final prototype must have their own architectures/configurations and weights trained from random initialisation. SpeechBrain/ESPnet are toolkits, not pre-trained models. We may use public datasets with compatible licences, but we will not load a pre-trained ASR/TTS checkpoint or merely fine-tune one and call it our own model.

**Cost reality:** the software libraries and some datasets are free, but cloud GPU time and continuously available hosted inference are not guaranteed to be free. A notebook runtime can expose a temporary test endpoint while it is active; it is not a dependable always-on inference service. We will prove a small experiment before selecting any paid hosting or assuming a free tier can sustain it.

**Payment scope:** calculate estimates and line-item breakdowns only. Do not connect a payment gateway, create real payment requests or mark an invoice as paid in this project phase.

**Call scope:** the first milestone is a browser voice session using the laptop microphone and speakers. No customer phone numbers, PSTN calls or outbound campaign system are required for the test project.

## 4. High-level architecture

```text
Employee / test customer
        |
        v
React + TypeScript frontend (localhost)
        |                       |
        | REST / WebSocket      | microphone audio / speaker playback
        v                       v
FastAPI backend             Voice session manager
        |                       |
        |                       v
        |               Our ASR model endpoint
        |               (remote GPU; random-init weights)
        |                       |
        |                    transcript
        |                       |
        |                       v
        |               Conversation orchestrator
        |                 Groq LLM API
        |                       |
        |                approved backend tools
        |                       |
        |                       v
        |               SQLite business records
        |                       |
        |                 structured response
        |                       |
        |                       v
        |               Our TTS model endpoint
        |               (remote GPU; random-init weights)
        |                       |
        |                  generated audio
        |                       |
        +<----------------------+
        |
        +--> Customer / vehicle / service / reference data
        +--> Service history / inspections / job cards
        +--> Customer request drafts / supervisor approvals
        +--> Estimates and calculated payment breakdowns
        +--> Local generated documents and audit events

Optional Vapi benchmark:
Browser <-> Vapi Web SDK <-> Vapi-managed orchestration
                         <-> custom ASR/TTS endpoints (later experiment)
```

The LLM never receives unrestricted database access. It can request typed backend tools such as `get_customer_vehicle_summary`, `get_service_entitlement`, `compare_reading_to_reference`, `find_available_slots` or `create_customer_request_draft`. The backend validates the tool arguments and returns only the permitted records.

The model may draft a customer request, service preference or estimate explanation. It cannot approve the request, approve chargeable work, override reference values, or change authoritative company records without an authorised employee action.

For a locally running app to call a model trained in the cloud, the model must be reachable through an inference endpoint. During experimentation this can be a temporary notebook-backed endpoint. The app itself remains local; the speech model inference does not.

## 5. Main employee screens

1. **Dashboard:** today's appointments, service reminders, vehicles currently in the workshop, requests awaiting supervisor approval and vehicles ready for collection.
2. **Customers:** search by phone, customer ID or registration number; show contact preferences and consent history.
3. **Vehicle profile:** registration, VIN/chassis number, make, model, year, exact variant, engine/fuel/transmission, odometer and linked customer.
4. **Service timeline:** prior visits, work performed, parts used, invoices, technician notes, inspection records and next service due.
5. **Service plans:** free-service eligibility, inclusions, exclusions, expiry and usage.
6. **Appointments:** due reminders, available slots, call outcomes and customer confirmations.
7. **Check-in and job card:** arrival time, odometer, fuel level, visible damage, customer-reported complaints, photographs and employee checklist.
8. **Inspection and diagnostics:** measured values, units, test conditions, company-provided reference range, source/version, deviation and technician sign-off.
9. **Customer request inbox:** AI-generated form drafts containing what the customer said, requested work, questions, preferences, appointment choices and confidence/uncertainty flags.
10. **Supervisor approval queue:** approve, reject or edit request details; require confirmation for any change to the official work order or estimate.
11. **Estimate builder:** labour, parts, taxes, discounts, free-service coverage, chargeable items and calculated total. No payment collection.
12. **Workshop board:** assigned technician/bay, task status, parts wait, quality check and expected completion.
13. **Voice console:** selected customer, verification state, call state, transcript, retrieved facts, tool actions, draft form and handover controls.
14. **Settings and audit:** employees/roles, service catalogue, company-supplied reference values, model configuration, dataset/evaluation tools, audit trail and backup.

## 6. Core data model

## 6. Core data model

The database should be relational. Use stable IDs and foreign keys rather than embedding a customer's entire history in one JSON field.

| Entity | Key information |
|---|---|
| Employee / Role | Login, role and permitted actions |
| Customer | Name, verified contact details, communication preferences and consent |
| Vehicle | Customer link, registration, VIN, make/model/year and exact technical variant |
| ServicePlan | Plan terms, eligibility window, inclusions, exclusions and visit usage |
| Appointment | Vehicle, date/time, reason, status and confirmation |
| ServiceVisit | Check-in/check-out, odometer and visit status |
| JobCard / WorkOrder | Customer complaint, inspection, assigned staff, tasks and workflow state |
| ServiceCatalogueItem | Service/part, description, current price, tax and applicability |
| ServiceRecord | Work completed, parts fitted, technician and next-due information |
| DiagnosticReading | Parameter, measured value, unit, test conditions, device, timestamp and technician |
| ReferenceSpecification | Vehicle applicability, parameter, expected range, unit, conditions, source and version |
| Estimate / EstimateLine | Labour, parts, taxes, covered/chargeable classification and calculated total; no payment transaction |
| Estimate / CostBreakdown | Labour, parts, taxes, discounts, free-service coverage, chargeable items and calculated totals; no payment transaction |
| CustomerRequestDraft | Structured extraction of requested work, symptoms, preferences, appointment choices, unresolved fields and source transcript spans |
| SupervisorApproval | Draft version, approver, decision, edits, timestamp and reason |
| CallSession | Purpose, customer, timestamps, outcome, transcript and recording consent |
| ConsentRecord | Purpose, channel, wording/version, timestamp, source and withdrawal |
| AuditEvent | Who changed what, when, and the before/after reference |

For diagnostic comparisons, record the complete context: exact vehicle variant, engine, measurement unit, operating temperature/load, test procedure, device/source and timestamp. A numeric value without test conditions may be misleading. If no reliable reference exists for the exact variant and conditions, show **Reference unavailable** and route it to a technician; never ask the LLM to guess an optimal range.

Reference data should be traceable to manufacturer service manuals, technical bulletins or other approved sources. Do not assume that a generic OBD-II value is an OEM specification for every car.

## 7. Voice agent design

The initial voice interaction happens between the laptop microphone and speakers. It is a simulated conversation; it does not dial a customer's phone. The employee selects a synthetic/test customer and vehicle, then starts a session.

1. **Audio capture:** browser microphone captures the speaker's audio and streams it to the local FastAPI service.
2. **Voice activity / turn handling:** begin with simple audio-level/VAD logic and explicit push-to-talk/stop controls. Add streaming endpointing and interruption handling after the first working loop.
3. **Our ASR model:** the remotely hosted model trained from random initial weights transcribes the audio. Return partial and final transcripts where supported.
4. **Conversation manager:** send the transcript plus a minimal, verified context to the Groq LLM. It identifies intent and requests typed backend tools.
5. **Context retrieval:** backend fetches only the relevant customer, vehicle, plan, service-history and company-supplied diagnostic reference records.
6. **Structured extraction:** the agent prepares a `CustomerRequestDraft` form. It extracts requested work, symptoms, customer preferences, appointment choice, questions and missing/uncertain details. Preserve evidence from the transcript for each important field.
7. **Supervisor review:** the draft appears in the employee interface. A supervisor can edit, approve or reject it. Approval is explicit and auditable. Only an approved draft can be converted into an official job-card/work-order change.
8. **Response generation:** the Groq LLM composes a short response grounded in backend results. It must repeat critical items such as dates, registration numbers, odometer values and prices for confirmation.
9. **Our TTS model:** the remotely hosted custom TTS model generates spoken audio. The browser plays it back.
10. **Session record:** save the outcome, transcript, draft, approvals and consent-controlled audio metadata.

The agent must identify itself as an AI assistant, verify that it is speaking to the right person before disclosing vehicle details, handle interruptions and unclear answers, repeat critical numbers, and offer employee handoff. It must not disclose customer data to an unverified caller.

### Tool and approval boundaries

| Action | Agent behaviour |
|---|---|
| Read a vehicle/service summary | Allowed only after customer verification |
| Explain service history and entitlement | Read the company records and explain their terms |
| Compare diagnostic readings | Use deterministic comparison against the reference values supplied by the company |
| Suggest an appointment | Allowed using actual availability |
| Draft appointment/customer preferences | Save as an unapproved draft |
| Create or change an official appointment/work order | Require supervisor approval in this test project |
| Recommend additional work | Explain only what company/technician records support; label optional vs required |
| Add chargeable work | Draft only; supervisor approval before committing |
| Calculate estimate | Generate line-item amounts and totals from the catalogue; no payment processing |
| Mark work as completed/vehicle ready | Only after authorised employee/workshop status update |

### Optional Vapi comparison

Vapi supports browser-based voice interfaces, tool calls, custom transcriber WebSockets and custom TTS endpoints. It could be tested as a benchmark after our own ASR/TTS have working endpoints. Vapi also retains its proprietary real-time orchestration layer, so using it would not mean we built the whole speech pipeline ourselves. Its current usage-based entry pricing lists $0.05 per minute of Vapi hosting, with model costs passed through separately and $5 initial credits; it is not an unlimited-free option.

For the core project, do not make Vapi a dependency. Keep a provider-neutral voice interface so a later experiment can compare Vapi and our own local-app orchestration on the same scripted evaluation calls.

## 8. Training your own speech capability

This project explicitly aims to build our own English ASR and TTS models from random initialisation. A pre-trained Whisper/Deepgram/ElevenLabs model, or fine-tuning an existing pretrained speech checkpoint, does **not** satisfy the main research objective.

A toolkit such as SpeechBrain or ESPnet can be used to implement training recipes, data loaders, feature extraction, checkpointing and evaluation. Using a toolkit does not mean using its pretrained weights. The SpeechBrain documentation includes a speech-recognition-from-scratch recipe and describes feature extraction, augmentation, CTC, transducer and attention-based model designs.

### 8.1 ASR — speech to text (first model to build)

Recommended first research architecture:

- Convert input waveforms into log-Mel filter-bank features.
- Build a compact CNN/subsampling + Conformer-style encoder with randomly initialised parameters.
- Train with a CTC objective as the first stable baseline.
- Decode token sequences into text; later evaluate beam search and a language-model rescoring component.
- Add streaming/chunked inference only after offline recognition quality is measured.

This is a learning/research architecture, not a guarantee of production accuracy. Start with a small training experiment, verify the end-to-end training pipeline, then increase model size/data only when the metrics improve.

### 8.2 ASR data plan

Training from random initialisation requires a substantial labelled speech corpus. The first training/evaluation foundation can use a public English dataset with a licence suitable for the project. LibriSpeech is an approximately 1,000-hour, 16 kHz read-English corpus; its 100-hour clean subset is a practical first experiment, not enough to establish real-world workshop-call accuracy by itself.

Read audiobook speech differs from spontaneous service conversations. Add a separate company-domain dataset, only with proper permission/consent, including:

- English speech from the expected customer and employee accents.
- Actual conversational turns, pauses, self-corrections, interruptions and background noise.
- Automotive terms, makes/models, parts, service names and common customer complaints.
- Numbers spoken in realistic forms: registrations, phone digits, dates, mileage, prices and appointment times.
- Corrected transcripts checked by a human annotator.

Split data by speaker, not randomly by individual clips, so the same speaker does not appear in both training and final test data. Keep a held-out workshop-style evaluation set that is never used for training.

### 8.3 TTS — text to speech (second model)

Train a separate English text-to-speech model from random initialisation. Begin with a single consistent speaker voice and a carefully recorded corpus where each audio clip has an exact matching transcript. Use clean recordings, consistent microphone distance, low background noise and a balanced range of phonemes, numbers, dates and automotive vocabulary.

TTS quality is a distinct problem from ASR accuracy. Evaluate pronunciation, intelligibility, naturalness, prosody, stability and real-time generation latency. Do not imitate a real person's voice without explicit permission and appropriate rights.

### 8.4 Evaluation gates

For ASR, measure:

- Word Error Rate (WER) and Character Error Rate (CER).
- Exact-match accuracy for high-impact slots: registration, model/variant, phone digits, dates, odometer, quantity, price and appointment time.
- Performance by speaker/accent, noise condition, audio device and utterance type.
- Partial-transcript latency and final-transcript latency.

For TTS, evaluate:

- Human-rated intelligibility and naturalness.
- Pronunciation of model names, parts, numbers, dates and amounts.
- Audio generation speed and playback interruptions.
- Consistency across short and long utterances.

Do not claim the model is accurate because it works on a few demo sentences. Maintain a fixed regression set and track every experiment.

### 8.5 Compute and realistic expectations

The GTX 1650 is excluded from speech-model training and inference. Use cloud GPU sessions for training and a remote GPU endpoint for inference. Free notebook GPU availability, session duration, memory and quotas can change. A free notebook may be sufficient for a small research experiment, but cannot be assumed to train large models quickly or host a permanently available, low-latency endpoint.

Training an ASR model from scratch is a substantial but valid research project. Training a good TTS model from scratch is another substantial project. A fully natural, streaming, production-grade speech system with no pretrained weights and no paid compute is not a realistic promise. We will start small, publish honest metrics and decide whether the free compute ceiling is limiting us before expanding.

### 8.6 Interim development behaviour

The final speech components should be our trained models. While ASR is being developed, the rest of the servicing application can be tested with typed transcripts. While TTS is being developed, the agent can display its response as text. A system/browser voice may be used only as an explicitly temporary test aid and removed from the final custom-model path.

## 9. Safety, privacy and operational controls

- **Identity verification:** do not reveal customer or vehicle data until the caller is verified with an appropriate check.
- **Human approval:** technicians approve diagnoses; customers approve chargeable work; authorised staff handle exceptions, discounts and refunds.
- **Auditability:** log tool calls, approvals, edits, payment status changes and employee identity.
- **Data minimisation:** send the speech provider only the data required for the current turn. Do not put API keys in frontend code.
- **Audio consent:** define whether calls are recorded, how the customer is informed, retention period, access permissions and deletion process. Do not keep recordings by default just because they are technically available.
- **Payment scope:** this project only calculates and displays estimates/payment breakdowns. There is no payment gateway, collection, payment link or paid-status workflow in the current scope.
- **Backups:** provide a database backup/restore workflow. Local-only does not mean data cannot be lost.
- **Fallback:** if a model times out, reaches a quota, or returns low-confidence/ambiguous output, the UI must let an employee continue manually or correct the transcript. Do not auto-approve generated customer-request forms.
- **Call scope:** the initial laptop-microphone simulation does not dial external phone numbers. Revisit telephony and applicable Indian calling rules only if the project later adds real phone calls.

## 10. Suggested repository structure

This is the proposed structure, not a claim that these folders already exist.

```text
Car_Service_Manager_TTS/
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── features/
│   │   │   ├── dashboard/
│   │   │   ├── customers/
│   │   │   ├── vehicles/
│   │   │   ├── appointments/
│   │   │   ├── job-cards/
│   │   │   ├── diagnostics/
│   │   │   ├── estimates/
│   │   │   ├── invoices/
│   │   │   └── voice-agent/
│   │   └── services/
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── workflows/
│   │   ├── voice/
│   │   │   ├── orchestrator.py
│   │   │   ├── tools.py
│   │   │   ├── providers/
│   │   │   │   ├── asr/
│   │   │   │   ├── llm/
│   │   │   │   ├── tts/
│   │   │   │   └── telephony/
│   │   │   └── policies/
│   │   └── main.py
│   ├── migrations/
│   └── tests/
├── training/
│   ├── asr/
│   │   ├── config.yaml
│   │   ├── prepare_librispeech.py
│   │   ├── train_tokenizer.py
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── metrics.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── plot_metrics.py
│   │   ├── inference.py
│   │   ├── server.py
│   │   └── README.md
│   ├── tts/                   # planned second from-scratch speech model
│   └── data/                  # local licensed/consented audio; ignored by Git
├── sample_data/
├── generated_documents/       # ignored by Git
├── local_data/                # ignored by Git; database and backups
├── .env.example
├── .gitignore
├── architecture.md
├── progress.md
└── README.md
```

Keep customer exports, real recordings, API keys, databases, invoices and generated customer documents out of Git. Commit only synthetic sample records and safe test audio.

## 11. Phased delivery plan

### Phase 0 — Confirm scope and prepare sample data
English-only; local employee application; laptop microphone/speaker conversations; no real calls; no payment integration; company records as the source of truth; reference values supplied by you; supervisor approval required for AI-created customer-request forms.

### Phase 1 — Local application and structured data
Build the frontend shell, FastAPI backend, SQLite schema, synthetic seed data, customer/vehicle profiles, service history, service plans, reference values, diagnostic comparisons and employee dashboard.

### Phase 2 — ASR research: first model from scratch
Prepare a licensed English dataset and company-domain recording plan. Implement a random-initialised ASR architecture, train a small baseline on a cloud GPU, build evaluation scripts and record WER plus critical-field accuracy. No pretrained ASR checkpoint.

### Phase 3 — Voice loop and business tools
Connect browser microphone audio to our ASR endpoint; send transcripts to Groq for conversational reasoning; connect typed tools to the backend; display retrieved context and structured customer-request drafts. Until custom TTS is ready, responses can be shown as text.

### Phase 4 — Customer request and supervisor workflow
Add the AI-generated form, source transcript evidence, editable fields, missing-information flags, supervisor approval/rejection and an audit log. Only approved forms may update official records.

### Phase 5 — TTS research: second model from scratch
Collect a permitted single-speaker English recording corpus with exact transcripts. Build and train a random-initialised TTS model remotely. Evaluate intelligibility, pronunciation, naturalness and generation latency, then connect it to browser playback.

### Phase 6 — Complete voice-agent evaluation
Test the end-to-end conversation for service reminders, entitlement explanations, appointment preferences, customer-reported complaints, estimate explanations and vehicle collection. Test low-confidence recognition, corrections, interruptions and API/model outages.

### Phase 7 — Optional Vapi benchmark
Run matched scripted conversations through Vapi Web SDK and our own pipeline. Compare ASR errors, critical field accuracy, end-of-turn behaviour, interruptions, latency, user experience and cost. Vapi remains optional; real telephony is outside the current scope.

### Phase 8 — Model improvement
Expand the custom ASR/TTS training datasets only when held-out evaluation identifies a measurable weakness. Document data provenance, training configurations, model checkpoints and test results.

## 12. Remaining decisions and assumptions

The major product decisions supplied by the user are now fixed:

- **Language:** English only.
- **Truth source:** accurate customer/vehicle records already held by the company.
- **Reference values:** supplied by the user and stored in the application.
- **Model objective:** train our own ASR and TTS weights from random initialisation; do not substitute pretrained speech checkpoints or fine-tuning for the main objective.
- **Execution:** local application; remote cloud model inference; no local Ollama/model serving.
- **Voice interaction:** laptop microphone and speakers for the test project.
- **Vapi:** optional comparison/voice orchestration experiment, not a dependency.
- **Payments:** calculate estimate and payment breakdown only; no real payments.
- **Customer changes:** AI drafts a structured form; supervisor reviews and approves before official records change.

Remaining technical inputs to collect during implementation:

1. Company customer/vehicle/service-history export format and sample synthetic records.
2. The reference-value spreadsheet or schema, including units, thresholds and vehicle applicability.
3. A permitted English speech dataset and, if available, consented company call recordings.
4. The desired English voice/accent for the TTS model and whether a speaker can record a clean training corpus.
5. The cloud GPU notebook/runtime available for training and temporary inference.

**Recommended first milestone:** build the database and reference-value comparison plus a small ASR-from-scratch experiment in parallel. This establishes both halves of the project early: the application has accurate data, and the custom speech-model objective is being tested with measurable results.

## 13. Reference links

- [SpeechBrain speech recognition from scratch tutorial](https://speechbrain.readthedocs.io/en/stable/tutorials/tasks/speech-recognition-from-scratch.html)
- [SpeechBrain PyTorch speech toolkit](https://github.com/speechbrain/speechbrain)
- [LibriSpeech ASR corpus — OpenSLR, approximately 1,000 hours, CC BY 4.0](https://openslr.org/12/)
- [TensorFlow Datasets LibriSpeech overview](https://www.tensorflow.org/datasets/catalog/librispeech)
- [Vapi Web SDK / browser voice calls](https://docs.vapi.ai/quickstart/web)
- [Vapi custom transcriber integration](https://docs.vapi.ai/customization/custom-transcriber)
- [Vapi custom TTS integration](https://docs.vapi.ai/customization/custom-voices/custom-tts)
- [Vapi voice data flow and orchestration boundaries](https://docs.vapi.ai/security-and-privacy/data-flow)
- [Vapi pricing](https://vapi.ai/pricing)
