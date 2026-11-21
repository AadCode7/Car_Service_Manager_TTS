# Project Progress

**Project:** Car Service Manager with Voice Agent  
**Repository:** https://github.com/AadCode7/Car_Service_Manager_TTS  
**Working branch:** `main`  
**Last updated:** 2026-11-21

## Project objective

Build a local employee-facing car service management application with a voice agent. The frontend, backend and database run on the laptop. Speech recognition, LLM inference and speech generation use external APIs; no Ollama or other local model inference is planned.

The voice agent should retrieve accurate customer/vehicle/service data and help with reminders, service explanations, appointment scheduling and collection calls. Employees use the same application for customer check-in, job cards, inspections, diagnostics, estimates, approvals, workshop progress, invoices and payment tracking.

## Scope and constraints

- Local application; no deployment requirement for the initial project.
- No local LLM, ASR or TTS inference because of the GTX 1650 hardware constraint.
- Prefer free developer tiers and free API credits. Treat provider quotas, rate limits, model access and pricing as changeable; do not assume unlimited free usage.
- Keep AI providers behind adapters so providers can be changed independently.
- The database, not the LLM, is the source of truth.
- Use synthetic data until privacy, consent, access and backup controls are ready.
- All significant customer-facing or financial actions require backend validation and explicit confirmation.
- Real telephone calls and payment processing are later integrations; browser-based simulated calls and test payments come first.
- Work directly on `main`, as requested. Do not create a feature branch unless explicitly requested.

## Current repository state

- [x] Confirmed repository exists and is writable.
- [x] Confirmed default branch is `main`.
- [x] Confirmed repository was empty at the start of this project (size 0; no existing code or README to preserve).
- [x] Add initial architecture and progress documents.
- [ ] Add the initial application scaffold after planning decisions are confirmed.

## Architecture decisions proposed

| Area | Proposed initial decision | Status |
|---|---|---|
| Frontend | React + TypeScript + Vite + Tailwind | Proposed |
| Backend | Python + FastAPI | Proposed |
| Database | SQLite + SQLAlchemy + Alembic | Proposed |
| Text LLM | Groq API behind a provider interface | Proposed |
| ASR | Groq Whisper baseline; evaluate Sarvam for Indian-language accuracy | Needs language decision |
| TTS | Groq Orpheus for supported English; evaluate Sarvam Bulbul for Indian languages | Needs language decision |
| Voice calls | Browser simulation first; telephony later | Proposed |
| Payment | Invoice/payment request in test mode first; trusted provider verification for paid status | Proposed |
| Custom model | Fine-tune an open-weight model in a cloud GPU notebook after building a labelled dataset | Needs user confirmation |
| Deployment | Local only for initial project | Confirmed |

## Progress log

### 2026-11-21 — Planning baseline

- Inspected the GitHub repository and confirmed it is empty with `main` as the default branch.
- Defined the product as an employee operations system with a voice interface, not simply a voice chatbot.
- Proposed a local React + FastAPI + SQLite architecture with external model APIs.
- Separated ASR, LLM orchestration, TTS and telephony into independent provider interfaces.
- Defined the safety boundary: structured records are authoritative; the LLM explains and requests validated tools; employees/technicians/customers approve consequential actions.
- Identified vehicle variant-specific service data and diagnostic-reference provenance as core requirements.
- Identified Indian-language speech support and real outbound telephony as decisions that affect provider selection and compliance.
- Added a staged implementation plan to `architecture.md`.
- Committed `architecture.md` and `progress.md` directly to `main` as the initial project documents.

## Implementation checklist

### Phase 0 — Requirements and foundations
- [ ] Confirm spoken languages and code-mixing requirements.
- [ ] Confirm browser simulation before real calls.
- [ ] Confirm single-branch vs multi-branch scope.
- [ ] Identify initial car makes/models and available service/diagnostic data.
- [ ] Confirm invoice/UPI request vs payment gateway scope.
- [ ] Confirm fine-tuning vs training from scratch.
- [ ] Confirm local Python and Node.js versions available.

### Phase 1 — Local application and structured data
- [ ] Create repository scaffold without overwriting existing files.
- [ ] Add safe configuration loading and `.env.example`.
- [ ] Add SQLite database and migrations.
- [ ] Add synthetic seed data.
- [ ] Build employee dashboard.
- [ ] Build customer and vehicle search/profile.
- [ ] Display service history and service-plan entitlements.
- [ ] Add backend and frontend tests.

### Phase 2 — Service operations
- [ ] Appointment scheduling and availability checks.
- [ ] Vehicle check-in and job-card creation.
- [ ] Technician inspection and diagnostic recording.
- [ ] Versioned, sourced vehicle specification reference data.
- [ ] Estimate creation and customer approval.
- [ ] Workshop status board and quality-control step.

### Phase 3 — Voice simulation
- [ ] Add ASR provider adapter.
- [ ] Add LLM provider adapter and structured tool definitions.
- [ ] Add TTS provider adapter.
- [ ] Build browser-based voice session.
- [ ] Add customer verification and confirmation prompts.
- [ ] Add call transcript and outcome timeline.
- [ ] Test due-service and vehicle-ready call scenarios.

### Phase 4 — Evaluation and speech data
- [ ] Create an opt-in speech data collection process.
- [ ] Build domain vocabulary and transcription correction workflow.
- [ ] Evaluate WER/CER and critical-field accuracy.
- [ ] Add multilingual, noise and regression test sets.
- [ ] Compare hosted providers on the same test set.

### Phase 5 — Documents and payments
- [ ] Generate job cards, estimates and invoices.
- [ ] Create mock/test payment requests.
- [ ] Add verified payment-provider status callback/reconciliation.
- [ ] Add document backup and export.

### Phase 6 — Custom speech model
- [ ] Select model and licence after evaluating collected data.
- [ ] Prepare training/validation/test splits by speaker.
- [ ] Fine-tune in a cloud GPU notebook.
- [ ] Evaluate against hosted baseline.
- [ ] Decide on a remote inference host only if accuracy gains justify the added cost and operations.

### Phase 7 — Live telephony (optional)
- [ ] Select provider and confirm India call compliance.
- [ ] Implement caller identity, purpose, consent and opt-out handling.
- [ ] Add inbound/outbound call routing and interruption handling.
- [ ] Add failure recovery, call retries and employee handoff.
- [ ] Test with internal numbers before customer calls.

## Known risks / open issues

- Free API quotas can change and may not cover production call volumes.
- Groq's documented TTS voices currently cover English and Arabic; Indian-language TTS needs a separate provider/model.
- ASR correctness for registration numbers, mileage, dates, model names and prices must be tested separately from general transcript quality.
- Manufacturer diagnostic limits depend on exact vehicle variant and test conditions. Missing reference data must be displayed as unavailable, not guessed.
- Actual outbound phone calls require a telephony provider and current regulatory checks; a speech API alone cannot place a phone call.
- Payment completion must come from a trusted gateway response or authorised reconciliation, never from the agent's conversation.
- Local databases and audio files need backup, access control, retention and deletion policies.
- Training a truly custom model requires consented, correctly labelled, representative audio. Fine-tuning a pretrained open-weight model is the practical first target; training from scratch is not the initial plan.

## Working rules for future changes

1. Inspect the current files before changing them.
2. Preserve existing working code and features.
3. Make changes directly on `main`.
4. Add or update tests for each implemented feature.
5. Update this file after each meaningful milestone, including what changed and what remains.
6. Do not add real customer data, API keys, recordings or generated customer documents to Git.
7. Do not mark a task complete unless it has actually been implemented and checked.
