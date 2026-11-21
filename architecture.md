# Car Service Manager with Voice Agent

**Project status:** Planning / Phase 0  
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
8. Track the service, quality checks, invoice and payment status.
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

Keep the first version as a simple local modular application. Avoid microservices and avoid adding infrastructure before there is a clear need.

| Layer | Initial choice | Responsibility |
|---|---|---|
| Employee frontend | React + TypeScript + Vite | Dashboard, customer/vehicle records, job cards, inspections, calls and invoices |
| UI styling | Tailwind CSS | Consistent employee-facing interface |
| Backend | Python + FastAPI | Business rules, APIs, validation and AI orchestration |
| Database | SQLite | Local customer, vehicle, service and payment records |
| Data access | SQLAlchemy + Alembic | Database models and controlled schema migrations |
| Background jobs | APScheduler initially | Local reminders and scheduled tasks while the app is running |
| LLM | Groq API adapter initially | Conversation reasoning, structured extraction and tool selection |
| Speech-to-text | Provider adapter: Groq Whisper; evaluate Sarvam for Indic speech | Convert customer audio into text |
| Text-to-speech | Provider adapter: Groq Orpheus for supported English voices; evaluate Sarvam Bulbul for Indian languages | Speak the agent's response |
| Telephony | Mock call adapter first; provider integration later | Real outbound/inbound calls require a phone/telephony service |
| Documents | HTML/PDF generation | Job cards, inspection sheets, estimates and invoices |
| Tests | Pytest + frontend tests | Verify business rules, permissions and end-to-end workflows |

Use a provider interface for each AI capability. Do not call Groq or another provider directly from UI components or business workflows. This lets us switch providers without redesigning the application.

**Speech provider note (checked 21 November 2026):** Groq documents multilingual Whisper speech-to-text, but its documented Orpheus TTS voices currently cover English and Arabic, not Hindi or Marathi. Sarvam documents Indian-language speech-to-text and text-to-speech, including Hindi and Marathi, but offers ₹100 of signup credits rather than unlimited free usage. These are candidate adapters, not a promise that production-scale speech will remain free. Confirm current model access, rate limits, pricing and commercial terms before enabling real calls.

## 4. High-level architecture

```text
Employee
   |
   v
React + TypeScript frontend (localhost)
   |                       |
   | REST / WebSocket      | audio capture / playback
   v                       v
FastAPI backend        Voice session interface
   |                       |
   |                       v
   |                 Speech orchestration
   |                 ASR -> conversation
   |                 -> approved tools -> TTS
   |                       |
   |                 External model APIs
   |                 (Groq / selected provider)
   |
   +--> Application services and business rules
   |      Customers / Vehicles
   |      Service plans / Service history
   |      Appointments / Job cards
   |      Diagnostics / Estimates
   |      Inventory / Invoices / Payments
   |      Calls / Consent / Audit
   |
   +--> SQLite database
   |
   +--> Local documents and generated PDFs
   |
   +--> Provider adapters
          Telephony (mock first)
          Payment (test/mock first)
          Email/SMS (later, optional)
```

The LLM never receives unrestricted database access. It can request specific, typed tools such as `get_customer_vehicle_summary`, `get_service_entitlement`, `find_available_slots`, `create_appointment` or `create_estimate_draft`. The backend validates the user's identity, permissions, current workflow state and required confirmations before executing a tool.

## 5. Main employee screens

1. **Dashboard:** today's appointments, overdue service reminders, vehicles currently in the workshop, work awaiting approval, vehicles ready for collection and outstanding invoices.
2. **Customers:** search by phone, customer ID or registration number; show contact preferences and consent history.
3. **Vehicle profile:** registration, VIN/chassis number, make, model, year, exact variant, engine/fuel/transmission, odometer, warranty and linked customer.
4. **Service timeline:** prior visits, work performed, parts used, invoices, technician notes, inspection records and next service due.
5. **Service plans:** free-service eligibility, inclusions, exclusions, expiry and usage.
6. **Appointments and calls:** due reminders, call attempts, outcomes, customer confirmations and rescheduling.
7. **Check-in and job card:** arrival time, odometer, fuel level, visible damage, customer-reported complaints, photographs and employee checklist.
8. **Inspection and diagnostics:** measured values, units, test conditions, reference range, source, deviation and technician sign-off.
9. **Estimate and approvals:** labour, parts, taxes, discounts, covered items, optional work and approval status.
10. **Workshop board:** assigned technician/bay, task status, parts wait, quality check and expected completion.
11. **Invoice and payment:** final line items, payment request/QR/link, provider reference and independently verified payment status.
12. **Voice agent console:** selected customer, permitted context, call state, transcript, tool actions, confirmation prompts and handover controls.
13. **Settings and audit:** employees/roles, service catalogue, reminder rules, model providers, API status, audit trail and data backup.

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
| Estimate / EstimateLine | Labour, parts, covered/chargeable classification and customer approval |
| Invoice / Payment | Totals, taxes, due amount, payment request, provider reference and verified status |
| CallSession | Purpose, customer, timestamps, outcome, transcript and recording consent |
| ConsentRecord | Purpose, channel, wording/version, timestamp, source and withdrawal |
| AuditEvent | Who changed what, when, and the before/after reference |

For diagnostic comparisons, record the complete context: exact vehicle variant, engine, measurement unit, operating temperature/load, test procedure, device/source and timestamp. A numeric value without test conditions may be misleading. If no reliable reference exists for the exact variant and conditions, show **Reference unavailable** and route it to a technician; never ask the LLM to guess an optimal range.

Reference data should be traceable to manufacturer service manuals, technical bulletins or other approved sources. Do not assume that a generic OBD-II value is an OEM specification for every car.

## 7. Voice agent design

Treat speech recognition, conversation, business actions and speech generation as separate components.

1. **Audio input:** browser microphone for local testing; telephony audio stream only after a provider is selected.
2. **ASR:** transcribe the customer's speech. Keep the transcript and language metadata for the session.
3. **Conversation manager:** identify intent, maintain the current call state and decide which allowed tool is required.
4. **Context retrieval:** fetch only the relevant customer, vehicle, plan, service and appointment data through backend functions.
5. **Business tools:** validate and execute allowed actions. Any booking, paid work, cancellation, customer-facing estimate or other consequential change needs the appropriate confirmation.
6. **Response composition:** create a concise answer from returned structured facts. Include the source/context internally so the employee can inspect it.
7. **TTS:** synthesise the approved response and play it back.
8. **Call outcome:** save the disposition, confirmed appointment or follow-up action, transcript and consent-controlled recording metadata.

The agent must identify itself as an AI assistant, verify that it is speaking to the right person before disclosing vehicle details, handle interruptions and unclear answers, repeat critical numbers, and offer transfer to an employee. It should not disclose a full VIN, address, invoice or vehicle history to an unverified caller.

### Initial tool permissions

| Action | Agent behaviour |
|---|---|
| Read vehicle/service summary | Allowed after customer verification |
| Explain free-service entitlement | Allowed using the stored plan terms |
| Suggest a slot | Allowed |
| Book/reschedule a slot | Only after customer confirmation and availability re-check |
| Explain inspection findings | Only from technician-approved findings |
| Recommend optional work | Explain evidence and price; never mark as approved |
| Add chargeable work | Requires explicit customer approval |
| Generate estimate/invoice draft | Backend action; employee review initially |
| Generate payment request | Only from a finalised invoice |
| Mark invoice as paid | Never based on speech; only a verified payment-provider result or authorised employee reconciliation |
| Mark vehicle ready | Only after workshop and quality-control status says ready |

The first release should use a **simulated call**: employee selects a customer, starts a browser voice session, and tests the full conversation without placing a telephone call. Real PSTN calls should be a later integration.

## 8. Training your own speech capability

A key distinction: an LLM is not a speech model. ASR converts audio to text; an LLM interprets the text and selects actions; TTS converts the response to audio. Training or fine-tuning one does not automatically train the others.

Do not train a speech model from scratch as the first step. Start with strong pretrained hosted models, establish an accuracy baseline, build a consented domain dataset, then fine-tune an open-weight ASR model in a cloud notebook when the data justifies it. Training can happen in a hosted GPU notebook; the GTX 1650 is not used for model training or inference. Any custom model must be served remotely if we later choose to use it in the app.

### Data needed for domain adaptation

- Correct pronunciations of manufacturer, model, variant, part and service names.
- Registration numbers, phone numbers, VIN fragments, dates, quantities, prices and mileage.
- Indian English and the actual languages/code-mixing used by the company's customers (for example Hindi, Marathi and Hinglish, if required).
- Common customer phrases, workshop terminology, background noise and different phone/microphone quality.
- Human-corrected transcripts linked to audio, with speaker consent and data minimisation.

Keep a held-out evaluation set from different speakers. Evaluate word/character error rate, but also measure the accuracy of critical fields such as registration numbers, dates, amounts, mileage, model names and appointment times. A low overall word error rate does not guarantee that these high-impact fields are correct.

For TTS, first select a natural voice and language that customers understand. Custom voice training/cloning is a separate project, requires clear permission from the voice owner and suitable commercial rights, and should not imitate a real person without authorisation.

## 9. Safety, privacy and operational controls

- **Identity verification:** do not reveal customer or vehicle data until the caller is verified with an appropriate check.
- **Human approval:** technicians approve diagnoses; customers approve chargeable work; authorised staff handle exceptions, discounts and refunds.
- **Auditability:** log tool calls, approvals, edits, payment status changes and employee identity.
- **Data minimisation:** send the speech provider only the data required for the current turn. Do not put API keys in frontend code.
- **Audio consent:** define whether calls are recorded, how the customer is informed, retention period, access permissions and deletion process. Do not keep recordings by default just because they are technically available.
- **Payment security:** never ask for card PIN, OTP, CVV or banking passwords. Use a payment provider's hosted checkout/payment flow; never trust a spoken claim or an LLM response as proof of payment.
- **Backups:** provide a database backup/restore workflow. Local-only does not mean data cannot be lost.
- **Fallback:** if a model times out, reaches a quota, or returns low-confidence/ambiguous output, the UI must let an employee continue manually.
- **India telephony compliance:** before real automated outbound calls, check the current TRAI/TCCCPR rules and the selected telecom provider's onboarding requirements. The 18 September 2026 government summary describes A2P calls and a requirement for entities using them to pre-declare their use and caller IDs to their telecom service provider. Keep consent, call purpose, caller ID and opt-out handling as explicit configuration and records; obtain telecom/legal review before production calling.

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
│   ├── data/                  # ignored by Git; consent-controlled
│   ├── manifests/
│   ├── notebooks/
│   ├── scripts/
│   └── evaluation/
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

### Phase 0 — Requirements and foundations
Confirm languages, workshop workflow, service-plan rules, data sources, local environment and the meaning of “own speech model”. No production integrations yet.

### Phase 1 — Local application and structured data
Create the frontend shell, FastAPI backend, SQLite schema, sample data and employee/customer/vehicle screens. The employee must be able to view a vehicle's accurate history manually.

### Phase 2 — Service operations
Add appointments, service eligibility, check-in, job cards, inspection forms, diagnostic readings, source-backed reference comparisons, estimates and approval tracking.

### Phase 3 — Voice assistant in simulation
Add hosted ASR, LLM tool calling and TTS through provider adapters. Let an employee test the full conversation without placing a telephone call, using synthetic data first.

### Phase 4 — Voice accuracy and evaluation
Create a consented, reviewed speech dataset. Add domain vocabulary, critical-field confirmation, multilingual tests, regression tests and model/provider comparisons.

### Phase 5 — Payments and documents
Generate job cards, estimates, invoices and payment requests in test mode. Verify payment status only through a trusted provider response.

### Phase 6 — Training and custom model evaluation
Fine-tune a suitable open-weight ASR model using a cloud GPU notebook if the collected data shows a measurable gap. Compare it with hosted APIs before deciding whether the custom model is worth operating. TTS customisation is a separate decision.

### Phase 7 — Real telephony (optional)
Integrate a compliant telephony provider, real call routing, recording/consent handling, retries, opt-outs and live call monitoring. This is not included in the initial local-only prototype.

## 12. Decisions to confirm before deeper implementation

These decisions affect model selection and the data model. We can keep the rest of the architecture provider-neutral until they are answered.

1. **Languages:** English only, Hindi, Marathi, Hinglish, or more? Which languages must the agent both understand and speak?
2. **Call mode:** Is the first milestone a browser-based simulated call, or must it place real calls to customers from the start?
3. **Business scope:** One service centre initially, or multiple branches with separate staff, bays, prices and inventory?
4. **Vehicle coverage:** Which manufacturers/models should the initial reference data support? Do you have service manuals, diagnostic reports or historical job-card exports?
5. **Payments:** Is the first version expected to create invoices and UPI payment requests, or to integrate with an actual payment gateway?
6. **Speech model ownership:** Is fine-tuning an existing open-weight speech model acceptable, or do you specifically mean training model weights from scratch? Fine-tuning is the practical route for this project.

**Recommended first milestone:** Build the local employee application and a reliable vehicle/service-history database using synthetic data. Then prove one full journey—service due reminder, simulated customer call, confirmed booking, check-in, job card, technician-approved inspection, estimate approval and collection call—before adding real customer data or live calls.

## 13. Reference links

- [Groq Speech-to-Text documentation](https://console.groq.com/docs/speech-to-text)
- [Groq Text-to-Speech documentation](https://console.groq.com/docs/text-to-speech/)
- [Groq rate limits](https://console.groq.com/docs/rate-limits)
- [Sarvam model and language list](https://docs.sarvam.ai/api/getting-started/models)
- [Sarvam pricing, credits and rate limits](https://docs.sarvam.ai/api/getting-started/pricing)
- [TRAI / Government of India update, 18 September 2026](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2311989)
- [Kaggle notebook GPU documentation](https://www.kaggle.com/docs/notebooks)
