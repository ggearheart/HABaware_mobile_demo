# HABaware Tech Stack

This document captures the recommended architecture for HABaware — a generative AI and predictive water quality mobile app requiring high-performance UI, offline-capable edge inference, and secure cloud data ingestion.

---

## 1. Mobile Front-End

| Decision | Choice |
|---|---|
| **Primary framework** | Flutter (Impeller rendering engine) |
| **Alternative** | React Native (Expo + New Architecture) — only if team has deep TypeScript/web ecosystem needs |

Flutter is preferred for its native strength in graphics-heavy, map-driven UIs — essential for rendering color-coded HAB risk maps — and its consistent cross-platform performance.

---

## 2. On-Device Machine Learning (Edge AI)

| Decision | Choice |
|---|---|
| **Inference runtime** | LiteRT (formerly TFLite) or ONNX Runtime Mobile |
| **Hardware acceleration** | Google AI Edge tools → NNAPI (Android), CoreML (iOS) |
| **Model candidates** | Lightweight LSTMs, LightGBM |

Users near remote lakes may have no cellular service. The predictive model (e.g., chlorophyll-a spike forecasting, cyanotoxin risk scoring) must run fully on-device for offline safety warnings. Hardware accelerators keep inference fast and power-efficient.

---

## 3. Generative AI (Risk Communication)

| Mode | Choice |
|---|---|
| **Cloud (connected)** | Anthropic Claude API (primary) or Google Gemini API |
| **On-device / offline** | MediaPipe LLM Inference API or AI Edge Torch Generative API (SLMs) |

When the on-device model flags elevated risk, numeric predictions are passed to a cloud LLM via a structured prompt. The LLM generates plain-language, actionable advisories (e.g., *"High bloom risk detected near this location — avoid water contact and keep pets away"*). On-device SLMs serve as fallback when connectivity is unavailable or when privacy is the top priority.

> **Note on model choice:** Claude is preferred for its strong instruction-following, low hallucination rate on factual/safety-critical outputs, and native tool-use support for structured data grounding. GPT-4o or Gemini are viable alternatives.

---

## 4. Back-End & Data Processing

| Component | Choice | Reason |
|---|---|---|
| **Cloud platform** | GCP or AWS | Serverless scaling, managed ML services |
| **Data ingestion** | Python serverless functions | Stream NASA Earthdata (satellite imagery), USGS/EPA sensor feeds, CA SWAMP FHAB API |
| **Database** | Supabase (PostgreSQL + PostGIS) | Spatial queries required: match GPS coordinates to waterbodies, calculate distance to bloom risk zones |

PostGIS is a hard requirement for any location-based HAB risk feature — standard relational or document databases cannot efficiently handle geospatial proximity queries.

---

## 5. Key Data Sources

| Source | Data Type |
|---|---|
| CA SWAMP FHAB Field Program | Ground-truth bloom observations, cyanotoxin measurements, advisory status |
| CA SWAMP FHAB Remote Sensing | Satellite-derived bloom indices, surface extent, intensity maps |
| NASA Earthdata | Multi-spectral satellite imagery |
| USGS / EPA sensor networks | Real-time water quality parameters (temperature, turbidity, nutrients) |

---

## Open Architecture Questions

- **ML models:** LSTMs vs. Transformers for HAB prediction on Sacramento–San Joaquin Delta time-series data
- **Prompt engineering:** How to structure system prompts so GenAI risk communication is regulatory-compliant and avoids over- or under-warning
- **Device requirements:** Minimum hardware specs for on-device inference (RAM, neural accelerator availability)

---

## References

- Flutter vs. React Native 2026: https://www.techrev.us/blog/flutter-vs-react-native-2026/
- LiteRT (TFLite successor): https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/
- Edge AI inference: https://ai.google.dev/edge/litert/inference
- NNAPI acceleration: https://medium.com/softaai-blogs/nnapi-explained-the-ultimate-2025-guide-to-androids-ai-acceleration-33c0087f2ddf
- AI Edge Torch Generative API: https://developers.googleblog.com/ai-edge-torch-generative-api-for-custom-llms-on-device/
- NASA ML for HABs: https://www.nasa.gov/science-research/earth-science/nasa-developed-ai-could-help-track-harmful-algae/
- PostGIS for water quality: https://emerginginvestigators.org/articles/24-196
