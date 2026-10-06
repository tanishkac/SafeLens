# SafeLens: Technical Architecture & System Explanation Guide

This guide provides a comprehensive, step-by-step breakdown of how **SafeLens** works. It is structured so that you can explain each module, file, and modality clearly and confidently.

---

## 1. High-Level Overview

**SafeLens** is an AI-powered multimodal video safety and content moderation system. 
Rather than analyzing a video using just one data type, SafeLens analyzes **three synchronized modalities**:

1. **Audio**: Speech-to-text transcription with precise word-level timestamps.
2. **Vision (VLM & ViT)**: Scene boundary detection (ViT) and detailed frame-by-frame visual descriptions (Vision-Language Model).
3. **Text (OCR)**: On-screen text, banners, symbols, signs, and subtitles.

All three sources of evidence are fused together and fed into a **Reasoning LLM (Large Language Model)** to evaluate harm context, categorize violations (Hate Symbols, Extremism, Violence, Harassment), calculate confidence scores, and attribute modality factor weights.

```
                  ┌───────────────────────────────────────────────┐
                  │              Input Video (.mp4)               │
                  └───────────────────────┬───────────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
    [ AUDIO STREAM ]              [ VIDEO FRAMES ]              [ SCENE TRANSITIONS ]
            │                             │                             │
            ▼                             ▼                             ▼
  WhisperX Transcription           EasyOCR & VLM ViT Scene Boundary
  & Word-Level Alignment          (Text & Frame Captions)        Segmentation
            │                             │                             │
            └─────────────────────────────┼─────────────────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │    Multimodal Fusion & LLM Engine   │
                       │    (Contextual Harm Reasoning)      │
                       └──────────────────┬──────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │     Structured Safety Report &      │
                       │          Database Storage           │
                       └─────────────────────────────────────┘
```

---

## 2. Modality 1: Audio Transcription & Alignment

### 📍 Where the Code Exists
* **Core Tool**: [`web/tools/transcription.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/tools/transcription.py) (Function: `transcribe_whole_video`)
* **Service Wrapper**: [`web/services/transcript.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/services/transcript.py) (Function: `load_transcript`)

### 🛠️ What the Code Does
* Extracts the raw audio track from the video file.
* Transcribes all spoken speech into text.
* Generates exact, millisecond-level start and end timestamps for **every single spoken word**.
* Stores the transcript in the SQLite database and caches it to `transcript.json`.

### ⚙️ How It Works (Step-by-Step)
1. **Audio Extraction via FFmpeg**:
   WhisperX uses a bundled `ffmpeg` binary to decode the video into a 16kHz mono PCM WAV stream.
2. **Voice Activity Detection (VAD)**:
   It uses **Pyannote Audio** to segment speech from background music, silence, or noise.
3. **Automatic Speech Recognition (ASR)**:
   It runs **Faster-Whisper (`large-v2`)** to convert speech audio segments into raw text.
4. **Phonetic Word Alignment**:
   It feeds the text and audio into a **Wav2Vec2** acoustic model to align each character phonetically with its exact timestamp in the video (`(word, timestamp)` tuples).

---

## 3. Modality 2: Visual Understanding (ViT & VLM)

Visual understanding in SafeLens operates in two phases: **Temporal Scene Segmentation** and **Semantic Frame Captioning**.

### A. Temporal Scene Boundary Detection (ViT)
* **📍 Where the Code Exists**: [`web/app/orchestration/segmentation.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/app/orchestration/segmentation.py) (Function: `find_visual_boundaries_in_span`)
* **🛠️ What it Does**: Finds where camera cuts, scene changes, or visual shifts happen so the video is divided into logical, coherent time segments.
* **⚙️ How it Works**:
  * Extracts frames across the timeline at regular intervals.
  * Passes frames through a **Vision Transformer (`google/vit-base-patch16-224`)** to extract dense visual embedding vectors.
  * Calculates cosine similarity between consecutive frame embeddings. When the similarity drops below a threshold (`SEG_SCENE_THRESHOLD=0.85`), a scene cut boundary is marked.

### B. Frame Vision-Language Captioning (VLM)
* **📍 Where the Code Exists**: [`web/tools/image_classifier.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/tools/image_classifier.py) (Function: `classify_image`)
* **🛠️ What it Does**: Describes what is visually happening in each sampled frame (people, actions, symbols, clothing, weapons, vehicles, environment).
* **⚙️ How it Works**:
  * Loads the sampled frame from disk and encodes it into a JPEG base64 Data URL.
  * Sends a multimodal request to the Vision-Language Model (`google/gemini-2.5-flash` or `Qwen2.5-VL` via OpenRouter API).
  * Prompts the VLM to objectively detail all visible elements, flags, insignia, and actions in the frame.

---

## 4. Modality 3: Optical Character Recognition (OCR)

### 📍 Where the Code Exists
* **Core Tool**: [`web/tools/ocr.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/tools/ocr.py) (Function: `run_ocr`)
* **Frame Collector**: [`web/app/orchestration/segment_analyzer.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/app/orchestration/segment_analyzer.py) (Function: `_ocr_frame`)

### 🛠️ What the Code Does
* Reads written or printed text embedded in the video frames (posters, banners, signs, vehicle markings, clothing text, protest signs, subtitles).

### ⚙️ How It Works (Step-by-Step)
1. Initializes an **EasyOCR Reader** running on CPU/GPU.
2. Uses the **CRAFT (Character Region Awareness for Text Detection)** deep learning model to find text bounding boxes in the image.
3. Uses a **CRNN (Convolutional Recurrent Neural Network)** with CTC loss to recognize characters and output cleaned text strings.
4. Associates the recognized text with the frame's timestamp (e.g. `[0.0s] Deutsche Waehlt`).

---

## 5. Multimodal Fusion & Decision Engine

### 📍 Where the Code Exists
* **Orchestrator**: [`web/app/orchestration/segment_analyzer.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/app/orchestration/segment_analyzer.py) (Function: `llm_decide`)
* **LLM Client Router**: [`web/tools/llm.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/tools/llm.py) (Class: `SafetyLLM` / `OpenRouterProvider`)
* **Pipeline Coordinator**: [`web/services/analysis_pipeline.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/services/analysis_pipeline.py)

### 🛠️ What the Code Does
* Fuses the synchronized **Audio Transcript**, **OCR Text**, and **Visual Captions** for each video segment into a structured prompt.
* Uses an advanced LLM (`Gemini 2.5 Flash`) to evaluate whether harmful, discriminatory, extremist, or violent material is present.
* Produces a structured JSON decision:
  * `is_harmful`: boolean (`true` / `false`)
  * `confidence`: score between 0% and 100%
  * `categories`: list of violated categories (e.g. `["Hate Symbols", "Extremism"]`)
  * `explanation`: natural language justification
  * `factor_weights`: breakdown of which modality contributed most (`visual: 90%`, `text: 10%`, `audio: 0%`)

---

## 6. Database Architecture & Schema

SafeLens uses **SQLAlchemy ORM** backed by an SQLite database (`safelens.db`).

### 📍 Where the Code Exists
* **Schema Definition**: [`web/database.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/database.py)
* **Persistence Logic**: [`web/services/persistence.py`](file:///d:/Sudeep/CODING%28From%20Asus%20Tuf%20Laptop%29/Visual%20studio%20code%20files/0%28RESUME%29/Major%20Project%20Prototypes/1_MVP/SafeLens/web/services/persistence.py)

### 🗄️ Database Tables & Relationships

```
 ┌──────────────┐       1:N       ┌──────────────┐       1:N       ┌──────────────────┐
 │   Account    │────────────────►│    Video     │────────────────►│   HarmfulEvent   │
 └──────────────┘                 └──────┬───────┘                 └────────┬─────────┘
                                         │                                  │ 1:N
                                     1:1 │                                  ├───────────────┐
                                         ▼                                  ▼               ▼
                                  ┌──────────────┐                 ┌────────────────┐ ┌──────────────┐
                                  │Transcription │                 │ VisualEvidence │ │AudioEvidence │
                                  └──────────────┘                 └───────┬────────┘ └──────────────┘
                                                                           │ 1:N
                                                                           ▼
                                                                   ┌────────────────┐
                                                                   │   ImageLabel   │
                                                                   └────────────────┘
```

### Table Breakdown:

| Table Name | Primary Purpose | Key Fields |
| :--- | :--- | :--- |
| **`accounts`** | User profile & session tracking. | `id`, `name`, `email`, `created_at` |
| **`videos`** | Master record for each uploaded video file. | `id`, `account_id`, `original_filename`, `file_path`, `analysis_status`, `safety_rating`, `summary` |
| **`transcriptions`** | Complete speech-to-text transcript & alignment. | `id`, `video_id`, `full_text`, `word_timestamps` (JSON) |
| **`analysis_runs`** | Execution audit logs for performance & cost tracking. | `id`, `video_id`, `model_used`, `tokens_prompt`, `tokens_completion`, `latency_ms` |
| **`harmful_events`** | Timestamped violations detected by the LLM. | `id`, `video_id`, `start_time`, `end_time`, `confidence_score`, `categories`, `factor_weights`, `explanation` |
| **`visual_evidence`** | OCR text linked to a specific harmful event. | `id`, `harmful_event_id`, `ocr_text` |
| **`image_labels`** | Individual visual captions and model labels. | `id`, `visual_evidence_id`, `label`, `category`, `confidence` |
| **`audio_evidence`** | Spoken snippet linked to a specific harmful event. | `id`, `harmful_event_id`, `transcript_snippet` |

---

## 7. Quick Summary for Presentation

When presenting to your teacher, you can summarize the pipeline in **4 key bullet points**:

1. **Audio (WhisperX + Pyannote + Wav2Vec2)**: Automatically separates speech from noise and generates word-level timestamps so we know *what* was said and *when*.
2. **Vision (ViT + VLM)**: Uses Vision Transformers to segment scenes at natural visual transitions, and sends sampled frames to a Vision-Language Model to describe the visual scene in detail.
3. **OCR (EasyOCR)**: Uses CRAFT and CRNN neural networks to extract on-screen banners, protest signs, and text overlays.
4. **Fusion & Reasoning (Gemini 2.5 Flash / SafetyLLM)**: Combines all three data streams into a single contextual analysis prompt to detect harms like hate symbols, extremism, or violence, providing human-readable explanations and modality factor weights.
