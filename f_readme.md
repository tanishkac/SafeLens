https://openrouter.ai/workspaces/default/keys
https://aistudio.google.com/api-keys

-----------------------------------------------------

Confidence Score Math Formula & Calculation
The safety engine uses a multi-tier mathematical confidence system defined in 

ConfidenceCalibrator
:

A. Token Log-Odds & Softmax Formulation
When token log-probabilities are returned by the LLM provider, the raw logit $z$ is computed from the binary decision tokens: $$z = \ln p(\text{"true"}) - \ln p(\text{"false"})$$

The uncalibrated raw probability is: $$P_{\text{raw}} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

B. Temperature & Platt Scaling Calibration
To correct for LLM overconfidence on safety classifications, Platt scaling with temperature scaling is applied: $$P_{\text{calibrated}} = \sigma\left(\frac{z}{T} + b\right) = \frac{1}{1 + \exp\left(-\left(\frac{z}{T} + b\right)\right)}$$

$T$: Temperature parameter ($T = 1.35$ by default)
$b$: Platt bias intercept ($b = 0.0$ by default)
C. Fallback & Prompt Bounds Clamping
When using third-party free router endpoints where logprobs are omitted upstream, the calibrator performs bound normalization: $$C_{\text{final}} = \min(1.0, \max(0.0, C_{\text{prompt}}))$$

D. Decision Classification Threshold
A segment is classified as harmful if: $$\text{is_harmful} = \text{True} \iff \Big(\text{LLM flag} = \text{True}\Big) \lor \Big(\text{len}(\text{categories}) > 0 ;\land; P_{\text{calibrated}} \ge 0.50\Big)$$

In your run, segments 1, 2, and 5 satisfied $P \ge 0.92 \gg 0.50$ with matching harm categories, properly qualifying them as verified harmful events.

-----------------------------------------------------

-----------------------------------------------------
1. Mathematical Calibration & Core Calculation


web/app/orchestration/calibrator.py
:
Core Implementation: Implements the 

ConfidenceCalibrator
 class.
Mathematical Formula: $$\text{Log-odds logit: } z = \ln p(\text{"true"}) - \ln p(\text{"false"})$$ $$\text{Raw Probability: } P_{\text{raw}} = \sigma(z) = \frac{1}{1 + e^{-z}}$$ $$\text{Platt / Temperature Scaled: } P_{\text{calibrated}} = \sigma\left(\frac{z}{T} + b\right) = \frac{1}{1 + e^{-(z/T + b)}}$$
Extracts token logprobs (_logprobs, top_logprobs) from LLM outputs and applies fallback heuristics if token probabilities are unavailable.
2. Segment Analysis & Suspicion Scoring


web/app/orchestration/segment_analyzer.py
:
Integrates ConfidenceCalibrator with multi-modal evidence (vision caption, audio transcript, OCR).
Calculates multi-modal factor weights (factor_weights: visual %, audio %, text %).
Normalizes segment confidence scores ($0 - 100%$).


web/app/planning/llm_planner.py
:
Calculates suspicion confidence (suspicion_conf) for each segment using LLM prompt evaluation and keyword scoring against SUSPICION_LLM_CONF_THRESHOLD.
3. Aggregation, Storage & API Serving


web/services/reporting.py
:
Aggregates overall video confidence: $$\text{overall_confidence_score} = \text{round}\left(\frac{1}{N}\sum_{i=1}^{N} \text{confidence}_i\right)$$


web/services/persistence.py
:
Saves segment confidence_score and video overall_confidence_score into the SQLite database.


web/routers/videos.py
:
Serves confidence scores in /api/analyze/{video_id}/results and /api/user/videos.
4. Frontend Clustering & UI Presentation


web/frontend/src/utils/clustering.ts
:
Clusters adjacent events and computes cluster maxConfidence and avgConfidence.


web/frontend/src/app/[videoId]/page.tsx
:
Normalizes $0-100$ backend confidence into $0.0-1.0$ format for the UI state.


web/frontend/src/components/VidstackPlayer.tsx
 & 

InspectorPanel.tsx
:
Color-codes the video timeline and inspector tabs based on confidence thresholds (e.g. $\ge 90%$ Red, $\ge 70%$ Orange).
-----------------------------------------------------

# SafeLens: AI-Powered Hateful & Harmful Video Moderation

SafeLens is an AI-powered multimodal video moderation system that detects and flags harmful or hateful content in videos using video frame extraction, OCR, Whisper audio transcription, and LLM reasoning.

This version is configured to **run completely out-of-the-box** without any Google Auth or external OIDC server dependencies.

---

## 📋 Prerequisites

Make sure you have the following installed on your machine:

1. **Python 3.10+ / 3.11+**: [Download Python](https://www.python.org/downloads/) *(During installation, make sure to check **"Add Python to PATH"**)*.
2. **Node.js (v18+ or v20+ LTS)**: [Download Node.js](https://nodejs.org/).
3. **pnpm** package manager:
   ```bash
   npm install -g pnpm
   ```
4. **Git**: [Download Git](https://git-scm.com/).
5. **OpenRouter API Key** *(for LLM analysis)*:
   - Create a free account at [https://openrouter.ai/](https://openrouter.ai/).
   - Generate an API key at [https://openrouter.ai/keys](https://openrouter.ai/keys).

---

## 🚀 Quick Start Guide (Step-by-Step)

### Step 1: Clone the Repository

Open your terminal (PowerShell, Command Prompt, or Bash) and clone the repository:

```bash
git clone <YOUR_REPO_URL>
cd SafeLens
```

---

### Step 2: Set Up and Start the Backend (FastAPI)

1. **Navigate to the `web` folder**:
   ```bash
   cd web
   ```

2. **Create and activate a Python virtual environment**:
   - **On Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
     *(If you encounter a script execution policy error on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*
   - **On Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment (`web/.env`)**:
   Create a `.env` file inside the `web/` folder:
   - **On Windows (PowerShell)**:
     ```powershell
     Copy-Item .env.example .env
     ```
   - **On Linux / macOS**:
     ```bash
     cp .env.example .env
     ```

   Open `web/.env` in your text editor and ensure:
   - **Database**: `DATABASE_URL=sqlite:///./safelens.db` *(Uses local SQLite out-of-the-box, no Postgres required)*.
   - **OpenRouter Key**: Set your API key on line 11:
     ```env
     OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here
     ```

5. **Start the Backend Server**:
   ```bash
   python server.py
   ```
   You will see:
   ```text
   INFO: Starting FastAPI video upload server...
   INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```
   *Keep this terminal running!*

---

### Step 3: Set Up and Start the Frontend (Next.js)

1. **Open a SECOND terminal window** and navigate to `web/frontend`:
   ```bash
   cd SafeLens/web/frontend
   ```

2. **Configure Frontend Environment (`web/frontend/.env`)**:
   Create a `.env` file inside the `web/frontend/` folder:
   - **On Windows (PowerShell)**:
     ```powershell
     Copy-Item .env.example .env
     ```
   - **On Linux / macOS**:
     ```bash
     cp .env.example .env
     ```
   *(Ensure `BACKEND_URL="http://localhost:8000"` is present in `web/frontend/.env`)*.

3. **Install Frontend Dependencies**:
   ```bash
   pnpm install --ignore-scripts
   ```
   *(or `pnpm install` then run `pnpm approve-builds`)*.

4. **Start the Frontend Development Server**:
   ```bash
   pnpm dev
   ```

---

### Step 4: Open and Use the Application

1. Open your web browser and go to:
   👉 **[http://localhost:3000](http://localhost:3000)**

2. You are automatically signed in as **SafeLens User** without any login prompts.

3. **Analyze a Video**:
   - Select a model from the model selector dropdown (e.g. `google/gemini-3-flash-preview` or `google/gemini-2.5-flash`).
   - Drag & drop a video file (`.mp4`, `.mov`, `.avi`, `.webm`, etc.) or paste a supported video URL.
   - The analysis pipeline will automatically:
     1. Extract key video frames and perform OCR.
     2. Transcribe audio using Whisper with word-level timestamps.
     3. Send multimodal context to the policy LLM via OpenRouter.
     4. Display an interactive inspection timeline with flagged harmful events, categories, confidence scores, and transcript evidence.

---

## 🛠️ Summary of Fixes & Out-of-the-Box Features

- **No Auth / OIDC Server Required**: Bypassed external Google/OIDC dependency. The app provides a persistent local user session (`default-user`).
- **Zero Database Server Setup**: Uses SQLite (`safelens.db`) by default with auto-provisioning tables.
- **Cross-Platform Compatibility**: Cleaned `requirements.txt` to run seamlessly on Windows, macOS, and Linux without platform-locked CUDA wheels.
- **Seamless Frontend Routing**: Auto-fallbacks for Next.js API routes preventing `undefined/api` errors.

---

## 🐳 Alternative: Running with Docker

If you have Docker Desktop installed, you can also run the entire application via Docker Compose:

```bash
cd SafeLens
docker compose --profile core up --build
```
Access the application at **[http://localhost:3000](http://localhost:3000)**.
