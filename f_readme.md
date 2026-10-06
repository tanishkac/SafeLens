https://openrouter.ai/workspaces/default/keys

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
