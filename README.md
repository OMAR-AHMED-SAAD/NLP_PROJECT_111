# RAG System

This project implements a Retrieval-Augmented Generation (RAG) system, designed to provide accurate and context-aware responses. It's split into a user-friendly Gradio frontend and a flexible Python backend.

## Features

* **Interactive Frontend:** A simple Gradio interface for easy interaction with the RAG system.
* **Flexible Backend:** Supports multiple large language models (LLMs) and prompt strategies.
* **Pre-loaded Database:** Comes pre-configured with the SQuAD 1.1 dataset for immediate use.
* **Context Logging:** Automatically logs retrieved contexts for each query, aiding in debugging and understanding.

---

## Getting Started

Follow these instructions to get your RAG system up and running.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/OMAR-AHMED-SAAD/NLP_PROJECT_111.git](https://github.com/OMAR-AHMED-SAAD/NLP_PROJECT_111.git)
    cd your-rag-project
    ```
2.  **Install dependencies:**
    * Navigate to the `frontend` directory and install its dependencies:
        ```bash
        cd frontend
        pip install -r requirements.txt
        cd ..
        ```
    * Navigate to the `backend` directory and install its dependencies:
        ```bash
        cd backend
        pip install -r requirements.txt
        cd ..
        ```

---

## Running the Application

This RAG system consists of two independent components: the frontend and the backend. You'll need to run both separately.

### 1. Running the Frontend

The frontend provides the graphical user interface for interacting with the RAG system.

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Run the Gradio application:
    ```bash
    python app.py
    ```
    This will typically open a local URL in your browser where you can access the Gradio interface.

### 2. Running the Backend

The backend handles the RAG logic, including model inference and retrieval.

1.  **Configure Environment Variables:**
    * Rename `backend/env.example` to `backend/.env`:
        ```bash
        cd backend
        cp env.example .env
        ```
    * Open the `.env` file in a text editor. Here, you can configure your chat model and system prompt.

2.  **Chat Model Options:**
    Modify the `MODEL_CHAT_CLASS` and `MODEL_CHAT_NAME` variables in `.env` based on your desired LLM:

    * **Ollama (Local Models):**
        To use Ollama, set `MODEL_CHAT_CLASS` to `"ollama"` and `MODEL_CHAT_NAME` to the desired model you have downloaded.
        ```ini
        MODEL_CHAT_CLASS="ollama"
        MODEL_CHAT_NAME="phi-4" # Or any other Ollama model (e.g., "llama2", "mistral")
        ```
        Make sure you have [Ollama](https://ollama.com/) installed and the desired model pulled locally (e.g., `ollama pull phi-4`).

    * **Google Gemini (API Key Required):**
        To use Google Gemini, set `MODEL_CHAT_CLASS` to `"google"` and `MODEL_CHAT_NAME` to your preferred Gemini model. You'll also need to provide your `GOOGLE_API_KEY`.
        ```ini
        MODEL_CHAT_CLASS="google"
        MODEL_CHAT_NAME="gemini-2.0-flash"
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY" # Replace with your actual Google API Key
        ```
        **How to get a Google API Key for Gemini:**
        1.  Go to the Google AI Studio website: [https://aistudio.google.com/](https://aistudio.google.com/)
        2.  Sign in with your Google account.
        3.  On the left-hand navigation, click "Get API Key" or "Create API Key" if you don't have one.
        4.  Copy the generated API key and paste it into your `.env` file for the `GOOGLE_API_KEY` variable.

    * **Finetuned Encoder (Hardcoded Model):**
        If you want to use the hardcoded finetuned DistilBERT model, set `MODEL_CHAT_CLASS` to `"finetuned_encoder"`. The `MODEL_CHAT_NAME` for this option is internally set to `"distilled-bert-finetuned"`.
        ```ini
        MODEL_CHAT_CLASS="finetuned_encoder"
        MODEL_CHAT_NAME="distilled-bert-finetuned" # This model is hardcoded and cannot be changed here
        ```

3.  **System Prompt Options:**
    Choose your desired prompting strategy by setting the `SYSTEM_PROMPT` variable in `.env`:

    * **Chain-of-Thought Prompting:**
        ```ini
        SYSTEM_PROMPT="cot_prompt"
        ```
    * **Standard System Prompt:**
        ```ini
        SYSTEM_PROMPT="system_prompt"
        ```
    * **Zero-Shot Prompting:**
        ```ini
        SYSTEM_PROMPT="zero_shot_prompt"
        ```

4.  **Run the Backend Server:**
    Once your `.env` file is configured, start the backend server from the `backend` directory:
    ```bash
    python -m server
    ```
    You should see a log message indicating successful initialization:
    ```
    [INFO] server: RAG model initialized successfully
    ```
    This confirms that the RAG model is ready to process queries.

---

## Usage

1.  Ensure both the frontend and backend servers are running.
2.  Access the Gradio frontend in your web browser (usually `http://127.0.0.1:7860`).
3.  Type your questions into the input field and press Enter or click the submit button. The RAG system will retrieve relevant information from its loaded database and generate a response.

---

## Logs

A `logs` folder is created in the `backend` directory. Inside, you'll find files named under the timestamp of each question you ask (e.g., `2023-10-27_15-30-00.txt`). These files contain the contexts retrieved from the database for that specific query, which can be useful for debugging and understanding the retrieval process.