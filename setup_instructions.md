# Environment & Local LLM Setup Instructions
## Project: Agentic RAG-Based AI Research Assistant
> **A Comprehensive Guide for College Evaluators & Students**

This document provides complete, step-by-step terminal instructions to set up an isolated Python environment, install all Natural Language Processing (NLP) dependencies, download the required language models, and configure **Ollama** to run the local **`llama3`** Large Language Model.

---

## 1. Prerequisites
- **Python:** Python 3.9, 3.10, or 3.11 installed on your system.
- **Git / Terminal:** PowerShell or Command Prompt on Windows; Bash or Zsh on macOS / Linux.
- **Ollama:** Installed locally on your operating system ([https://ollama.com](https://ollama.com)).

---

## 2. Step 1: Create and Activate a Python Virtual Environment (`venv`)

Creating an isolated virtual environment guarantees reproducible package versions without conflicting with system packages.

### Windows (PowerShell)
```powershell
# 1. Navigate to your project directory
cd d:\nlp

# 2. Create the virtual environment named 'venv'
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\activate
```

### Windows (Command Prompt - CMD)
```cmd
cd d:\nlp
python -m venv venv
venv\Scripts\activate.bat
```

### macOS / Linux (Terminal)
```bash
# 1. Navigate to your project directory
cd ~/nlp

# 2. Create the virtual environment named 'venv'
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate
```

*(Once activated, your terminal prompt will display `(venv)` at the beginning).*

---

## 3. Step 2: Create the `requirements.txt` File

Create a file named `requirements.txt` in your project root with the following contents:

```text
streamlit
langchain
langchain-community
langchain-ollama
chromadb
pypdf
sentence-transformers
textblob
bertopic
deep-translator
duckduckgo-search
pandas
jupyter
spacy
```

---

## 4. Step 3: Install Python Dependencies & spaCy Model

With your virtual environment active, run the following commands in your terminal:

```bash
# 1. Upgrade pip package installer
pip install --upgrade pip

# 2. Install all core project dependencies
pip install -r requirements.txt

# 3. Download the English NLP pipeline for spaCy (used for Information Extraction / NER)
python -m spacy download en_core_web_sm
```

---

## 5. Step 4: Install Ollama and Pull the `llama3` Model

This project is built to execute **100% locally and offline** using Ollama. No cloud API keys (such as OpenAI or AWS Bedrock) are required.

### 5.1 Verify Ollama Installation
Open your terminal and confirm Ollama is installed:
```bash
ollama --version
```
*(If Ollama is not yet installed, download the installer for your OS from [https://ollama.com/download](https://ollama.com/download) and launch it).*

### 5.2 Pull the `llama3` Model
Download the official Meta LLaMA 3 model:
```bash
ollama pull llama3
```

### 5.3 Verify the Local Model Registry
Verify that `llama3` is ready on your machine:
```bash
ollama list
```
You should see `llama3:latest` listed in the output table.

---

## 6. Step 5: Running the Project Deliverables

### Option A: Launch the Interactive Pedagogical Jupyter Notebook
For viva evaluation and step-by-step syllabus practical demonstration:
```bash
jupyter notebook Agentic_RAG_Project.ipynb
```
Run each cell sequentially from Cell 1 through Cell 14.

### Option B: Launch the Interactive Streamlit Web Application
For the live frontend demonstration to professors:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser, upload an academic PDF, and interact with the ReAct Agent in real time.
