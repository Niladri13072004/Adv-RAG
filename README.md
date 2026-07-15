# Advanced RAG System & Evaluation Dashboard

A state-of-the-art **Advanced Retrieval-Augmented Generation (RAG)** pipeline and interactive evaluation dashboard. The system integrates advanced retrieval, query expansion, self-correction, and model routing strategies to deliver highly accurate answers, accompanied by a comprehensive Gradio-based metrics dashboard to benchmark and inspect system performance.

---

## 🚀 Key Features

### 1. Advanced RAG Pipeline (`main.py`)
- **Dynamic Query Routing**: Automatically routes general vs. specific queries to specialized models.
- **Query Rewriting & Multi-Query**: Rewrites conversational context into standalone search queries and performs multi-query expansion to maximize retrieval recall.
- **Parent Document Retrieval**: Retrieves specific dense chunks but feeds the larger parent document context to the LLM to avoid context fragmentation.
- **Context Compression**: Compresses retrieved passages into high-density context notes to save context window length and minimize token usage.
- **Cross-Encoder Reranking**: Re-scores initial retrievals using a Cross-Encoder to ensure the most relevant chunks are ranked highest.
- **Self-Reflection & Verification**: Employs self-evaluation loops to check answers against retrieved facts, preventing hallucinations and applying self-correction when necessary.
- **Multi-Format Document Ingestion**: Supports uploading and parsing `.pdf` and `.docx` documents dynamically into a local Chroma vector database.

### 2. Interactive Evaluation Dashboard (`dashboard.py`)
- **Visual Analytics**: Interactive Plotly plots showing accuracy by category, response times distribution, judge evaluations, and latency.
- **Comprehensive Benchmarking Metrics**:
  - **Retrieval Metrics**: Recall@K, MRR (Mean Reciprocal Rank), Keyword Coverage.
  - **Generation Metrics**: Strict correctness/accuracy, average response time.
  - **LLM Judge Ratings**: Average ratings for Answer Accuracy, Completeness, and Relevance (scored out of 10).
- **Per-Question Inspection**: Drill down into individual evaluation runs to inspect the user's question, expected vs. actual answers, categories, response time, and the exact retrieved chunks used.

---

## 🛠️ Technology Stack

- **Frameworks**: [LangChain](https://github.com/langchain-ai/langchain), [Gradio](https://github.com/gradio-app/gradio)
- **Vector Database**: [ChromaDB](https://github.com/chroma-core/chroma)
- **Embeddings**: [HuggingFace Embeddings](https://huggingface.co/docs/hub/spaces-sdks-python) (sentence-transformers)
- **Reranker**: Sentence-Transformers Cross-Encoder
- **LLM Providers**: Google Gemini API, Groq Cloud, OpenRouter, and local Ollama instances
- **Visualization**: Plotly, Pandas, TailwindCSS (for metric cards)

---

## 📦 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Set Up Virtual Environment
Create and activate a virtual environment to manage project dependencies:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the `.env.example` file to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```
Open `.env` and configure:
- `OPENROUTER_API_KEY`: API key for OpenRouter.
- `GEMINI_API_KEY`: API key for Google Gemini.
- `HF_TOKEN`: HuggingFace token for downloading model components.
- `GROQ_API_KEY`: Groq Cloud API key.

---

## 🖥️ Running the Applications

### 💬 Chatbot & Ingestion App
Start the main Gradio interface to ingest documents and chat with the Advanced RAG agent:
```bash
python main.py
```
Open the local URL displayed in the terminal (usually `http://127.0.0.1:7860`).

### 📊 Evaluation Dashboard
Start the Gradio dashboard to view and analyze evaluation results loaded from `evaluation_results.json`:
```bash
python dashboard.py
```
Open the local URL displayed in the terminal (usually `http://127.0.0.1:7861`). Click the **"Load Evaluation"** button in the UI to import and plot the metrics.

---

## 📂 Project Structure

```
├── src/
│   └── config.py        # Environment & model settings loader
├── .env.example         # Template for environment secrets
├── .gitignore           # Excludes virtual env, API keys, caches, and local databases
├── dashboard.py         # Gradio application for evaluating RAG metrics
├── main.py              # Main RAG agent pipeline and chatbot UI
├── requirements.txt     # Python dependencies list
└── LICENSE              # Open-source license (MIT)
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
