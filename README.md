# 🔎 Research Agent v6.0

A lightweight, intelligent research assistant that fetches, ranks, and explains recent AI papers from arXiv.

It runs fully in Streamlit and acts as a **"Universal Brain"** for research—you can plug in OpenAI, Google Gemini, or Groq to handle the reasoning, or run it entirely offline with a **Free Local Model**.

It produces ranked tables, citation impact scores, and **"Plain English" summaries** for the top papers.

🌐 **Try the Live App:** https://research-aiagent.streamlit.app/

---

## ⚾ The "Moneyball" Ranking Engine

We have completely overhauled how papers are ranked. We stopped asking LLMs to guess "Will this paper be famous?" (which turned out to be unreliable) and built a **Moneyball Algorithm**.

- **Hard Data (84% Weight):** The agent now queries Semantic Scholar to analyze the **"Author Velocity"** (recent citation momentum) of the paper's authors.
- **Soft Data (16% Weight):** The LLM analyzes the abstract for **"Market Fit"** (Is this a trending topic?) and **"Novelty."**

**The Result:** A **6x increase in precision@10**, meaning the papers at the top of your list are statistically much more likely to be influential.

### Note on New Papers

Papers less than 5 days old often lack citation data in Semantic Scholar. These are marked as **"Too new for impact score"** and ranked purely by their relevance to your query.

---

## ⚡ Groq Integration (Llama 3.3)

You can now use **Groq** as your intelligence provider. This allows you to run high-performance open-source models (like **Llama 3.3 70B**) with blazing fast inference speeds and **free API access**.

> **Note:** In Groq mode, the agent uses local embeddings for search (all-MiniLM-L6-v2) and sends only the filtered candidates to Groq for analysis.

---

## 🎯 Features

- **Semantic Search:** Finds papers conceptually related to your query, not just keyword matches.
- **Plain English Summaries:** Translates academic jargon into clear bullet points.
- **"Moneyball" Impact Scores:** Predicts 1-year citation impact using real author data.
- **Multi-Provider Support:** OpenAI, Gemini, Groq, or Local (No API Key).
- **Export to ZIP:** Download all data (JSONs, Markdown report) for offline use.

---

## 🗂️ Categories Supported

We capture all major Computer Science subcategories. You can select specific ones to narrow your search:

- Artificial Intelligence (cs.AI)
- Machine Learning (cs.LG)
- Human-Computer Interaction (cs.HC)
- Computation and Language (cs.CL)
- Computer Vision (cs.CV)
- Robotics (cs.RO)
- Information Retrieval (cs.IR)
- Neural and Evolutionary Computing (cs.NE)
- Software Engineering (cs.SE)
- Cryptography and Security (cs.CR)
- Data Structures and Algorithms (cs.DS)
- Databases (cs.DB)
- Social and Information Networks (cs.SI)
- Multimedia (cs.MM)
- Information Theory (cs.IT)
- Performance (cs.PF)
- Multiagent Systems (cs.MA)

---

## 📦 Installation (Local)

### 1. Clone the repository:

```bash
git clone https://github.com/nurtekinsavasai/arxiv-ai-agent-v2.git
cd arxiv-ai-agent-v2
```

### 2. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
# Ensure you install groq if it wasn't automatically picked up
pip install groq
```

### 4. Run the app:

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## 🛠️ For Developers

- **`app.py`:** The entire application logic resides here. It is a single-file Streamlit app designed for portability.

- **Modifying Prompt Logic:** Look for the **Moneyball scoring functions** and `scibert_classify_papers` in `app.py` to change how the AI judges papers.

- **Adding Providers:** The code uses a `LLMConfig` dataclass. To add a new provider (e.g., Anthropic), add the client initialization in `call_llm`. Retrieval embeddings always run on local SPECTER2/MiniLM regardless of provider — only classification and citation scoring are provider-specific.

---

## 📊 Comparison Table

| Option | Requirements | Best For |
|--------|-------------|----------|
| **OpenAI** | API Key | Highest quality summaries and narrative analysis. (GPT-5.2, etc.) |
| **Gemini** | API Key (Google AI Studio) | Speed and large context windows. (Gemini 3 Pro, etc.) |
| **Groq** | API Key (Free) | Blazing fast open-source models (Llama 3.3). |
| **Free Local** | None | Offline usage, zero cost. Uses heuristics instead of LLM analysis. |
