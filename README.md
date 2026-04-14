# 🔍 LLM Research Agent

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production‑ready **Retrieval-Augmented Generation (RAG)** research assistant that searches the web, stores results in a vector database with persistent memory, and answers questions using a large language model.

## ✨ Features

- 🌐 **Web Search** – Uses Tavily API for clean, relevant search results
- 🧠 **Semantic Retrieval** – ChromaDB vector database finds information by meaning, not just keywords
- 💾 **Persistent Memory** – Research accumulates across runs (saved to disk locally)
- 🔒 **Context Isolation** – Each query's results are stored with metadata; retrieval filters by exact query to prevent topic pollution
- 🔄 **Graceful Fallback** – Handles rephrased questions by falling back to full collection search
- 🚀 **Fast LLM** – Groq's Llama 3.1 8B generates concise, well-structured answers
- 📊 **Stateful Pipeline** – Built with LangGraph for modular, observable execution

## 🏗️ Architecture


Each node in the LangGraph state machine:
- `search_node`: Fetches 5 web results via Tavily
- `store_node`: Stores results with UUIDs, query metadata, and timestamps
- `retrieve_node`: Retrieves top-3 relevant chunks (filtered by query, with fallback)
- `synthesize_node`: Generates final answer using Groq LLM

## 📋 Prerequisites

- Python 3.9 or higher
- API keys (free tiers available):
  - [Tavily](https://tavily.com) – for web search
  - [Groq](https://console.groq.com) – for LLM inference

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/llm-research-agent.git
cd llm-research-agent

2. Install dependencies
pip install -r requirements.txt

3. Set up environment variables
cp .env.example .env
# Edit .env and add your actual API keys

4. Run the agent
python src/research_agent.py

python research_agent.py
Enter your research query: Latest Cancer News
[search] Searching the web...
[search] Got 5 results
[store] Storing results in ChromaDB...
[store] Stored 5 new documents (total: 45)
[retrieve] Retrieving relevant chunks...
[retrieve] Retrieved 3 chunks
[synthesize] Generating final answer...

==================================================
FINAL ANSWER:
**Latest Cancer News - March 2026**

According to the latest information available from News-Medical, researchers at Mayo Clinic are investigating a novel radiopharmaceutical therapy for breast cancer. A clinical trial will be conducted to study a potent new treatment, aiming to improve outcomes for breast cancer patients.

**Key Details:**

- **Area of Research:** Breast Cancer
- **New Approach:** Radiopharmaceutical therapy
- **Clinical Trial:** Currently underway (as of March 13, 2026)
- **Objective:** To evaluate the efficacy and safety of the new treatment

We suggest staying updated on this emerging treatment and any future developments, as more information becomes available.
==================================================
