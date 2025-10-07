# 🤖 Agentic Mobile Recommender
> *An Agentic AI Demo for Conversational, Context-Aware Mobile & Accessories Recommendations*

---

## 🧩 Executive Problem Statement
### **Beyond Static Recommendations — Toward Agentic, Conversational Retail AI**

Traditional recommender systems—used across e-commerce and telecom—often rely on **static filters** or **rule-based logic**.  
They break down when customers:

- Change preferences dynamically (budget, brand, camera, or gaming priorities)  
- Expect **interactive, human-like** shopping experiences  
- Need **context-aware bundling** (e.g., suggest MagSafe Charger after iPhone 15)

Retailers need a **real-time adaptive recommender** that reasons like a **personal sales expert** — dynamic, conversational, and intelligent.

---

## ⚙️ Solution Overview — *Agentic Mobile Recommender*

A lightweight **FastAPI-based Agentic AI system** powered by **OpenAI GPT-4o-mini** and **text embeddings**.  
It combines **semantic understanding**, **tool calling**, and **domain knowledge** to offer personalized, conversational recommendations.

| Layer | Description | Technologies |
|:------|:-------------|:-------------|
| 🗨️ **User Interaction** | Chat-style interface where users describe their needs in natural language | FastAPI + Pydantic |
| 🧠 **Agentic Brain** | LLM with function-calling ability for reasoning and dynamic tool use | GPT-4o-mini |
| 🧰 **Domain Tools** | `search_catalog()` for phones/accessories + `accessory_bundle()` for compatibility suggestions | Python, custom tool schema |
| 🧮 **Vector Intelligence** | Embedding-based semantic similarity search | OpenAI `text-embedding-3-small` + NumPy |
| 💾 **Session Memory** | Maintains user context (budget, brand affinity, preferences) | In-memory session store |

---

## 💡 Demo Highlights

### 🔍 **Natural Query Understanding**
> “I need an Android phone under $900 with great camera and battery.”  
→ Returns **Pixel 9** or **Galaxy S24**, ranked by semantic similarity, preference weights, and budget filters.

### 🎯 **Intelligent Bundling**
After selecting **Pixel 9**, the system automatically recommends **a compatible case + fast charger**, respecting budget constraints.

### 💬 **Adaptive Dialogue**
If constraints are too strict:  
> “No phones found under $600 with 50 MP camera. Shall I expand the price range?”

---

## 🌐 Why It Matters

| Benefit | Business Impact |
|:---------|:----------------|
| 💬 **Conversational Commerce** | Enables natural, chat-based product discovery that reduces drop-offs |
| 🧩 **Dynamic Reasoning** | Adjusts to evolving customer intent in real time |
| 🔁 **Reusable Design** | Applicable to telecom, insurance, retail, and travel ecosystems |
| ⚡ **Lightweight Deployment** | FastAPI microservice pattern deployable on enterprise AI infra |
| 🔒 **Secure & Modular** | Uses tool schemas for controlled reasoning and API safety |

---

## 🚀 Vision — *From Recommender to AI Shopping Companion*

| Today | Tomorrow |
|:------|:----------|
| 🛒 Agentic recommender for mobiles & accessories | 🌍 Omni-channel **AI sales companion** that evolves with user behavior |
| 🤝 Simple semantic search + reasoning | 🧭 Full **Agentic AI mesh** connecting LLM reasoning, memory, and personalization |
| 📱 Works with phones and add-ons | 🧩 Extendable to telco plans, gadgets, and digital services |

---

## 🧠 Tech Stack Summary

```bash
Framework: FastAPI
Language: Python 3.10+
LLM Model: gpt-4o-mini
Embedding Model: text-embedding-3-small
Core Packages: openai, numpy, pydantic, uvicorn


```To kickstart

python -m venv venv 

source venv/bin/activate

pip install poetry


# Spin the backend services

export OPENAI_API_KEY=your_key
uvicorn app:app --reload --port 80001

# Test the agentic recommendation services 

python test_recommendation.py