# Aurora Bay RAG System (Challenge 5)
BigQuery Embeddings + Vector Search + Gemini + FastAPI

---

## 1) What this project is
This repository contains my implementation of a Retrieval-Augmented Generation (RAG) chatbot for the fictional town of Aurora Bay, Alaska.

The chatbot answers user questions using a proprietary FAQ dataset stored in BigQuery instead of relying on the LLM to guess.

Key idea:
Retrieve the most relevant FAQ records using vector search, then generate an answer grounded in that retrieved content.

---

## 2) Dataset
Source CSV:
gs://labs.roitraining.com/aurora-bay-faqs/aurora-bay-faqs.csv

Stored in BigQuery as:
- Raw FAQ table
- Embedded FAQ table

Each record contains:
- Question
- Answer
- Embedding vector (ARRAY<FLOAT64>)

---

## 3) Architecture Flow

1. User → FastAPI /chat
2. FastAPI → RAG Service
3. Embed user question
4. BigQuery VECTOR_SEARCH retrieves top-K
5. Build context
6. Gemini generates grounded answer
7. Return response
8. Optional logging to BigQuery

This design reduces hallucination by grounding responses in FAQ data.

---

## 4) Repository Structure

- Challenge_5.ipynb
- main.py
- rag_service.py
- requirements.txt
- Dockerfile

---

## 5) RAG Workflow

### Step A — Load FAQ to BigQuery
CSV imported using load job.

### Step B — Generate embeddings
Vertex AI embedding model used.

### Step C — Vector Search
BigQuery VECTOR_SEARCH with cosine similarity.

### Step D — Grounded Generation
Gemini instructed to answer only using retrieved FAQ context.

---

## 6) Evaluation
Vertex AI EvalTask used for:
- Coherence
- Safety

---

## 7) Running the API

uvicorn main:app --host 0.0.0.0 --port 8081

## 8) Deployment

The service can be containerized and deployed to Cloud Run.
Environment variables are used for configuration such as:

BigQuery table IDs
Gemini model name
log table ID
