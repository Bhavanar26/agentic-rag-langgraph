# Agentic RAG with LangGraph

A question-answering system over PDF lecture notes. An agent decides whether a question needs the documents, checks whether the retrieved text is relevant, and rewrites the search query when it isn't. It is compared against a plain RAG baseline.

## Pipeline
PDFs -> chunks (800 chars, 100 overlap) -> MiniLM embeddings -> FAISS index

question -> route (retrieve / direct) -> retrieve (k=6) -> grade relevance -> rewrite query and retry (max 2) -> generate answer

## Stack
Python, LangChain, LangGraph, FAISS, sentence-transformers (all-MiniLM-L6-v2), Hugging Face Transformers, Qwen2.5-3B-Instruct (fp16, Colab T4)

## Evaluation
12 hand-written questions: 6 from the documents, 3 general knowledge, 3 not answerable from the documents. Scored by keyword match.

| | Plain RAG | Agentic RAG |
|---|---|---|
| Document questions | 5/6 | 6/6 |
| General questions | 1/3 | 3/3 |
| Unanswerable | 3/3 | 3/3 |
| Overall | 9/12 | 12/12 |

Routing was correct on 9/9 document and general questions.

## Findings
- The router initially sent document questions to the direct path because it did not know what the documents contained. Adding a short topic description and defaulting to retrieval when unsure fixed this.
- Query rewriting recovered one answer (black hat hackers) that plain retrieval missed, at the cost of extra LLM calls.
- Plain RAG scores low on general questions because it is restricted to the document context. This reflects design, not weak retrieval.
- Test labels were corrected after error analysis: one question was moved to unanswerable (not in the corpus), one ambiguous question was removed (the PDFs describe two different "first steps"), and some keyword lists were widened.

## Limitations
- Only 12 questions, written by me. The router prompt and labels were tuned on the same questions, so results are optimistic and not a held-out benchmark.
- Keyword matching is crude and can mis-score correct answers.
- The on-topic Canada question was routed to the direct path, though both systems refused safely.
- A 3B model is a weak router and grader. Future work: a larger model, a reranker, hybrid search, or retrieval-score-based routing.

## Files
- agentic_rag.ipynb - full notebook
- eval_final.csv - per-question results with answers
- The source PDFs are not included.
