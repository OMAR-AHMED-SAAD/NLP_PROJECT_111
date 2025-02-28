# NLP_PROJECT_111
# Project Plan

## **Goal Task**
Perform **topic modeling** on YouTube transcripts using unsupervised learning to extract meaningful topics and use them for improved indexing in a RAG system or summarization.

---

## **Methods**
###   1 Latent Dirichlet Allocation (LDA)
- Assumes each transcript is a **mixture of multiple topics**.
- Each word is assigned to a topic based on probability distributions.
- Requires **preprocessing**:  
  - **Stopword removal**  
  - **Lemmatization**  

###  2 Non-negative Matrix Factorization (NMF)
- Requires **TF-IDF transformation** to represent word importance.
- Decomposes the document-word matrix into topic-based components.
###  3 BERTopic (BERT + Topic Modeling)
-  Uses BERT embeddings + clustering (e.g., UMAP + HDBSCAN).

---

## **Labeling Topics**
- Extract **top words per topic** from LDA/NMF results.
- Identify **most frequently repeated words** per topic.
- Use these words to **infer topic labels manually or automatically**.

---

## **Potential Uses After Topic Modeling**
### 1. Enhancing RAG Indexing
1. **Index similar topics together**:
   - Add **topic labels** to the transcript metadata.
   - Include a **topic distribution vector** alongside the **base embedding**.
  
2. **Improve query retrieval**:
   - Filter and retrieve documents based on dominant topics before performing similarity search.

### 2.  Better Summarization
- Summarize **similar topics together** for more structured and richer summaries.
- Cover a **wider range of information** by grouping transcripts under the same topic.

---
