# TASK 1B Persona-Driven Document Intelligence System

## Input Specification

- **Documents**: A collection of 3–10 related PDFs
- **Persona**: Role definition including domain knowledge and focus areas 
- **Job-To-Be-Done**: A concrete task relevant to the persona 

## Approach

This system extracts the most relevant sections from documents tailored to a persona’s goal using a multi-step pipeline:

1. **Heading-Based Segmentation**:
   - Utilized a custom heading detection system (from Task 1A) that uses positional, text and font features fed into a lightbgm model to classify text and headings
   - Text is segmented between consecutive detected headers for finer-grained analysis

2. **Semantic Representation & Ranking**:
   - Model: [`BAAI/bge-small-en`](https://huggingface.co/BAAI/bge-small-en)
   - Built a query dynamically using the provided persona and task:
     ```
     "Represent this query for retrieving supporting documents: 
     As a [persona], you need to [job]. Focus on actionable or instructional content."
     ```
   - Encoded both the query and document sections into embeddings using the bge-small-en model
   - Computed cosine similarity of embedded query and each embedded section to rank sections by relevance

3. **Selection & Prioritization**:
   - Selected top-ranked sections across documents
   - Assigned an `importance_rank` to each
   - Generated refined sub-sections for detailed output
  
4. **Why BGE and Key Benefits**:
   - The `BAAI/bge-small-en` model is a lightweight, instruction tuned embedding model designed for semantic search, ranking, and retrieval tasks.
   - It performs exceptionally well in **zero-shot** settings, where the model must generalize to new domains and tasks without fine tuning.
   - Importantly, BGE is trained to accept **query-response style inputs**: this means it understands and leverages prompts like user questions, tasks, or instructions, making it ideal for persona driven scenarios.
   - ** Why we chose it: **:
     - **Lightweight** (~120MB): Easily fits within the 1GB system limit.
     - **Instruction-aware & Query-Response Friendly**: Accepts prompts like `"As a financial analyst, summarize..."` and embeds them meaningfully for retrieval.
     - **High-quality semantic embeddings**: Captures deep contextual relevance across diverse topics and writing styles.
     - **Fast CPU inference**: Enables full document processing within strict time constraints.


