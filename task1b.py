import json
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json_for_1b

json_for_1b.run()

model = SentenceTransformer("bge_model")
"""
with open("sections.json", "r", encoding="utf-8") as f:
    sections = json.load(f)
"""
sections = json_for_1b.combined_sections

with open("input/challenge1b_input.json", "r", encoding="utf-8") as f:
    challenge = json.load(f)

persona = challenge["persona"]["role"]
job = challenge["job_to_be_done"]["task"]
query = (
    f"Represent this query for retrieving supporting documents: "
    f"As a {persona}, you need to {job}. Focus on actionable or instructional content."
)
query_emb = model.encode(query)

results = []
for section in sections:
    content = section.get("content", "").strip()
    if not content:
        continue

    section_emb = model.encode(content)
    score = cosine_similarity([query_emb], [section_emb])[0][0]

    results.append({
        "score": float(score),
        "heading": section.get("heading"),
        "page": section.get("page"),
        "content": content,
        "source_file": section.get("source_file")
    })

results = sorted(results, key=lambda x: x["score"], reverse=True)
top_k = results[:5]

formatted_output = {
    "metadata": {
        "input_documents": challenge.get("documents", []),
        "persona": challenge["persona"]["role"],
        "job_to_be_done": challenge["job_to_be_done"]["task"],
        "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    },
    "extracted_sections": [
        {
            "document": r["source_file"],
            "section_title": r["heading"],
            "importance_rank": idx + 1,
            "page_number": r["page"]
        }
        for idx, r in enumerate(top_k)
    ],
    "subsection_analysis": [
        {
            "document": r["source_file"],
            "refined_text": r["content"],
            "page_number": r["page"]
        }
        for r in top_k
    ]
}

with open("output/challenge1b_output.json", "w", encoding="utf-8") as f:
    json.dump(formatted_output, f, indent=4, ensure_ascii=False)

print("Output written to challenge1b_output.json")
