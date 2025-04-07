from fastapi import FastAPI, Query
from typing import List
import faiss
import pickle
import numpy as np
import os
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyBZDJOoKDRnV89ME9QY5-xAeY5AgJS73bQ")  # replace with your actual key

# Load FAISS index and metadata
index = faiss.read_index("gemini_txt_index.faiss")

with open("gemini_txt_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Path to text files
TEXT_FOLDER = "txts"

# Gemini summarizer function
def gemini_summary(text):
    prompt = f"""
Your job is to summarize this assessment description and give output like this as plain text:
- Assessment name and URL
- Remote Testing Support (Yes/No)
- Adaptive/IRT Support (Yes/No)
- Duration
- Test type

A - Ability & Aptitude  
B - Biodata & Situational Judgement  
C - Competencies  
D - Development & 360  
E - Assessment Exercises  
K - Knowledge & Skills  
P - Personality & Behavior  
S - Simulations

mention test type completely

Text:
{text}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# Search endpoint
@app.get("/search")
def search(query: str = Query(..., description="Natural language query"), top_k: int = 5):
    try:
        # Embed query
        response = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_vector = np.array(response["embedding"], dtype=np.float32).reshape(1, -1)

        # FAISS search
        distances, indices = index.search(query_vector, top_k)

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            meta = metadata[idx]
            filename = meta['filename']
            txt_path = os.path.join(TEXT_FOLDER, filename)

            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
                    summary = gemini_summary(full_text)

                results.append({
                    "rank": i + 1,
                    "filename": filename,
                    "distance": float(distances[0][i]),
                    "summary": summary
                })
            else:
                results.append({
                    "rank": i + 1,
                    "filename": filename,
                    "distance": float(distances[0][i]),
                    "error": "File not found"
                })

        return {"query": query, "results": results}

    except Exception as e:
        return {"error": str(e)}
