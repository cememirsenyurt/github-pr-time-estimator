# backend/python-service/openai_nlp.py

import os
from dotenv import load_dotenv
from transformers import pipeline

# Load your .env for other vars if needed (e.g. OPENAI_API_KEY fallback)
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '..', '.env')))

# Initialize a small summarization pipeline only once:
# 'facebook/bart-large-cnn' is under the Apache 2.0 License and free to use.
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1    # CPU; set to 0 if you have a CUDA GPU
)

def summarize_text(text: str) -> str:
    """
    Summarize PR body using a local, free-to-use HF model.
    Falls back to a simple slice if anything breaks.
    """
    try:
        # Hugging Face models expect fairly short inputs (<1024 tokens),
        # so we truncate to, say, the first 5000 characters
        snippet = text[:5000]
        out = summarizer(snippet, max_length=60, min_length=20, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        # Fallback: just return the first 200 chars
        print(f"⚠️ Summarization failed ({e}), falling back to stub.")
        return text[:200] + ("…" if len(text) > 200 else "")
