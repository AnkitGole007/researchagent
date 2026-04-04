import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app import LLMConfig, predict_citations_direct, Paper

# Simulate an empty API key and see if the function successfully falls back or throws an Auth Error
llm_config = LLMConfig(api_key="", model="llama-3.1-8b-instant", api_base=None, provider="groq")
# Active configuration using the patch logic
active_llm_config = (llm_config if llm_config.api_key.strip() else None) if llm_config.provider in ("openai", "gemini", "groq") else None

print(f"active_llm_config is {active_llm_config} (Should be None)")

test_paper = Paper(
    arxiv_id="test",
    title="Test Paper",
    authors=["Author"],
    email_domains=[],
    abstract="Abstract",
    submitted_date=datetime.datetime.now(),
    pdf_url="",
    arxiv_url=""
)
# Make it primary so it goes into scoring
test_paper.focus_label = "primary"

try:
    papers = predict_citations_direct([test_paper], active_llm_config)
    print("SUCCESS: predict_citations_direct executed gracefully.")
    print("Citation logic:", papers[0].predicted_citations)
except Exception as e:
    print("FAILED with Exception:", type(e).__name__, e)

