FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=1        # ← blocks HF network calls at runtime

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download models at build time (cached in image layer)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from transformers import AutoTokenizer; \
from adapters import AutoAdapterModel; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('BAAI/bge-reranker-base'); \
tok = AutoTokenizer.from_pretrained('allenai/specter2_base'); \
m = AutoAdapterModel.from_pretrained('allenai/specter2_base'); \
m.load_adapter('allenai/specter2_adhoc_query', source='hf', load_as='specter2_adhoc_query'); \
print('Models baked successfully')"

COPY . .
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]