# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: backend + baked models ----
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/.cache/huggingface
ENV PORT=8080

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download models at build time (cached in image layer)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from transformers import AutoTokenizer; \
from adapters import AutoAdapterModel; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('BAAI/bge-reranker-v2-m3'); \
tok = AutoTokenizer.from_pretrained('allenai/specter2_base'); \
m = AutoAdapterModel.from_pretrained('allenai/specter2_base'); \
m.load_adapter('allenai/specter2_adhoc_query', source='hf', load_as='specter2_adhoc_query'); \
print('Models baked successfully')"

ENV TRANSFORMERS_OFFLINE=1

COPY . .
COPY --from=frontend-builder /frontend/dist ./frontend/dist

EXPOSE 8080
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
