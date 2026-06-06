from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / ".chromadb"
COLLECTION_NAME = "advanced_rag_collection"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

BM25_TOP_K = 10
VECTOR_TOP_K = 10
FUSED_TOP_K = 20
RERANK_TOP_K = 5

RRF_K = 60
MAX_RETRIEVAL_ATTEMPTS = 2

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QA_MODEL = "deepset/roberta-base-squad2"
