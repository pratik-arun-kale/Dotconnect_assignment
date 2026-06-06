import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import EMBEDDING_MODEL
from ingestion.chunker import Chunker
from ingestion.embedding_generator import EmbeddingGenerator
from ingestion.index_builder import IndexBuilder
from ingestion.loader import DocumentLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    loader = DocumentLoader()
    documents = loader.load_documents()
    if not documents:
        logger.error("No documents found in the data directory.")
        return

    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)

    embedder = EmbeddingGenerator(EMBEDDING_MODEL)
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    builder = IndexBuilder()
    builder.build_collection(chunks, embeddings)
    logger.info("Index build completed successfully.")


if __name__ == "__main__":
    main()
