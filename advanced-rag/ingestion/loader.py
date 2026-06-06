import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    metadata: dict


class DocumentLoader:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def load_documents(self) -> List[Document]:
        documents = []
        if not self.data_dir.exists():
            logger.warning("Data directory %s does not exist", self.data_dir)
            return documents

        for file_path in sorted(self.data_dir.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            title = file_path.stem
            documents.append(Document(doc_id=title, title=title, text=text))
            logger.debug("Loaded document %s with %s characters", title, len(text))

        logger.info("Loaded %d documents from %s", len(documents), self.data_dir)
        return documents
