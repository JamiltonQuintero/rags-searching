import logging
from typing import Dict, Any, List

import pdfplumber
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
import dotenv
import os

from langchain_postgres.vectorstores import DistanceStrategy

from app.db_manager.text_processor import TextPreprocessor

dotenv.load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class VectorDatabaseNaive:
    def __init__(self):
        self.connection = os.getenv('POSTGRES_CONNECTION_STRING')
        if not self.connection:
            raise ValueError("POSTGRES_CONNECTION_STRING environment variable is not set")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.collection_name = "documents-naive"

        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection,
            use_jsonb=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
        self.text_processor = TextPreprocessor()

    def process_pdf(self, file_path: str) -> List[Document]:
        """Procesa un archivo PDF usando el chunking de Jina"""
        raw_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n\n"

        documents = []

        chunks = self.text_processor.get_semantic_chunks(raw_text)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk_id": i,
                        "chunk_total": len(chunks)
                    }
                )
            )

        logger.debug(f"Creando embeddings para {len(documents)} chunks de productos")
        self.vector_store.add_documents(documents)
        logger.debug("Embeddings creados y almacenados")


    def search(self, query: str,
                     k: int = 20,):
        logger.debug(f"Query original: {query}")

        # Usar la query expandida para buscar en la base de datos vectorial local
        results = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k
        )

        # Procesar los resultados como lo hacías antes
        documents = []
        for doc, score in results:
            document = {
                "id": doc.metadata.get("id"),
                "metadata": doc.metadata,
                "content": doc.page_content,
                "score": score,
            }
            documents.append(document)

        logger.debug(f"Total de resultados formateados: {len(documents)}")
        return documents




