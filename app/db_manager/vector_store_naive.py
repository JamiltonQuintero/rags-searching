import logging
from typing import Dict, Any, List
import requests
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

        self.jina_api_key = os.getenv('JINA_API_KEY')

    def process_pdf(self, file_path: str) -> List[Document]:
        """Procesa un archivo PDF usando el chunking de Jina"""
        raw_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Limpiar caracteres NUL y normalizar el texto
                    text = text.replace('\x00', '')
                    raw_text += text + "\n\n"

        documents = []
        #chunks = self.text_processor.get_semantic_chunks(raw_text)
        chunks = self.chunk_by_tokenizer_api(raw_text)

        for i, chunk in enumerate(chunks):
            # Asegurarse de que el chunk también esté limpio
            clean_chunk = chunk.replace('\x00', '')
            documents.append(
                Document(
                    page_content=clean_chunk,
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

    def chunk_by_tokenizer_api(self, input_text: str, max_chunk_length: int = 1000):
        # Definir el endpoint y el payload de la API
        url = 'https://segment.jina.ai/'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.jina_api_key}'
        }
        payload = {
            "content": input_text,
            "return_chunks": True,
            "max_chunk_length": max_chunk_length,
            "tokenizer": "cl100k_base",  # Puedes ajustar el tokenizer si es necesario
            "return_overlaps": False  # Asumimos que no necesitamos solapamiento
        }

        # Realizar la solicitud a la API
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()

        if 'error' in response_data:
            raise Exception(f"Error en tokenizer API: {response_data['error']}")

        # Extraer chunks y número total de tokens del response
        chunks = response_data.get("chunks", [])
        num_tokens = response_data.get("num_tokens", 0)

        return chunks, num_tokens

    def generate_embedding(self, chunks) -> List[Document]:
        documents = []
        for i, chunk in enumerate(chunks):
            # Asegurarse de que el chunk también esté limpio
            clean_chunk = chunk.replace('\x00', '')
            documents.append(
                Document(
                    page_content=clean_chunk,
                    metadata={
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




