import logging
from typing import List, Dict, Any
import os
import requests
import pdfplumber

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from app.db_manager.text_processor import TextPreprocessor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ElasticsearchStore:
    def __init__(self):
        # Configuración de Elasticsearch
        self.es_url = os.getenv('ES_URL', 'http://localhost:9200')
        self.es_client = Elasticsearch(hosts=[self.es_url])

        # Configuración de embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Nombres de campos e índice
        self.index_name = "documents"
        self.text_field = "content"
        self.vector_field = "embedding"
        self.metadata_field = "metadata"
        
        self.text_processor = TextPreprocessor()
        self.jina_api_key = os.getenv('JINA_API_KEY')

        # Crear índice si no existe
        self._create_index()

    def _create_index(self):
        """Crea el índice con el mapping adecuado si no existe"""
        if not self.es_client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        self.text_field: {"type": "text"},
                        self.vector_field: {
                            "type": "dense_vector",
                            "dims": 1536,  # Dimensión para text-embedding-3-small
                            "index": True,
                            "similarity": "cosine"
                        },
                        self.metadata_field: {"type": "object"}
                    }
                }
            }
            self.es_client.indices.create(index=self.index_name, body=mapping)
            logger.debug(f"Índice {self.index_name} creado")

    def process_pdf(self, file_path: str) -> List[Document]:
        """Procesa un archivo PDF y lo almacena en Elasticsearch"""
        # Extraer texto del PDF
        raw_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n\n"

        # Obtener chunks semánticos
        #chunks = self.text_processor.get_semantic_chunks(raw_text)
        chunks, num_tokens = self.chunk_by_tokenizer_api(text, max_chunk_length=1000)

        # Crear documentos y embeddings
        documents = []
        _requests = []
        
        for i, chunk in enumerate(chunks):
            # Crear documento de Langchain
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "chunk_id": i,
                    "chunk_total": len(chunks)
                }
            )
            documents.append(doc)
            
            # Generar embedding
            vector = self.embeddings.embed_documents([chunk])[0]
            
            # Preparar documento para Elasticsearch
            es_doc = {
                "_index": self.index_name,
                "_id": f"{file_path}-{i}",
                self.text_field: chunk,
                self.vector_field: vector,
                self.metadata_field: doc.metadata
            }
            _requests.append(es_doc)

        # Indexar documentos en bulk
        if requests:
            bulk(self.es_client, _requests)
            self.es_client.indices.refresh(index=self.index_name)
            logger.debug(f"Indexados {len(_requests)} documentos en Elasticsearch")

        return documents
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
            "tokenizer": "cl100k_base",
            "return_overlaps": False
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

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Realiza una búsqueda KNN en Elasticsearch"""
        # Generar embedding de la query
        query_vector = self.embeddings.embed_query(query)

        # Construir consulta KNN
        search_query = {
            "size": k,
            "knn": {
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": 100
            }
        }

        # Ejecutar búsqueda
        response = self.es_client.search(
            index=self.index_name,
            body=search_query
        )

        # Procesar resultados
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "id": hit["_id"],
                "content": hit["_source"][self.text_field],
                "metadata": hit["_source"][self.metadata_field],
                "score": hit["_score"]
            }
            results.append(result)

        logger.debug(f"Encontrados {len(results)} resultados")
        return results