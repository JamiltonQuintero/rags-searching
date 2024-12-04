import pdfplumber
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional, Tuple, Dict
from dotenv import load_dotenv
import numpy as np
import os
# Inicializar el cliente de OpenAI con Portkey
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
from collections import defaultdict
# Inicializar BM25 para búsqueda léxica
from rank_bm25 import BM25Okapi
import requests

from app.db_manager.text_processor import TextPreprocessor

load_dotenv()

# Añadir este prompt al inicio de la clase
CONTEXTUAL_RAG_PROMPT = """
Given the document below, we want to explain what the chunk captures in the document.
Give your explanation in the same language as the input document.

{WHOLE_DOCUMENT}

Here is the chunk we want to explain:

{CHUNK_CONTENT}

Answer ONLY with a succinct explanation of the meaning of the chunk in the context of the whole document above.
"""

class VectorDatabaseContextualOpenIA:
    def __init__(self):

        self.embeddings  = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        self.connection = os.getenv('POSTGRES_CONNECTION_STRING')
        if not self.connection:
            raise ValueError("POSTGRES_CONNECTION_STRING environment variable is not set")
            
        self.collection_name = "documents-contextual"
        
        # Inicializar el text splitter (lo mantenemos como respaldo)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection,
            use_jsonb=True
        )

        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=PORTKEY_GATEWAY_URL,
            default_headers=createHeaders(
                provider="openai",
                api_key=os.getenv("PORTKEY_API_KEY")
            )
        )
        self.bm25 = None
        self.contextual_chunks = []
        # Añadir API key de Jina para el chunking
        self.jina_api_key = os.getenv('JINA_API_KEY')
        self.text_processor = TextPreprocessor()

    def chunk_by_tokenizer_api(self, input_text: str, max_chunk_length: int = 1000):
        """
        Utiliza la API de Jina para realizar el chunking del texto.
        """
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

        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()

        if 'error' in response_data:
            raise Exception(f"Error en tokenizer API: {response_data['error']}")

        chunks = response_data.get("chunks", [])
        num_tokens = response_data.get("num_tokens", 0)

        return chunks, num_tokens

    def _split_text_into_sections(self, text: str, max_total_tokens: int = 8000):
        """
        Divide el texto en secciones donde el número total de tokens no exceda max_total_tokens.
        """
        sections = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + (max_total_tokens * 4), text_length)
            section_text = text[start:end]
            sections.append(section_text.strip())
            start = end

        return sections

    def process_pdf(self, file_path: str) -> List[Document]:
        """Procesa un archivo PDF usando el chunking de Jina"""
        raw_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n\n"
        
        documents = []
        sections = self._split_text_into_sections(raw_text)
        
        total_chunks = 0
        for section_idx, section_text in enumerate(sections):
            # Obtener chunks usando la API de Jina
            chunks, num_tokens = self.chunk_by_tokenizer_api(section_text)

            # Crear documentos de Langchain para cada chunk
            for chunk_idx, chunk_text in enumerate(chunks):
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": file_path,
                        "section_id": section_idx,
                        "chunk_id": total_chunks + chunk_idx,
                        "chunk_total": len(chunks)
                    }
                )
                documents.append(doc)
            total_chunks += len(chunks)
        
        return documents

    def process_pdf(self, file_path: str) -> List[Document]:
        """Procesa un archivo PDF usando el chunking de Jina"""
        raw_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n\n"

        documents = []
        sections = self._split_text_into_sections(raw_text)

        total_chunks = 0
        # for section_idx, section_text in enumerate(sections):
        # Obtener chunks usando la API de Jina
        # chunks, num_tokens = self.chunk_by_tokenizer_api(section_text)
        chunks = self.text_processor.get_semantic_chunks(raw_text)

        # Crear documentos de Langchain para cada chunk
        for chunk_idx, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": file_path,
                    "chunk_id": chunk_idx,
                    "chunk_total": len(chunks)
                }
            )
            documents.append(doc)

        return documents
    def _create_contextual_chunks(self, documents: List[Document]) -> List[Document]:
        """Implementación del Contextual RAG según el paper de Anthropic"""
        contextual_documents = []
        docs_by_source = self._group_docs_by_source(documents)
        
        for source, docs in docs_by_source.items():
            full_document = self._get_full_document(docs)
            
            for doc in docs:
                contextual_explanation = self._generate_context(
                    CONTEXTUAL_RAG_PROMPT.format(
                        WHOLE_DOCUMENT=full_document,
                        CHUNK_CONTENT=doc.page_content
                    )
                )
                
                # Crear chunk contextual combinando explicación y contenido original
                contextual_content = f"{contextual_explanation}\n\n{doc.page_content}"
                contextual_doc = Document(
                    page_content=contextual_content,
                    metadata={
                        **doc.metadata,
                        "contextual_explanation": contextual_explanation,
                        "original_content": doc.page_content,
                        "has_context": True
                    }
                )
                contextual_documents.append(contextual_doc)
        
        # Actualizar BM25 con los nuevos chunks contextuales
        self._update_bm25_index(contextual_documents)
        return contextual_documents

    def _update_bm25_index(self, documents: List[Document]):
        """Actualiza el índice BM25 con los chunks contextuales"""
        if not documents:
            return

        # Tokenizar documentos para BM25
        tokenized_corpus = [
            doc.page_content.lower().split() 
            for doc in documents
        ]
        
        # Actualizar el índice BM25 y los chunks contextuales
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.contextual_chunks = documents

    async def add_pdf_to_store(self, file_path: str) -> List[str]:
        """Añade un PDF a la base de datos vectorial con contexto usando el chunking de Jina"""
        # Procesar el PDF usando el chunking de Jina
        initial_documents = self.process_pdf(file_path)
        
        # Crear chunks con contexto
        contextual_documents = self._create_contextual_chunks(initial_documents)
        
        # Generar IDs únicos para cada chunk
        ids = [f"doc_{file_path}_{i}" for i in range(len(contextual_documents))]
        
        # Añadir a PGVector
        self.vector_store.add_documents(contextual_documents)
        return ids

    async def contextual_search(
        self,
        query: str,
        k: int = 20,
        filter_dict: Optional[dict] = None
    ) -> List[dict]:
        """Búsqueda híbrida usando embeddings y BM25 según el paper"""
        
        # Verificar si tenemos índice BM25
        if self.bm25 is None or not self.contextual_chunks:
            # Si no hay índice BM25, solo usar búsqueda por embeddings
            vector_results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            return [{
                "content": doc.page_content,
                "metadata": doc.metadata,
                "rrf_score": 1.0 - score, # Normalizar score
                "chunk_id": doc.metadata.get("chunk_id")
            } for doc, score in vector_results]
        
        # Si hay índice BM25, realizar búsqueda híbrida
        vector_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_k = np.argsort(bm25_scores)[-k:][::-1]
        
        combined_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=[(self.contextual_chunks[i], bm25_scores[i]) for i in bm25_top_k]
        )
        
        return combined_results[:k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        k: int = 60
    ) -> List[dict]:
        """Implementa Reciprocal Rank Fusion para combinar resultados"""
        rrf_scores = defaultdict(float)
        
        # Procesar resultados de embeddings
        for rank, (doc, score) in enumerate(vector_results, 1):
            rrf_scores[doc.metadata["chunk_id"]] += 1 / (rank + k)
        
        # Procesar resultados de BM25
        for rank, (doc, score) in enumerate(bm25_results, 1):
            rrf_scores[doc.metadata["chunk_id"]] += 1 / (rank + k)
        
        # Normalizar scores RRF al rango [0,1]
        max_score = max(rrf_scores.values())
        for chunk_id in rrf_scores:
            rrf_scores[chunk_id] /= max_score if max_score > 0 else 1
        
        # Ordenar y formatear resultados
        processed_results = []
        for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            doc = next(d for d in self.contextual_chunks if d.metadata["chunk_id"] == chunk_id)
            processed_results.append({
                "content": doc.metadata["original_content"],
                "contextual_explanation": doc.metadata["contextual_explanation"],
                "metadata": doc.metadata,
                "rrf_score": rrf_score
            })
        
        return processed_results

    def _generate_context(self, prompt: str) -> str:
        """
        Genera una explicación contextual para un chunk usando OpenAI con Portkey caching.
        
        Args:
            prompt (str): El prompt formateado con el documento completo y el chunk
        Returns:
            str: La explicación contextual generada
        """
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",  # Puedes usar gpt-3.5-turbo para reducir costos
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert document analyzer. Your job is to provide concise contextual explanations based on the given document."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=100,
                # Portkey se encarga automáticamente del caching
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating context with OpenAI: {e}")
            return ""

    def _group_docs_by_source(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Agrupa los documentos por su fuente.
        
        Args:
            documents: Lista de documentos a agrupar
        Returns:
            Diccionario con documentos agrupados por fuente
        """
        docs_by_source: Dict[str, List[Document]] = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        # Ordenar documentos por chunk_id dentro de cada fuente
        for source in docs_by_source:
            docs_by_source[source].sort(key=lambda x: x.metadata.get("chunk_id", 0))
        
        return docs_by_source

    def _get_full_document(self, docs: List[Document]) -> str:
        """
        Reconstruye el documento completo a partir de sus chunks.
        
        Args:
            docs: Lista de documentos (chunks) de una misma fuente
        Returns:
            Texto completo del documento reconstruido
        """
        # Ordenar chunks por su posición original
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get("chunk_id", 0))
        
        # Reconstruir el documento
        full_text = ""
        for doc in sorted_docs:
            # Obtener el contenido original si existe, sino usar page_content
            content = doc.metadata.get("original_content", doc.page_content)
            full_text += content + "\n\n"
        
        return full_text.strip()
