
import requests
import numpy as np
from pypdf import PdfReader
from typing import List, Dict, Optional
import pdfplumber
import os
from dotenv import load_dotenv

class JinaLateChunkingDB:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Listas para almacenar los documentos y embeddings indexados
        self.docs = []
        self.embeddings = []
        # Cargar API Key de Jina desde variables de entorno
        self.jina_api_key = os.getenv('JINA_API_KEY')
        # Modelo de embeddings a utilizar
        self.embedding_model = 'jina-embeddings-v3'

    import pdfplumber

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extrae texto de un archivo PDF preservando el formato original.
        """
        raw_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n\n"
        return raw_text.strip()

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

    def _split_text_into_sections(self, text: str, max_total_tokens: int = 8000):
        """
        Divide el texto en secciones donde el número total de tokens no exceda max_total_tokens.
        """
        sections = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + (max_total_tokens * 4), text_length)  # Estimación de 4 caracteres por token
            section_text = text[start:end]
            sections.append(section_text.strip())
            start = end

        return sections

    def _generate_chunk_embeddings(self, chunks: List[str], full_text: str):
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.jina_api_key}'
        }

        # Preparar la entrada para la API de embeddings
        input_texts = [full_text] + chunks

        data = {
            "model": self.embedding_model,
            "task": "text-matching",
            "late_chunking": True,
            "embedding_type": "float",
            "input": input_texts
        }

        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()

        if 'error' in response_data:
            raise Exception(f"Error en embeddings API: {response_data['error']}")

        embeddings = [np.array(item['embedding']) for item in response_data['data']]

        # Ignorar el primer embedding (full_text) y retornar los embeddings de los chunks
        return embeddings[1:]

    def index_document(self, text: str, metadata: Optional[Dict] = None):
        """
        Indexa el documento completo, manejando el límite de tokens de la API de Jina.
        """
        # Obtener los chunks y el número total de tokens usando la API de segmentación
        chunks, num_tokens = self.chunk_by_tokenizer_api(text, max_chunk_length=1000)

        # Verificar si el número total de tokens excede el límite de la API
        if num_tokens + sum(len(chunk) for chunk in chunks) > 8194:
            # Dividir el texto en secciones más pequeñas
            sections = self._split_text_into_sections(text)
            total_chunks = 0

            for section_idx, section_text in enumerate(sections):
                # Obtener chunks y número de tokens para la sección
                section_chunks, section_num_tokens = self.chunk_by_tokenizer_api(section_text, max_chunk_length=1000)

                # Verificar si la sección excede el límite de tokens
                if section_num_tokens + sum(len(chunk) for chunk in section_chunks) > 8194:
                    raise Exception("La sección y sus chunks exceden el límite de tokens de la API de Jina.")

                # Generar embeddings con late chunking
                chunk_embeddings = self._generate_chunk_embeddings(section_chunks, section_text)

                # Almacenar los chunks y sus embeddings
                for idx, (chunk_text, chunk_embedding) in enumerate(zip(section_chunks, chunk_embeddings)):
                    doc = {
                        'text': chunk_text,
                        'embedding': chunk_embedding,
                        'metadata': {
                            **(metadata or {}),
                            'section_id': section_idx,
                            'chunk_id': idx,
                            'chunk_total': len(section_chunks)
                        }
                    }
                    self.docs.append(doc)
                    self.embeddings.append(chunk_embedding)
                    total_chunks += 1

            print(f"Se han indexado {total_chunks} chunks.")
        else:
            # Generar embeddings con late chunking
            chunk_embeddings = self._generate_chunk_embeddings(chunks, text)

            # Almacenar los chunks y sus embeddings
            for idx, (chunk_text, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
                doc = {
                    'text': chunk_text,
                    'embedding': chunk_embedding,
                    'metadata': {
                        **(metadata or {}),
                        'chunk_id': idx,
                        'chunk_total': len(chunks)
                    }
                }
                self.docs.append(doc)
                self.embeddings.append(chunk_embedding)

            print(f"Se han indexado {len(chunks)} chunks.")

    def index_pdf(self, file_path: str) -> List[str]:
        text = self._extract_text_from_pdf(file_path)
        self.index_document(text, metadata={'source': file_path})
        return [f"doc_{file_path}"]

    def search(self, query: str, k: int = 3) -> List[Dict]:
        query_embedding = self._generate_query_embedding(query)
        scores = self._calculate_similarity(query_embedding, self.embeddings)
        top_k_indices = np.argsort(scores)[-k:][::-1]

        processed_results = []
        for idx in top_k_indices:
            doc = self.docs[idx]
            score = scores[idx]
            result = {
                'content': doc['text'],
                'metadata': doc['metadata'],
                'score': float(score)
            }
            processed_results.append(result)

        return processed_results

    def _generate_query_embedding(self, query: str):
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.jina_api_key}'
        }
        data = {
            "model": self.embedding_model,
            "task": "text-matching",
            "embedding_type": "float",
            "input": [query]
        }

        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()

        if 'error' in response_data:
            raise Exception(f"Error en embeddings API: {response_data['error']}")

        query_embedding = np.array(response_data['data'][0]['embedding'])
        return query_embedding

    def _calculate_similarity(self, query_embedding, embeddings_list):
        embeddings_array = np.asarray(embeddings_list, dtype=np.float32)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        # Normalizar los vectores
        query_norm = np.linalg.norm(query_embedding)
        embeddings_norms = np.linalg.norm(embeddings_array, axis=1)
        dot_products = np.dot(embeddings_array, query_embedding)
        similarities = dot_products / (embeddings_norms * query_norm + 1e-10)
        return similarities

    def optimize_index(self):
        pass  # No es necesario en esta implementación