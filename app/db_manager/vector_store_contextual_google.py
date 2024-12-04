from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional, Tuple, Dict
from pypdf import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()

# Añadir este prompt al inicio de la clase
CONTEXTUAL_RAG_PROMPT = """
Given the document below, we want to explain what the chunk captures in the document.

{WHOLE_DOCUMENT}

Here is the chunk we want to explain:

{CHUNK_CONTENT}

Answer ONLY with a succinct explanation of the meaning of the chunk in the context of the whole document above.
"""

class VectorDatabaseContextualGoogle:
    def __init__(self):

        self.embeddings  = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        self.connection = os.getenv('POSTGRES_CONNECTION_STRING')
        if not self.connection:
            raise ValueError("POSTGRES_CONNECTION_STRING environment variable is not set")
            
        self.collection_name = "documents"
        
        # Inicializar el text splitter
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

        self.query_augmenter = QueryAugmenter()

        # Inicializar el cliente de Gemini
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.llm = genai.GenerativeModel("gemini-1.5-pro")

    def process_pdf(self, file_path: str) -> List[Document]:
        """Procesa un archivo PDF y lo convierte en documentos divididos"""
        pdf = PdfReader(file_path)
        raw_text = ""
        
        # Extraer texto de todas las páginas
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Limpiar caracteres NUL del texto
                text = text.replace('\x00', '')
                raw_text += text + "\n\n"
        
        # Dividir el texto en chunks usando el text splitter
        documents = self.text_splitter.create_documents(
            texts=[raw_text],
            metadatas=[{"source": file_path}]
        )
        
        # Añadir números de página a los metadatos
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "chunk_id": i,
                "chunk_total": len(documents)
            })
        
        return documents

    def _create_contextual_chunks(self, documents: List[Document]) -> List[Document]:
        """Crea chunks con contexto usando un LLM para generar explicaciones contextuales"""
        contextual_documents = []
        
        # Agrupar documentos por fuente
        docs_by_source: Dict[str, List[Document]] = {}
        for doc in documents:
            source = doc.metadata["source"]
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        # Procesar cada fuente por separado
        for source, docs in docs_by_source.items():
            # Obtener el documento completo para contexto
            full_document = ""
            for doc in sorted(docs, key=lambda x: x.metadata["chunk_id"]):
                full_document += doc.page_content + "\n\n"
            
            for doc in docs:
                # Generar explicación contextual usando LLM
                prompt = CONTEXTUAL_RAG_PROMPT.format(
                    WHOLE_DOCUMENT=full_document,
                    CHUNK_CONTENT=doc.page_content
                )
                
                # Aquí deberías hacer la llamada al LLM
                # Por ahora usaré un placeholder
                contextual_explanation = self._generate_context(prompt)
                
                # Crear nuevo documento con contexto generado por LLM
                contextual_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "contextual_explanation": contextual_explanation,
                        "has_context": True
                    }
                )
                contextual_documents.append(contextual_doc)
        
        return contextual_documents

    async def add_pdf_to_store(self, file_path: str) -> List[str]:
        """Añade un PDF a la base de datos vectorial con contexto"""
        # Procesar el PDF en chunks iniciales
        initial_documents = self.process_pdf(file_path)
        
        # Crear chunks con contexto
        contextual_documents = self._create_contextual_chunks(initial_documents)
        
        # Generar IDs únicos para cada chunk
        ids = [f"doc_{file_path}_{i}" for i in range(len(contextual_documents))]
        
        # Añadir a PGVector
        self.vector_store.add_documents(contextual_documents, ids=ids)
        return ids

    async def contextual_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> List[dict]:
        """Búsqueda que considera el contexto almacenado"""
        
        # Realizar búsqueda inicial
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        # Procesar resultados incluyendo el contexto
        processed_results = []
        for doc, score in results:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            
            # Añadir contexto si está disponible
            if doc.metadata.get("has_context"):
                result["contextual_explanation"] = doc.metadata.get("contextual_explanation", "")
                
                # Calcular score contextual
                context_relevance = (
                    len(result["contextual_explanation"]) > 0
                )
                result["context_score"] = score * (1.2 if context_relevance else 1.0)
            
            processed_results.append(result)
        
        # Ordenar por score contextual si está disponible
        processed_results.sort(
            key=lambda x: x.get("context_score", x["score"]),
            reverse=True
        )
        
        return processed_results[:k]

    def _generate_context(self, prompt: str) -> str:
        """
        Genera una explicación contextual para un chunk usando Gemini.
        Utiliza el sistema de caché de Gemini para optimizar los llamados.
        
        Args:
            prompt (str): El prompt formateado con el documento completo y el chunk
        Returns:
            str: La explicación contextual generada
        """
        try:
            import datetime
            from google.generativeai import caching
            
            # Crear una caché con TTL de 1 hora (ajustable según necesidades)
            cache = caching.CachedContent.create(
                model='models/gemini-1.5-flash-001',
                display_name='contextual_cache',
                system_instruction=(
                    'You are an expert document analyzer. Your job is to provide '
                    'concise contextual explanations based on the given document.'
                ),
                contents=[prompt],
                ttl=datetime.timedelta(hours=1),
            )

            # Construir el modelo con la caché
            model = genai.GenerativeModel.from_cached_content(cached_content=cache)
            
            # Generar respuesta
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.0,
                    'max_output_tokens': 100,
                }
            )
            
            return response.text.strip()
        except Exception as e:
            print(f"Error generating context with Gemini: {e}")
            return ""
