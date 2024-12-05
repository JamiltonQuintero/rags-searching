import logging
import os
from typing import Optional, List

import pdfplumber
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.db_manager.jina_vector_store import JinaLateChunkingDB
from app.db_manager.vector_store_contextual import VectorDatabaseContextualOpenIA

from fastapi import FastAPI, UploadFile, File, HTTPException
from app.db_manager.vector_store_naive import VectorDatabaseNaive
from app.db_manager.elasticsearch_store import ElasticsearchStore

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Configuración de PostgreSQL
vector_db = VectorDatabaseContextualOpenIA()

# Instancia global de JinaLateChunkingDB
jina_db = JinaLateChunkingDB()
naive = VectorDatabaseNaive()

# Instancia global de ElasticsearchStore
es_store = ElasticsearchStore()

# Modelos de datos separados para cada tipo de query
class ContextualQueryPayload(BaseModel):
    query: str
    filters: Optional[dict] = None
    k: Optional[int] = 20

class JinaQueryPayload(BaseModel):
    query: str
    filters: Optional[dict] = None
    k: Optional[int] = 20

class ComparativeQueryPayload(BaseModel):
    query: str
    filters: Optional[dict] = None
    k: Optional[int] = 20

class ElasticsearchQueryPayload(BaseModel):
    query: str
    filters: Optional[dict] = None
    k: Optional[int] = 20

@app.post("/upload-pdf")
async def upload_pdf_contextual(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {"error": "El archivo debe ser un PDF"}

    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)


        """Procesa un archivo PDF usando el chunking de Jina"""
        raw_text = ""
        with pdfplumber.open(temp_file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Limpiar caracteres NUL y normalizar el texto
                    text = text.replace('\x00', '')
                    raw_text += text + "\n\n"

        documents = []
        chunks, num_tokens = naive.chunk_by_tokenizer_api(raw_text)

        # Procesar el PDF con el enfoque contextual
        naive.generate_embedding(chunks)
        await vector_db.add_pdf_to_store_embeddings(chunks)
        jina_db.index_document_embedding(chunks, num_tokens)

        return {
            "filename": file.filename,
            "status": "success",
            "message": "PDF cargado y procesado correctamente con motor contextual",
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/upload-pdf/naive")
async def upload_pdf_contextual(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {"error": "El archivo debe ser un PDF"}

    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Procesar el PDF con el enfoque contextual
        naive.process_pdf(temp_file_path)

        return {
            "filename": file.filename,
            "status": "success",
            "message": "PDF cargado y procesado correctamente con motor naive",
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/upload-pdf/contextual")
async def upload_pdf_contextual(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {"error": "El archivo debe ser un PDF"}

    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Procesar el PDF con el enfoque contextual
        ids = await vector_db.add_pdf_to_store(temp_file_path)

        return {
            "filename": file.filename,
            "status": "success",
            "message": "PDF cargado y procesado correctamente con motor contextual",
            "document_ids": ids
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/upload-pdf/jina")
async def upload_pdf_jina(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {"error": "El archivo debe ser un PDF"}

    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Procesar el PDF con Jina usando la instancia global
        jina_db.process_pdf(temp_file_path)

        return {
            "filename": file.filename,
            "status": "success",
            "message": "PDF cargado y procesado correctamente con motor Jina",
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/upload-pdf/elasticsearch")
async def upload_pdf_elasticsearch(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {"error": "El archivo debe ser un PDF"}

    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Procesar el PDF con Elasticsearch
        documents = es_store.process_pdf(temp_file_path)

        return {
            "filename": file.filename,
            "status": "success",
            "message": "PDF cargado y procesado correctamente con Elasticsearch",
            "chunks_processed": len(documents)
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/query/naive")
async def process_jina_query(payload: JinaQueryPayload):
    try:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")

        # Utilizar la instancia global
        results = naive.search(
            query=payload.query,
            k=payload.k
        )

        if not results:
            return {
                "query": payload.query,
                "results": [],
                "message": "No se encontraron resultados para la consulta"
            }

        return {
            "query": payload.query,
            "results": results,
            "message": "Búsqueda exitosa"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {str(e)}")

@app.post("/query/contextual")
async def process_contextual_query(payload: ContextualQueryPayload):
    try:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")

        results = await vector_db.contextual_search(
            query=payload.query,
            k=payload.k
        )

        if not results:
            return {
                "query": payload.query,
                "results": [],
                "message": "No se encontraron resultados para la consulta"
            }

        # Simplify the results to include only original_content and rrf_score
        simplified_results = [
            {
                "original_content": result["metadata"]["original_content"],
                "rrf_score": result["rrf_score"],
                "chunk_id": result["metadata"]["chunk_id"]
            }
            for result in results
        ]

        return {
            "query": payload.query,
            "results": simplified_results,
            "message": "Búsqueda exitosa"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {str(e)}")

@app.post("/query/jina")
async def process_jina_query(payload: JinaQueryPayload):
    try:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")

        # Utilizar la instancia global
        results = jina_db.search(
            query=payload.query,
            k=payload.k
        )

        if not results:
            return {
                "query": payload.query,
                "results": [],
                "message": "No se encontraron resultados para la consulta"
            }

        return {
            "query": payload.query,
            "results": results,
            "message": "Búsqueda exitosa"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {str(e)}")

@app.post("/query/comparative")
async def process_comparative_query(payload: ComparativeQueryPayload):
    try:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")

        # Realizar búsquedas en paralelo
        contextual_results = await vector_db.contextual_search(
            query=payload.query,
            k=payload.k
        )

        jina_results = jina_db.search(
            query=payload.query,
            k=payload.k
        )

        # Normalizar los puntajes para cada conjunto de resultados
        normalized_contextual = _normalize_scores(contextual_results, score_key='rrf_score')
        normalized_jina = _normalize_scores(jina_results, score_key='score')

        # Combinar los resultados basándose en chunk_id
        combined_results = combine_results(normalized_contextual, normalized_jina)

        return {
            "query": payload.query,
            "combined_results": combined_results,
            "message": "Búsqueda comparativa exitosa"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta comparativa: {str(e)}")

@app.post("/query/hybrid-search")
async def process_hybrid_search(payload: ElasticsearchQueryPayload):
    try:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")

        results = es_store.search(
            query=payload.query,
            k=payload.k
        )

        if not results:
            return {
                "query": payload.query,
                "results": [],
                "message": "No se encontraron resultados para la consulta"
            }

        return {
            "query": payload.query,
            "results": results,
            "message": "Búsqueda híbrida exitosa"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {str(e)}")

def combine_results(contextual_results: List[dict], jina_results: List[dict]) -> List[dict]:
    """
    Combina los resultados de las búsquedas contextual y Jina basándose en chunk_id.
    """
    combined_results = []
    # Crear un diccionario para acceder a los resultados de Jina por chunk_id
    jina_results_dict = {res['metadata']['chunk_id']: res for res in jina_results}

    for c_res in contextual_results:
        chunk_id = c_res.get('chunk_id')
        jina_res = jina_results_dict.get(chunk_id)
        if jina_res:
            combined_result = {
                'content': c_res['content'],
                'metadata': c_res['metadata'],
                'contextual_score': c_res.get('normalized_score'),
                'jina_score': jina_res.get('normalized_score'),
                'contextual_explanation': c_res.get('contextual_explanation')
            }
            combined_results.append(combined_result)
        else:
            # Si no hay resultado de Jina para este chunk, puedes decidir cómo manejarlo
            pass  # O agregarlo con jina_score=None
    return combined_results

def _normalize_scores(results: List[dict], score_key: str) -> List[dict]:
    """Normaliza los puntajes al rango [0,1] para hacer comparables los resultados"""
    if not results:
        return results

    # Encontrar el puntaje máximo y mínimo
    scores = [result[score_key] for result in results]
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score

    # Normalizar los puntajes
    normalized_results = []
    for result in results:
        result_copy = result.copy()
        if score_range > 0:
            result_copy['normalized_score'] = (result[score_key] - min_score) / score_range
        else:
            result_copy['normalized_score'] = 1.0
        normalized_results.append(result_copy)

    return normalized_results

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )