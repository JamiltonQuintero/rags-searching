import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import logging

class TextPreprocessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        # Descargar recursos necesarios de NLTK
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def preprocesar_texto(self, texto):
        # Normalización
        texto = texto.lower()

        # Eliminar caracteres especiales y números, pero mantener acentos y ñ
        texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)

        # Tokenización
        tokens = word_tokenize(texto, language='spanish')

        # Eliminar stop words
        stop_words = set(stopwords.words('spanish'))
        tokens = [token for token in tokens if token not in stop_words]

        # Stemming
        stemmer = SnowballStemmer('spanish')
        tokens = [stemmer.stem(token) for token in tokens]

        # Reconstruir el texto
        texto_optimizado = ' '.join(tokens)

        # Dividir párrafos largos
        texto_optimizado = re.sub(r'(.{500,}?[.!?])\s', r'\1\n\n', texto_optimizado)

        # Eliminar líneas en blanco duplicadas
        texto_optimizado = re.sub(r'\n\s*\n', '\n\n', texto_optimizado)

        return texto_optimizado

    def get_chunks(self, texto):
        texto_limpio = self.preprocesar_texto(texto)
        chunks = self.text_splitter.split_text(texto_limpio)
        return chunks

    # Métodos adicionales para otros tipos de chunking (para referencia)
    def get_optimized_chunks(self, texto, max_chars=1000, overlap=200):
        texto_limpio = self.preprocesar_texto(texto)
        oraciones = sent_tokenize(texto_limpio, language='spanish')

        chunks = []
        chunk_actual = ""

        for oracion in oraciones:
            if len(chunk_actual) + len(oracion) <= max_chars:
                chunk_actual += oracion + " "
            else:
                chunks.append(chunk_actual.strip())
                palabras = chunk_actual.split()
                chunk_actual = " ".join(palabras[-overlap // 10:]) + " " + oracion + " "

        if chunk_actual:
            chunks.append(chunk_actual.strip())

        return chunks

    def get_semantic_chunks(self, texto, embedding_model="openai"):
        #texto_limpio = self.preprocesar_texto(texto)

        if embedding_model == "openai":
            embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        elif embedding_model == "ollama":
            embedding = OllamaEmbeddings(base_url="http://119.9.90.22:30010", model="jina/jina-embeddings-v2-base-es")
        else:
            raise ValueError("Modelo de embedding no soportado")

        semantic_splitter = SemanticChunker(embedding, breakpoint_threshold_type="percentile")
        docs = semantic_splitter.create_documents([texto])
        return [doc.page_content for doc in docs]


