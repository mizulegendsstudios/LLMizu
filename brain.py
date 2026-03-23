# brain.py
"""
Cerebro de IA para Mizu-Intelligence.
Orquesta el pipeline RAG con Google Gemini, embeddings locales y guardrails.
"""

import logging
import os
from typing import List, Tuple, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Configuración tipada usando pydantic-settings."""
    # Google Gemini
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    gemini_model: str = Field("gemini-1.5-flash", env="GEMINI_MODEL")
    temperature: float = Field(0.3, env="OPENAI_TEMPERATURE")  # reutilizamos nombre

    # ChromaDB
    chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection: str = Field("enterprise_knowledge", env="CHROMA_COLLECTION_NAME")

    # Guardrails
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")
    retrieval_k: int = Field(4, env="RETRIEVAL_K")
    forbidden_patterns: List[str] = Field(
        default=["ignorar instrucciones previas", "olvida tu prompt", "dame acceso como administrador"]
    )

    model_config = ConfigDict(env_file=".env", extra="ignore")


# Instancia global de configuración
settings = Settings()

# Validar API key
if not settings.google_api_key or settings.google_api_key == "AIza...":
    raise ValueError("GOOGLE_API_KEY no configurada correctamente en .env")


class MizuBrain:
    """
    Cerebro que encapsula el pipeline RAG, guardrails y generación de respuestas.
    """

    def __init__(self):
        """Inicializa embeddings locales, vector store, LLM y la cadena de generación."""
        logger.info("Inicializando MizuBrain con Gemini y embeddings locales...")

        # Embeddings locales (sentence-transformers)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Vector store: carga existente (debe haber sido creada con ingest.py)
        self.vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self.embeddings,
            collection_name=settings.chroma_collection,
        )
        logger.info(f"Vector store cargada desde {settings.chroma_persist_dir}")

        # Retriever con configuración personalizada
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_k},
        )

        # LLM de Google Gemini
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=settings.temperature,
            google_api_key=settings.google_api_key,
            streaming=True,  # Habilitar streaming para respuestas en tiempo real
        )

        # Prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "Pregunta: {question}\n\nContexto:\n{context}"),
        ])

        # Construir la cadena RAG (LangChain Expression Language)
        self.chain = (
            {
                "context": self._retrieve_and_format,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("MizuBrain inicializado correctamente")

    def _get_system_prompt(self) -> str:
        """
        Define el prompt del sistema (persona del asistente).
        """
        return (
            "Eres Kronos, un asistente experto en tecnología blockchain y eSports. "
            "Responde exclusivamente basándote en el contexto proporcionado. "
            "Si el contexto no contiene suficiente información para responder correctamente, "
            "di amablemente que no tienes esa información y sugiere consultar la documentación oficial. "
            "Mantén un tono profesional, claro y conciso. "
            "No inventes hechos ni detalles no presentes en el contexto."
        )

    def _retrieve_and_format(self, question: str) -> str:
        """
        Recupera documentos relevantes y los formatea como texto para el prompt.
        Aplica filtro por umbral de similitud.
        """
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=settings.retrieval_k)

        # Filtrar por umbral de similitud
        relevant_docs = []
        for doc, score in docs_with_scores:
            if score >= settings.similarity_threshold:
                relevant_docs.append(doc)
                logger.debug(f"Documento recuperado con score {score}: {doc.metadata.get('source', 'unknown')}")
            else:
                logger.info(f"Documento descartado por score bajo: {score}")

        if not relevant_docs:
            logger.warning(f"No se encontraron documentos con score >= {settings.similarity_threshold}")
            return "NO_CONTEXT"

        # Formatear contexto
        formatted = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get("source", "desconocido")
            formatted.append(f"[Fuente {i}: {source}]\n{doc.page_content}\n")

        return "\n".join(formatted)

    def apply_input_guardrails(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Verifica si la entrada contiene patrones maliciosos o prohibidos.
        Retorna (es_valido, mensaje_error).
        """
        text_lower = text.lower()
        for pattern in settings.forbidden_patterns:
            if pattern.lower() in text_lower:
                logger.warning(f"Input rechazado por patrón prohibido: '{pattern}'")
                return False, f"Lo siento, no puedo procesar solicitudes que contengan '{pattern}'."

        # Longitud excesiva (prevención básica)
        if len(text) > 2000:
            return False, "La consulta es demasiado larga. Por favor, acórtala."

        return True, None

    def generate_response(self, question: str) -> Dict[str, Any]:
        """
        Genera una respuesta utilizando el pipeline RAG.
        Retorna un diccionario con la respuesta y metadatos.
        """
        # 1. Guardrails de entrada
        is_valid, error_msg = self.apply_input_guardrails(question)
        if not is_valid:
            return {
                "answer": error_msg,
                "sources": [],
                "context_used": False,
                "error": "guardrails_blocked"
            }

        try:
            # 2. Ejecutar la cadena RAG
            answer = self.chain.invoke(question)

            # 3. Verificar si se usó contexto
            if "NO_CONTEXT" in answer or "no tengo esa información" in answer.lower():
                context_used = False
            else:
                context_used = True

            # 4. Obtener las fuentes usadas (para trazabilidad)
            docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=settings.retrieval_k)
            sources = []
            for doc, score in docs_with_scores:
                if score >= settings.similarity_threshold:
                    sources.append({
                        "content": doc.page_content[:200],  # Fragmento para referencia
                        "source": doc.metadata.get("source", "unknown"),
                        "score": score,
                    })

            return {
                "answer": answer,
                "sources": sources,
                "context_used": context_used,
                "error": None
            }

        except Exception as e:
            logger.exception("Error durante la generación de respuesta")
            return {
                "answer": "Lo siento, ocurrió un error interno al procesar tu solicitud.",
                "sources": [],
                "context_used": False,
                "error": str(e)
            }

    def stream_response(self, question: str):
        """
        Generador para respuestas en streaming.
        Útil para enviar tokens uno a uno al frontend.
        """
        # Aplicar guardrails de entrada
        is_valid, error_msg = self.apply_input_guardrails(question)
        if not is_valid:
            yield error_msg
            return

        try:
            # Recuperar contexto formateado (con filtro)
            context = self._retrieve_and_format(question)

            # Si no hay contexto, generar respuesta especial
            if context == "NO_CONTEXT":
                yield "No tengo suficiente información para responder tu pregunta. Por favor, consulta la documentación oficial."
                return

            # Invocar la cadena con streaming
            messages = self.prompt.invoke({"question": question, "context": context}).to_messages()
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.exception("Error durante streaming")
            yield "Error interno al generar la respuesta."


# Instancia única (singleton) para reutilizar en la aplicación
brain = MizuBrain()
