# ingest.py
"""
Pipeline de ingesta de conocimiento para Mizu-Intelligence.
Transforma documentos en embeddings vectoriales usando sentence-transformers
y los almacena en ChromaDB.
"""

import os
import json
import logging
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración desde variables de entorno
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "enterprise_knowledge")
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"  # Modelo local de sentence-transformers

# Inicializar embeddings (local)
logger.info(f"Usando modelo de embeddings: {EMBEDDINGS_MODEL}")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

# Configuración del text splitter para documentos largos
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def load_json_documents(file_path: str) -> List[Document]:
    """
    Carga un archivo JSON con estructura de preguntas/respuestas.
    Espera un array de objetos con claves "q" y "a".
    """
    if not os.path.exists(file_path):
        logger.error(f"Archivo no encontrado: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Adaptar según estructura esperada
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "data" in data:
        items = data["data"]
    else:
        logger.error(f"Estructura JSON no reconocida en {file_path}")
        return []

    documents = []
    for idx, item in enumerate(items):
        if "q" not in item or "a" not in item:
            logger.warning(f"Ítem {idx} sin 'q' o 'a', se omite")
            continue

        # Construir el contenido textual
        content = f"Pregunta: {item['q']}\nRespuesta: {item['a']}"

        # Metadatos para trazabilidad y filtrado
        metadata = {
            "source": os.path.basename(file_path),
            "type": "qa",
            "question": item["q"],
            "doc_id": f"{os.path.basename(file_path)}:{idx}",
        }

        # Agregar categoría si existe
        if "category" in item:
            metadata["category"] = item["category"]

        documents.append(Document(page_content=content, metadata=metadata))

    logger.info(f"Cargados {len(documents)} documentos desde {file_path}")
    return documents


def load_text_documents(file_path: str) -> List[Document]:
    """Carga un archivo de texto plano y lo divide en chunks."""
    if not os.path.exists(file_path):
        logger.error(f"Archivo no encontrado: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = TEXT_SPLITTER.split_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "source": os.path.basename(file_path),
            "type": "text",
            "chunk": i,
            "doc_id": f"{os.path.basename(file_path)}:chunk{i}",
        }
        documents.append(Document(page_content=chunk, metadata=metadata))

    logger.info(f"Cargados {len(documents)} chunks desde {file_path}")
    return documents


def create_vector_store(documents: List[Document]):
    """Crea o actualiza la base de datos vectorial."""
    logger.info(f"Creando vector store en {CHROMA_PERSIST_DIR} con colección '{CHROMA_COLLECTION}'")
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION,
        )
        # En Chroma >=0.4.x, persist es automático, pero se llama por compatibilidad
        vectorstore.persist()
        logger.info(f"Vector store creado con {len(documents)} documentos")
        return vectorstore
    except Exception as e:
        logger.error(f"Error al crear vector store: {e}")
        raise


def main():
    """Punto de entrada principal."""
    # Archivos de conocimiento a procesar
    knowledge_files = [
        {"path": "data/business_questions.json", "type": "json"},
        # {"path": "data/product_manual.txt", "type": "text"},
    ]

    all_documents = []

    for file_info in knowledge_files:
        path = file_info["path"]
        if not os.path.exists(path):
            logger.warning(f"Archivo {path} no existe, se omite")
            continue

        if file_info["type"] == "json":
            docs = load_json_documents(path)
        elif file_info["type"] == "text":
            docs = load_text_documents(path)
        else:
            logger.warning(f"Tipo {file_info['type']} no soportado para {path}")
            continue

        all_documents.extend(docs)

    if not all_documents:
        logger.error("No se cargó ningún documento. Asegúrate de tener archivos en la carpeta 'data'.")
        return

    logger.info(f"Total de documentos a indexar: {len(all_documents)}")
    create_vector_store(all_documents)
    logger.info("✅ Ingesta completada con éxito.")


if __name__ == "__main__":
    main()
