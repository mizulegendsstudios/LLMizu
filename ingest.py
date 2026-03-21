import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Supongamos que estas son tus businessQuestions de JS
data = [
    {"q": "¿Qué es TronCraft?", "a": "Es un multiverso digital basado en cubos..."},
    {"q": "Servicios de Mizu Legends", "a": "Coaching de MOBA y torneos eSports..."}
]

docs = [Document(page_content=f"Pregunta: {i['q']} Respuesta: {i['a']}") for i in data]

# Crear base de datos vectorial
vector_db = Chroma.from_documents(
    documents=docs, 
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)
print("✅ Base de datos de conocimiento creada.")
