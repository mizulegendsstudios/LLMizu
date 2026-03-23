# 🤖 Mizu-Intelligence: Asistente RAG por Voz

Sistema de arquitectura desacoplada que integra reconocimiento de voz en tiempo real con un motor de IA basado en RAG (Retrieval-Augmented Generation). Inicialmente diseñado para escalar el conocimiento de NPCs en entornos virtuales o asistentes corporativos, se adaptó para usar modelos gratuitos y locales.

## 🚀 Logros Alcanzados

- **Migración exitosa**: Se transformó un sistema de voz basado en reglas (JavaScript) a una arquitectura moderna con FastAPI, LangChain y ChromaDB.
- **Pipeline de ingesta**: Script `ingest.py` que convierte documentos JSON en una base de datos vectorial con embeddings locales (`sentence-transformers`), eliminando dependencia de APIs pagas.
- **Cerebro RAG**: Lógica central en `brain.py` que orquesta recuperación semántica, guardrails básicos (filtro de inputs maliciosos) y generación de respuestas con Google Gemini (gratuito).
- **API robusta**: Endpoints `/query` (síncrono) y `/query/stream` (streaming) con validación Pydantic, CORS configurable y manejo de errores.
- **Frontend de voz**: Servicio JavaScript (`voice-service.js`) que usa Web Speech API para capturar voz y sintetizar respuestas, con cola para evitar solapamientos.
- **Entorno de desarrollo**: Configuración lista para Codespaces, con dependencias fijadas en `requirements.txt`.

## 🧩 Obstáculos Superados

- **Limitación de cuota en OpenAI**: Se reemplazó `gpt-4o-mini` y `text-embedding-3-small` por:
  - Embeddings locales con `HuggingFaceEmbeddings` (modelo `all-MiniLM-L6-v2`).
  - Chat con Google Gemini (modelo `gemini-1.5-flash`), que ofrece cuota gratuita suficiente para pruebas.
- **Parseo de variables de entorno**: Se corrigió el formato de `FORBIDDEN_PATTERNS` (JSON) para evitar errores con `pydantic-settings`.
- **Instalación en Codespaces**: Se ajustaron comandos y archivos para que el proyecto funcione sin necesidad de terminal local, utilizando el entorno de GitHub Codespaces.

## 🎯 Objetivo Pendiente

- **Completar la integración final y ver el software funcionando con voz**:
  - Generar la base de datos vectorial ejecutando `python ingest.py`.
  - Levantar el servidor con `uvicorn main:app --reload --host 0.0.0.0`.
  - Servir el frontend (`index.html`) con `python -m http.server 8001`.
  - Realizar una prueba de voz completa: preguntar por micrófono y escuchar la respuesta sintetizada.
  - Documentar el proceso de despliegue en producción (Render, Railway o similar) para que el asistente sea accesible desde cualquier dispositivo.

## 📦 Instalación Rápida

1.  Clona el repositorio.
2.  Crea un archivo `.env` con tu `GEMINI_API_KEY` (obtén una gratis en [Google AI Studio](https://aistudio.google.com/)).
3.  Ejecuta `pip install -r requirements.txt`.
4.  Coloca tus documentos en `data/` (ejemplo en `data/business_questions.json`).
5.  Ejecuta `python ingest.py` para indexar.
6.  Lanza el servidor: `uvicorn main:app --reload --host 0.0.0.0`.
7.  Sirve el frontend: `python -m http.server 8001` y abre `http://localhost:8001`.

## 🛠️ Tecnologías Utilizadas

- **Backend**: FastAPI, Uvicorn, Pydantic, LangChain, ChromaDB.
- **IA**: Google Gemini (chat), Sentence Transformers (embeddings locales). *Diseñado originalmente con OpenAI GPT-4o y ChromaDB (Vector Store).*
- **Frontend**: JavaScript (Web Speech API), HTML/CSS. *Estilizado con CSS3 (similar a Tailwind).*

---

> **Nota**: El proyecto sigue en desarrollo activo.
