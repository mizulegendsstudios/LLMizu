📝 Prompt Maestro de Generación

    Actúa como un Senior AI Solution Architect. Mi objetivo es migrar un sistema de voz basado en reglas (JavaScript) a una arquitectura moderna de RAG (Retrieval-Augmented Generation) utilizando Python, FastAPI y LangChain.

    Reglas de Entrega:

        Entregarás el proyecto archivo por archivo.

        Después de cada archivo, te detendrás y esperarás a que yo diga "Siguiente" o te haga una pregunta antes de continuar.

        Cada archivo debe incluir comentarios técnicos explicando el porqué de las decisiones (ej. manejo de asincronía, validación de esquemas, etc.).

    Especificaciones Técnicas del Proyecto:

        Backend: FastAPI (Python) con soporte para Streaming y Pydantic para validación de tipos.

        Orquestación: LangChain para el pipeline de RAG.

        Vector Store: ChromaDB para persistencia local del conocimiento.

        Seguridad: Implementación de una capa de "Guardrails" básica para filtrar inputs maliciosos.

        Frontend: Una versión optimizada de mi código de voz actual que conecte vía fetch al backend de Python.

    Orden de los Archivos a entregar:

        requirements.txt (Dependencias exactas).

        .env.example (Variables de entorno necesarias).

        ingest.py (Script para transformar mis JSON de preguntas en una base de datos vectorial).

        brain.py (La lógica central de LangChain + RAG + Guardrails).

        main.py (El servidor FastAPI con manejo de errores y CORS).

        voice-service.js (La integración del frontend con el nuevo cerebro de IA).

    ¿Entendido? Si es así, comienza con el archivo 1: requirements.txt.

Por qué este prompt es "Nivel Pro":

    Requirements.txt: No solo pone librerías al azar, asegura que las versiones de langchain y openai sean compatibles para evitar errores de "import" que son comunes en IA.

    Ingest.py: Es la pieza de Ingeniería de Datos. Es lo que te permite decir en una entrevista: "Yo diseñé el pipeline de ingesta para transformar datos no estructurados en conocimiento semántico".

    Guardrails: Esto es vital para las vacantes que mencionaste ("Responsible AI practices"). Demuestra que te importa que la IA no diga cosas fuera de lugar o peligrosas.

    Desacoplamiento: Al separar brain.py (IA) de main.py (Servidor), aplicas el principio de Separación de Responsabilidades, algo que los Arquitectos de Software aman ver.
