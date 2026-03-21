# System Architecture

El proyecto sigue un patrón de **Microservicios Desacoplados**:

### 1. Ingestion Pipeline (Data Engineering)
- Los documentos (PDF, JSON, MD) se dividen mediante `RecursiveCharacterTextSplitter`.
- Se generan embeddings vectoriales y se almacenan en **ChromaDB**.

### 2. Retrieval Layer (The Search)
- Se implementa un **Retriever** con búsqueda por similitud de coseno.
- Filtro de relevancia para evitar "alucinaciones" del modelo.

### 3. Inference Layer (The Logic)
- **FastAPI** actúa como orquestador.
- Se utiliza una **Prompt Template** refinada para asegurar que la IA mantenga el "persona" del asistente (ej. Kronos).

### 4. Client Layer (The Interface)
- El frontend gestiona la cola de mensajes (`speechQueue`) para evitar solapamientos en la síntesis de voz.
