/**
 * voice-service.js
 * Servicio de voz para Mizu-Intelligence.
 * Conecta al backend RAG mediante fetch, con soporte de streaming y cola de síntesis.
 * Dependencias: Web Speech API (nativo, no requiere librerías externas).
 */

class MizuVoiceService {
    /**
     * Constructor.
     * @param {string} apiBaseUrl - Base URL del backend (ej. "http://localhost:8000")
     * @param {Object} options - Configuración opcional.
     * @param {boolean} options.useStreaming - Si true, usa /query/stream; si false, usa /query síncrono.
     * @param {string} options.language - Idioma para reconocimiento y síntesis (por defecto 'es-ES').
     * @param {number} options.voiceRate - Velocidad de la voz (0.5 a 2, por defecto 1).
     * @param {number} options.voicePitch - Tono de la voz (0 a 2, por defecto 1).
     */
    constructor(apiBaseUrl, options = {}) {
        this.apiBaseUrl = apiBaseUrl.replace(/\/$/, ''); // eliminar slash final si existe
        this.useStreaming = options.useStreaming !== undefined ? options.useStreaming : true;
        this.language = options.language || 'es-ES';
        this.voiceRate = options.voiceRate || 1;
        this.voicePitch = options.voicePitch || 1;

        // Cola de mensajes para síntesis (evita solapamientos)
        this.speechQueue = [];
        this.isSpeaking = false;

        // Configuración de reconocimiento de voz
        this.recognition = null;
        this.isListening = false;

        // Callbacks para eventos externos
        this.onListeningStart = null;
        this.onListeningEnd = null;
        this.onPartialResult = null;    // resultado intermedio (mientras se habla)
        this.onFinalResult = null;      // resultado final cuando se detecta pausa
        this.onSpeakingStart = null;
        this.onSpeakingEnd = null;
        this.onError = null;

        this.initSpeechRecognition();
    }

    // ------------------------------
    // Inicialización de APIs
    // ------------------------------
    initSpeechRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.error('❌ Web Speech API no soportada en este navegador.');
            if (this.onError) this.onError('SpeechRecognition not supported');
            return;
        }
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        this.recognition.lang = this.language;
        this.recognition.interimResults = true;   // recibir resultados parciales mientras se habla
        this.recognition.maxAlternatives = 1;
        this.recognition.continuous = false;       // detener automáticamente al detectar pausa

        this.recognition.onstart = () => {
            this.isListening = true;
            console.log('🎤 Escuchando...');
            if (this.onListeningStart) this.onListeningStart();
        };

        this.recognition.onend = () => {
            this.isListening = false;
            console.log('🛑 Fin de escucha.');
            if (this.onListeningEnd) this.onListeningEnd();
        };

        this.recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            if (finalTranscript !== '') {
                // Resultado final: procesar la pregunta
                if (this.onFinalResult) this.onFinalResult(finalTranscript);
                this.processUserInput(finalTranscript);
                // Detener escucha automáticamente (opcional, para no acumular)
                this.stopListening();
            } else if (interimTranscript !== '') {
                // Resultado parcial: útil para mostrar feedback visual
                if (this.onPartialResult) this.onPartialResult(interimTranscript);
            }
        };

        this.recognition.onerror = (event) => {
            console.error('❌ Error en reconocimiento:', event.error);
            if (this.onError) this.onError(event.error);
        };
    }

    // ------------------------------
    // Control de escucha
    // ------------------------------
    startListening() {
        if (!this.recognition) {
            console.error('Reconocimiento no inicializado');
            return;
        }
        if (this.isListening) {
            console.warn('Ya se está escuchando, deteniendo antes de reiniciar');
            this.stopListening();
        }
        try {
            this.recognition.start();
        } catch (e) {
            console.error('Error al iniciar reconocimiento:', e);
            if (this.onError) this.onError(e);
        }
    }

    stopListening() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
    }

    // ------------------------------
    // Síntesis de voz (con cola)
    // ------------------------------
    /**
     * Añade un texto a la cola de síntesis y la procesa.
     * @param {string} text - Texto a vocalizar.
     */
    speak(text) {
        if (!text || text.trim() === '') return;
        this.speechQueue.push(text.trim());
        this.processSpeechQueue();
    }

    processSpeechQueue() {
        if (this.isSpeaking) return;
        if (this.speechQueue.length === 0) return;

        this.isSpeaking = true;
        const text = this.speechQueue.shift();
        console.log(`🗣️ Hablando: ${text.substring(0, 50)}...`);

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = this.language;
        utterance.rate = this.voiceRate;
        utterance.pitch = this.voicePitch;

        utterance.onstart = () => {
            if (this.onSpeakingStart) this.onSpeakingStart();
        };
        utterance.onend = () => {
            this.isSpeaking = false;
            if (this.onSpeakingEnd) this.onSpeakingEnd();
            // Procesar siguiente elemento en cola
            this.processSpeechQueue();
        };
        utterance.onerror = (event) => {
            console.error('Error en síntesis:', event);
            this.isSpeaking = false;
            if (this.onError) this.onError(event);
            this.processSpeechQueue(); // continuar con el siguiente a pesar del error
        };

        window.speechSynthesis.cancel(); // opcional: interrumpir si se quiere priorizar nueva respuesta
        window.speechSynthesis.speak(utterance);
    }

    /**
     * Cancela toda la síntesis en curso y vacía la cola.
     */
    stopSpeaking() {
        window.speechSynthesis.cancel();
        this.isSpeaking = false;
        this.speechQueue = [];
    }

    // ------------------------------
    // Comunicación con el backend
    // ------------------------------
    /**
     * Envía la pregunta al backend y procesa la respuesta.
     * @param {string} question - Texto de la pregunta.
     */
    async processUserInput(question) {
        if (!question || question.trim() === '') return;

        // Mostrar feedback visual (si se implementa)
        console.log(`📤 Enviando: ${question}`);

        if (this.useStreaming) {
            await this.processStreaming(question);
        } else {
            await this.processSync(question);
        }
    }

    /**
     * Usa el endpoint /query/stream para recibir tokens incrementalmente.
     * Cada token se va añadiendo a un buffer y se sintetiza en fragmentos.
     * Para simplificar, se espera a recibir toda la respuesta y luego se sintetiza.
     * Para una experiencia más natural, se podría sintetizar en trozos, pero
     * se requiere un manejo más fino del corte de oraciones.
     */
    async processStreaming(question) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/query/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question }),
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Error ${response.status}: ${errorText}`);
            }

            // Leer el stream de texto
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let fullAnswer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                fullAnswer += chunk;
                // Opcional: mostrar fragmentos en UI mientras se reciben
                console.debug('Chunk recibido:', chunk);
            }

            // Una vez completa, sintetizar
            if (fullAnswer.trim()) {
                this.speak(fullAnswer);
            } else {
                this.speak('Lo siento, no pude obtener una respuesta.');
            }
        } catch (error) {
            console.error('Error en streaming:', error);
            this.speak('Hubo un problema al contactar al asistente. Inténtalo de nuevo.');
            if (this.onError) this.onError(error);
        }
    }

    /**
     * Usa el endpoint /query síncrono: obtiene respuesta completa más metadatos.
     * Ideal para mostrar fuentes o depuración.
     */
    async processSync(question) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question }),
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Error ${response.status}: ${errorText}`);
            }

            const data = await response.json();
            const answer = data.answer;
            const sources = data.sources;

            // Mostrar fuentes en consola (o en UI)
            if (sources && sources.length > 0) {
                console.log('📚 Fuentes utilizadas:', sources.map(s => `${s.source} (score: ${s.score})`).join(', '));
            }

            if (answer && answer.trim()) {
                this.speak(answer);
            } else {
                this.speak('No obtuve una respuesta válida.');
            }
        } catch (error) {
            console.error('Error en consulta síncrona:', error);
            this.speak('Error de conexión con el servidor.');
            if (this.onError) this.onError(error);
        }
    }

    // ------------------------------
    // Métodos de utilidad
    // ------------------------------
    /**
     * Configura los callbacks.
     */
    setCallbacks(callbacks) {
        if (callbacks.onListeningStart) this.onListeningStart = callbacks.onListeningStart;
        if (callbacks.onListeningEnd) this.onListeningEnd = callbacks.onListeningEnd;
        if (callbacks.onPartialResult) this.onPartialResult = callbacks.onPartialResult;
        if (callbacks.onFinalResult) this.onFinalResult = callbacks.onFinalResult;
        if (callbacks.onSpeakingStart) this.onSpeakingStart = callbacks.onSpeakingStart;
        if (callbacks.onSpeakingEnd) this.onSpeakingEnd = callbacks.onSpeakingEnd;
        if (callbacks.onError) this.onError = callbacks.onError;
    }
}

// Ejemplo de uso (comentado):
/*
// Crear instancia apuntando al backend local
const voice = new MizuVoiceService('http://localhost:8000', { useStreaming: true });

// Configurar callbacks para actualizar la UI
voice.setCallbacks({
    onListeningStart: () => console.log('Micrófono activado'),
    onListeningEnd: () => console.log('Micrófono desactivado'),
    onPartialResult: (text) => console.log('Parcial:', text),
    onFinalResult: (text) => console.log('Final:', text),
    onSpeakingStart: () => console.log('Comienza síntesis'),
    onSpeakingEnd: () => console.log('Termina síntesis'),
    onError: (err) => console.error('Error:', err)
});

// Iniciar escucha (ej. al hacer clic en un botón)
// voice.startListening();
*/
