# realtime.py — Traducción de voz en tiempo real

Pipeline de **4 procesos completamente independientes** para capturar audio, transcribirlo, traducirlo y reproducirlo, todo en paralelo y sin bloqueos entre etapas.

---

## Arquitectura

```
Micrófono
    │
    ▼
┌──────────┐   audio_queue   ┌─────┐   stt_queue   ┌───────────┐   translated_queue   ┌──────┐
│ CAPTURA  │ ──────────────► │ STT │ ────────────► │ TRANSLATE │ ───────────────────► │ PLAY │
└──────────┘  (bytes WAV)    └─────┘    (texto)     └───────────┘       (texto)         └──────┘
                                                                                           │
                                                                                           ▼
                                                                                        Altavoz
```

Cada etapa corre en su propio proceso del sistema operativo. Las colas (`mp.Queue`) son los únicos canales de comunicación entre ellas.

---

## Etapas

### 1. `CAPTURA` → `audio_queue`

- Escucha el micrófono **continuamente**.
- Detecta el fin de una frase (silencio > 0.8 s) o un límite de 30 s.
- Encola el chunk de audio como **bytes WAV** en `audio_queue`.

### 2. `STT` → `stt_queue`

- Carga **Faster-Whisper** (`small`, CPU) una sola vez al arrancar.
- Consume chunks de `audio_queue` y los transcribe al idioma de entrada configurado.
- Aplica filtro VAD integrado para ignorar silencios.
- Encola el texto transcripto en `stt_queue`.

### 3. `TRANSLATE` → `translated_queue`

- Consume texto de `stt_queue`.
- Si `idioma_entrada == idioma_salida` → el texto pasa **sin traducción** (sin llamada de red).
- Si difieren → usa **Google Translator** (`deep-translator`) para traducir.
- Encola el texto final en `translated_queue`.

### 4. `PLAY`

- Consume texto de `translated_queue`.
- Sintetiza voz con **gTTS** en el idioma de salida.
- Reproduce el audio con **pygame** en el dispositivo de salida seleccionado.

---

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) como gestor de entornos:

```bash
uv sync
```

**Dependencias principales:**

| Paquete | Uso |
|---------|-----|
| `faster-whisper` | Transcripción local (STT) |
| `deep-translator` | Traducción vía Google |
| `gtts` | Síntesis de voz (TTS) |
| `pygame` | Reproducción de audio |
| `speechrecognition` + `pyaudio` | Captura de micrófono |

---

## Uso

```bash
uv run python realtime.py
```

Al iniciar, el script pide de forma interactiva:

1. **Micrófono** — índice del dispositivo de entrada.
2. **Altavoz** — índice del dispositivo de salida.
3. **Idioma de entrada** — código ISO (`es`, `en`, …).
4. **Idioma de salida** — código ISO (`es`, `en`, …).

Luego se inician los 4 procesos y el pipeline comienza a funcionar.

Presiona **Ctrl+C** para detener todo de forma limpia.

---

## Logs en consola

Cada etapa imprime mensajes con su prefijo:

```
[CAPTURA]   Chunk → audio_queue  (48 320 bytes)
[STT]       (1.23s) → stt_queue: 'Hola, ¿cómo estás?'
[TRANSLATE] (0.45s) → translated_queue: 'Hello, how are you?'
[PLAY]      Sintetizando: 'Hello, how are you?'
```

---

## Configuración avanzada

Las siguientes constantes se pueden ajustar directamente en `realtime.py`:

| Constante | Ubicación | Descripción |
|-----------|-----------|-------------|
| `energy_threshold` | `capture_process` | Sensibilidad del micrófono (default `300`) |
| `pause_threshold` | `capture_process` | Segundos de silencio para cerrar frase (default `0.8`) |
| `"small"` | `stt_process` | Tamaño del modelo Whisper (`tiny`, `small`, `medium`, `large`) |
| `maxsize=20` | `audio_queue` | Máximo de chunks WAV en cola |
| `maxsize=50` | `stt_queue` / `translated_queue` | Máximo de textos en cola |
