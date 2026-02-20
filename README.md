# realtime.py — Traducción de voz en tiempo real

Pipeline de **5 procesos completamente independientes** para capturar audio, transcribirlo, traducirlo, pre-procesar la salida y reproducirla, todo en paralelo y sin bloqueos entre etapas.

---

## Arquitectura

```
Micrófono
    │
    ▼
┌──────────┐  audio_queue  ┌─────┐  stt_queue  ┌───────────┐  translated_queue  ┌──────────────┐  output_queue  ┌──────┐
│ CAPTURA  │ ────────────► │ STT │ ───────────► │ TRANSLATE │ ─────────────────► │   PREPROCES  │ ──────────────► │ PLAY │
└──────────┘  (bytes WAV)  └─────┘   (texto)    └───────────┘      (texto)        │    OUTPUT    │  (bytes WAV)   └──────┘
                                                                                   └──────────────┘                   │
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

### 4. `PREPROCES OUTPUT` → `output_queue`

- Consume texto de `translated_queue`.
- Sintetiza voz con **gTTS** en el idioma de salida (MP3).
- **Acelera el audio un 30 %** usando `pydub` (técnica de resampling).
- Encola los **bytes WAV** listos para reproducir en `output_queue`.

> **Nota:** la aceleración usa resampling (ajuste de frame_rate), lo que también
> eleva ligeramente el tono de la voz. Para velocidades moderadas (1.2–1.5×) el
> resultado es perfectamente inteligible. El factor se puede cambiar con el
> parámetro `speed` en `preproces_output_process` (default `1.30`).

### 5. `PLAY`

- Consume bytes WAV de `output_queue`.
- Reproduce directamente con **pygame** en el dispositivo de salida seleccionado.
- No hace ningún procesamiento: sólo reproduce.

---

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y **ffmpeg** en el PATH del sistema (necesario para que `pydub` decodifique MP3):

```bash
# macOS
brew install ffmpeg

# Dependencias Python
uv sync
```

**Dependencias principales:**

| Paquete | Uso |
|---------|-----|
| `faster-whisper` | Transcripción local (STT) |
| `deep-translator` | Traducción vía Google |
| `gtts` | Síntesis de voz (TTS) |
| `pydub` | Aceleración de audio (+30 %) |
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

Luego se inician los 5 procesos y el pipeline comienza a funcionar.

Presiona **Ctrl+C** para detener todo de forma limpia.

---

## Logs en consola

Cada etapa imprime mensajes con su prefijo:

```
[CAPTURA]   Chunk → audio_queue  (48 320 bytes)
[STT]       (1.23s) → stt_queue: 'Hola, ¿cómo estás?'
[TRANSLATE] (0.45s) → translated_queue: 'Hello, how are you?'
[PREPROCES] gTTS 1.10s | acelerar 0.05s → output_queue (76 800 bytes)
[PLAY]      Reproduciendo (76 800 bytes)…
```

---

## Configuración avanzada

| Parámetro | Ubicación | Descripción |
|-----------|-----------|-------------|
| `energy_threshold` | `capture_process` | Sensibilidad del micrófono (default `300`) |
| `pause_threshold` | `capture_process` | Segundos de silencio para cerrar frase (default `0.8`) |
| `"small"` | `stt_process` | Tamaño del modelo Whisper (`tiny`, `small`, `medium`, `large`) |
| `speed=1.30` | `preproces_output_process` | Factor de aceleración del audio (1.0 = sin cambio) |
| `maxsize=20` | `audio_queue` | Máximo de chunks WAV en cola |
| `maxsize=50` | `stt_queue` / `translated_queue` | Máximo de textos en cola |
| `maxsize=10` | `output_queue` | Máximo de chunks WAV pre-procesados en cola |
