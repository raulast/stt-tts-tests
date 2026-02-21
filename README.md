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

---

## Características principales

- **Multiprocesamiento**: 5 procesos aislados para latencia mínima.
- **Traducción**: Soporte para traducción en tiempo real usando Google Translate.
- **Doble motor TTS**: Soporte para **gTTS** (online) y **Piper** (offline/vía ONNX).
- **Aceleración dinámica**: El audio de salida se acelera un 30% (+1.30x) automáticamente.
- **Interfaz CLI**: Control total mediante flags para automatización y personalización.

---

## Instalación

```bash
# Sincronizar dependencias
uv sync
```

> **Requisito del sistema:** `ffmpeg` debe estar en el PATH para la decodificación y aceleración de audio (`brew install ffmpeg`).

---

## Uso

### Inicio rápido (interactivo)
```bash
uv run python realtime.py
```

### Automatización por Flags
Puedes saltarte los menús interactivos especificando los parámetros:
```bash
uv run python realtime.py --mic 0 --speaker "MacBook Pro Speakers" --input-lang es --output-lang en --tts piper --skip-enter
```

### Comandos de listado
Úsalos para identificar tus dispositivos antes de lanzar el pipeline:
- `uv run python realtime.py --list-mics`
- `uv run python realtime.py --list-speakers`
- `uv run python realtime.py --list-langs`
- `uv run python realtime.py --list-tts`

---

## Proveedores de TTS

### gTTS (Default)
- **Pro**: Alta calidad, no requiere modelos locales.
- **Contra**: Requiere internet, latencia de red.

### Piper
- **Pro**: Ejecución local (offline), baja latencia constante.
- **Contra**: Requiere modelos `.onnx` en el directorio `piper-models/`.
- **Modelos actuales**: `es` (Dave) y `en` (Joe) en calidad `medium`.

---

## Logs en consola

```
[CAPTURA]   Chunk → audio_queue (48,320 bytes)
[STT]       (1.23s) → stt_queue: 'Hola, ¿cómo estás?'
[TRANSLATE] (0.45s) → translated_queue: 'Hello, how are you?'
[PREPROCES] TTS 0.15s | acelerar 0.05s → output_queue (76,800 bytes)
[PLAY]      Reproduciendo (76,800 bytes)…
```
