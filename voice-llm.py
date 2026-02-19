"""
voice-llm.py
-------------
Conversación fluida por voz con un modelo LLM local (Ollama).

Arquitectura con 3 hilos:
  - Hilo principal: orquesta el flujo (escuchar → LLM → hablar)
  - Hilo de escucha: monitorea el micrófono INCLUSO durante la reproducción
    de audio para detectar interrupciones.
  - Hilo de reproducción: reproduce el audio gTTS de la respuesta del LLM.

Cuando el usuario habla mientras el asistente reproduce audio, la reproducción
se interrumpe inmediatamente y se procesa la nueva frase del usuario.

Modelo: minimax-m2.5:cloud (Ollama local en http://localhost:11434)

Uso:
    python voice-llm.py

Ctrl+C para salir.
"""

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import sys
import time
import json
import threading
from io import BytesIO

import requests
import pygame
import pygame._sdl2.audio as sdl2_audio
import speech_recognition as sr
import gtts
from faster_whisper import WhisperModel


# ─── Configuración global ────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "minimax-m2.5:cloud"
WHISPER_MODEL_SIZE = "small"
LANGUAGE = "es"                       # idioma de voz (entrada y salida por defecto)

# ─── Estado compartido entre hilos ────────────────────────────────────────────

class SharedState:
    """Estado mutable compartido entre hilos, protegido por locks."""

    def __init__(self):
        self.lock = threading.Lock()
        # Control de reproducción
        self.is_playing = False
        self.interrupted = False
        # Audio capturado durante interrupción
        self.interrupt_audio: bytes | None = None
        # Señal de parada global
        self.stop = False


# ─── Selección de dispositivos ────────────────────────────────────────────────

def select_devices():
    """Selecciona micrófono y altavoz interactivamente."""

    mic_names = sr.Microphone.list_microphone_names()
    print("\n=== Micrófono ===")
    for i, name in enumerate(mic_names):
        print(f"  [{i}] {name}")
    mic_idx = int(input("Índice: ").strip())

    pygame.mixer.init()
    speakers = tuple(sdl2_audio.get_audio_device_names(False))
    pygame.mixer.quit()

    print("\n=== Altavoz ===")
    for i, name in enumerate(speakers):
        print(f"  [{i}] {name}")
    spk_idx = int(input("Índice: ").strip())

    return mic_idx, speakers[spk_idx]


# ─── Whisper STT ──────────────────────────────────────────────────────────────

def transcribe(model: WhisperModel, wav_bytes: bytes, lang: str = LANGUAGE) -> str:
    """Transcribe audio WAV bytes → texto usando Faster-Whisper."""
    segments, _ = model.transcribe(
        BytesIO(wav_bytes),
        language=lang,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    return "".join(seg.text for seg in segments).strip()


# ─── Ollama LLM chat ─────────────────────────────────────────────────────────

def chat_ollama(messages: list[dict], state: SharedState) -> str:
    """
    Envía la conversación a Ollama y obtiene la respuesta con streaming.
    Si se interrumpe (state.interrupted), deja de consumir tokens y devuelve
    lo acumulado hasta ese momento.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
    }

    full_response = []
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if state.interrupted or state.stop:
                    break
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_response.append(token)
                    print(token, end="", flush=True)
                if chunk.get("done"):
                    break
    except requests.RequestException as exc:
        print(f"\n[LLM] Error de conexión con Ollama: {exc}")
        return ""

    print()  # nueva línea tras el streaming
    return "".join(full_response).strip()


# ─── Reproducción con interrupción ────────────────────────────────────────────

def play_tts(text: str, lang: str, speaker: str, state: SharedState):
    """
    Sintetiza voz con gTTS y reproduce. Puede ser interrumpida si
    state.interrupted se activa.
    """
    if not text:
        return

    # Generar audio
    try:
        tts = gtts.gTTS(text=text, lang=lang)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
    except Exception as exc:
        print(f"[TTS] Error gTTS: {exc}")
        return

    # Reproducir
    with state.lock:
        state.is_playing = True

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(devicename=speaker)
        pygame.mixer.music.load(buf)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            if state.interrupted:
                pygame.mixer.music.stop()
                print("\n[TTS] ⚡ Reproducción interrumpida por el usuario.")
                break
            pygame.time.Clock().tick(20)
    except Exception as exc:
        print(f"[TTS] Error reproducción: {exc}")
    finally:
        with state.lock:
            state.is_playing = False


# ─── Hilo de escucha continua (detecta interrupciones) ───────────────────────

def interrupt_listener(
    mic_idx: int,
    whisper_model: WhisperModel,
    state: SharedState,
):
    """
    Escucha continuamente el micrófono. Si detecta voz mientras el TTS
    está reproduciendo, marca la interrupción y guarda el audio capturado.
    Si no se está reproduciendo, ignora (el hilo principal gestiona la escucha).
    """
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.7

    mic = sr.Microphone(device_index=mic_idx)

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        while not state.stop:
            # Solo escuchamos para interrumpir cuando el TTS reproduce
            with state.lock:
                playing = state.is_playing

            if not playing:
                time.sleep(0.1)
                continue

            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=15)
                wav_bytes = audio.get_wav_data()

                # Solo interrumpimos si seguimos reproduciendo
                with state.lock:
                    if state.is_playing:
                        state.interrupted = True
                        state.interrupt_audio = wav_bytes
            except sr.WaitTimeoutError:
                continue
            except Exception:
                time.sleep(0.2)


# ─── Escucha normal (cuando NO se reproduce audio) ───────────────────────────

def listen_once(mic_idx: int, recognizer: sr.Recognizer) -> bytes | None:
    """Escucha una frase y devuelve los bytes WAV, o None si timeout."""
    mic = sr.Microphone(device_index=mic_idx)
    with mic as source:
        try:
            print("\n🎤 Escuchando...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
            return audio.get_wav_data()
        except sr.WaitTimeoutError:
            return None


# ─── Bucle principal de conversación ──────────────────────────────────────────

def conversation_loop(mic_idx: int, speaker: str, state: SharedState):
    """Bucle: escuchar → transcribir → LLM → hablar."""

    print("\n[INIT] Cargando modelo Whisper...")
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="float32")
    print("[INIT] Modelo Whisper listo.\n")

    # Hilo de escucha para interrupciones
    interrupt_thread = threading.Thread(
        target=interrupt_listener,
        args=(mic_idx, whisper_model, state),
        daemon=True,
    )
    interrupt_thread.start()

    # Historial de mensajes para contexto del LLM
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "Eres un asistente de voz amigable y conciso. "
                "Responde siempre en español de manera natural y breve, "
                "como si fuera una conversación hablada. "
                "Evita listas largas y markdown. Sé directo."
            ),
        }
    ]

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.7

    print("=" * 50)
    print("  🗣️  Conversación activa — habla cuando quieras")
    print("  ⚡ Puedes interrumpir al asistente hablando")
    print("  ❌ Ctrl+C para salir")
    print("=" * 50)

    while not state.stop:
        # ── Paso 1: obtener audio del usuario ──
        wav_bytes = None

        # ¿Hay audio de una interrupción pendiente?
        with state.lock:
            if state.interrupt_audio:
                wav_bytes = state.interrupt_audio
                state.interrupt_audio = None
                state.interrupted = False

        if not wav_bytes:
            wav_bytes = listen_once(mic_idx, recognizer)

        if not wav_bytes:
            continue  # timeout sin audio

        # ── Paso 2: transcribir ──
        t0 = time.time()
        text = transcribe(whisper_model, wav_bytes)
        dt = time.time() - t0

        if not text:
            print("  (no se detectó texto)")
            continue

        print(f"\n👤 Usuario ({dt:.1f}s): {text}")

        # Comandos de salida
        if text.strip().lower() in ("salir", "adiós", "exit", "quit", "bye"):
            play_tts("¡Hasta luego!", LANGUAGE, speaker, state)
            state.stop = True
            break

        # ── Paso 3: enviar al LLM ──
        messages.append({"role": "user", "content": text})

        print("🤖 Asistente: ", end="", flush=True)
        response = chat_ollama(messages, state)

        if not response:
            print("  (sin respuesta del LLM)")
            continue

        # Si el LLM fue interrumpido, aún guardamos lo que generó
        messages.append({"role": "assistant", "content": response})

        # ── Paso 4: reproducir respuesta (interrumpible) ──
        if not state.interrupted:
            play_tts(response, LANGUAGE, speaker, state)

        # Limpiar estado de interrupción para el próximo ciclo
        # (el audio de interrupción se procesará al inicio del bucle)

    print("\n[MAIN] Conversación terminada.")


# ─── Punto de entrada ────────────────────────────────────────────────────────

def main():
    print("╔═══════════════════════════════════════════════╗")
    print("║         🎙️  Voice LLM — Ollama Chat          ║")
    print("║   Modelo: minimax-m2.5:cloud                 ║")
    print("╚═══════════════════════════════════════════════╝")

    # Verificar que Ollama esté corriendo
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"\n⚠️  Modelo '{OLLAMA_MODEL}' no encontrado en Ollama.")
            print(f"   Modelos disponibles: {', '.join(models) or '(ninguno)'}")
            print(f"   Ejecuta: ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
        print(f"\n✅ Ollama activo — modelo '{OLLAMA_MODEL}' disponible")
    except requests.RequestException:
        print(f"\n❌ No se pudo conectar a Ollama en {OLLAMA_URL}")
        print("   Asegúrate de que Ollama esté corriendo: ollama serve")
        sys.exit(1)

    mic_idx, speaker = select_devices()
    state = SharedState()

    print(f"\n  Micrófono : índice {mic_idx}")
    print(f"  Altavoz   : {speaker}")
    input("\nPresiona Enter para comenzar...\n")

    try:
        conversation_loop(mic_idx, speaker, state)
    except KeyboardInterrupt:
        print("\n\n[MAIN] ¡Hasta luego! 👋")
        state.stop = True


if __name__ == "__main__":
    main()
