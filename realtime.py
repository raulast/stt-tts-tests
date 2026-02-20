"""
realtime.py
-----------
Cuatro procesos completamente separados:

  CAPTURA  →  audio_queue  →  STT  →  stt_queue  →  TRANSLATE  →  translated_queue  →  PLAY

  - [CAPTURA]   : escucha el micrófono de forma continua; encola chunks WAV
                  cuando detecta fin de frase (silencio) o tras el límite de tiempo.
  - [STT]       : transcribe cada chunk WAV con Faster-Whisper y encola el texto.
  - [TRANSLATE] : si input_lang ≠ output_lang traduce el texto; de lo contrario lo
                  pasa sin modificar. Encola el texto final.
  - [PLAY]      : sintetiza voz con gTTS y reproduce con pygame.

Uso:
    python realtime.py

Al arrancar se pide:
  - Dispositivo de entrada (micrófono)
  - Dispositivo de salida (altavoz)
  - Idioma de entrada  (es | en)
  - Idioma de salida   (es | en)

Ctrl+C para detener.
"""

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import time
import queue
import multiprocessing as mp
from io import BytesIO

import pygame
import pygame._sdl2.audio as sdl2_audio
import speech_recognition as sr
import gtts
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator


# ---------------------------------------------------------------------------
# Selección de dispositivos e idiomas (proceso principal)
# ---------------------------------------------------------------------------

def select_devices_and_languages() -> dict:
    """Muestra menús de selección y devuelve la configuración elegida."""

    # --- Micrófono ---
    mic_names = sr.Microphone.list_microphone_names()
    print("\n=== Selecciona un micrófono ===")
    for i, name in enumerate(mic_names):
        print(f"  [{i}] {name}")
    input_device_index = int(input("Índice del micrófono: ").strip())

    # --- Altavoz ---
    pygame.mixer.init()
    output_device_names = tuple(sdl2_audio.get_audio_device_names(False))
    pygame.mixer.quit()

    print("\n=== Selecciona un altavoz ===")
    for i, name in enumerate(output_device_names):
        print(f"  [{i}] {name}")
    output_device_index = int(input("Índice del altavoz: ").strip())
    output_device_name = output_device_names[output_device_index]

    # --- Idiomas ---
    langs = {"es": "Español", "en": "English"}
    print("\n=== Idioma de entrada ===")
    for code, label in langs.items():
        print(f"  [{code}] {label}")
    input_lang = input("Idioma de entrada: ").strip().lower()

    print("\n=== Idioma de salida ===")
    for code, label in langs.items():
        print(f"  [{code}] {label}")
    output_lang = input("Idioma de salida: ").strip().lower()

    return {
        "input_device_index": input_device_index,
        "output_device_name": output_device_name,
        "input_lang": input_lang,
        "output_lang": output_lang,
    }


# ---------------------------------------------------------------------------
# ETAPA 1 — CAPTURA: Micrófono → audio_queue
# ---------------------------------------------------------------------------

def capture_process(
    audio_queue: mp.Queue,
    input_device_index: int,
    stop_event: mp.Event,
):
    """
    Escucha el micrófono de forma continua.
    Encola los bytes WAV de cada chunk detectado en audio_queue.
    """
    print(f"[CAPTURA] Iniciando — micrófono índice {input_device_index}")

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    # Fin de frase tras 0.8 s de silencio
    recognizer.pause_threshold = 0.8

    microphone = sr.Microphone(device_index=input_device_index)

    with microphone as source:
        print("[CAPTURA] Ajustando al ruido ambiente (1 s)…")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[CAPTURA] Listo. Escuchando…\n")

        while not stop_event.is_set():
            try:
                # Espera hasta 5 s para detectar audio; acepta frases de hasta 30 s
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=30)
                wav_bytes = audio.get_wav_data()
                audio_queue.put(wav_bytes)
                print(f"[CAPTURA] Chunk → audio_queue  ({len(wav_bytes):,} bytes)")
            except sr.WaitTimeoutError:
                # Sin audio en los últimos 5 s → reintentar
                continue
            except Exception as exc:
                print(f"[CAPTURA] Error: {exc}")
                time.sleep(0.5)

    print("[CAPTURA] Detenido.")


# ---------------------------------------------------------------------------
# ETAPA 2 — STT: audio_queue → WhisperModel → stt_queue
# ---------------------------------------------------------------------------

def stt_process(
    audio_queue: mp.Queue,
    stt_queue: mp.Queue,
    input_lang: str,
    stop_event: mp.Event,
):
    """
    Toma chunks WAV de audio_queue, los transcribe con Faster-Whisper
    y pone el texto resultante en stt_queue.
    """
    print("[STT] Cargando modelo Faster-Whisper (small)…")
    model = WhisperModel("small", device="cpu", compute_type="float32")
    print("[STT] Modelo listo. Esperando audio…\n")

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            wav_bytes = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        t0 = time.time()
        try:
            segments, _ = model.transcribe(
                BytesIO(wav_bytes),
                language=input_lang,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            text = "".join(seg.text for seg in segments).strip()
        except Exception as exc:
            print(f"[STT] Error en transcripción: {exc}")
            continue

        if not text:
            print("[STT] Transcripción vacía — ignorando.")
            continue

        dt = time.time() - t0
        print(f"[STT] ({dt:.2f}s) → stt_queue: {text!r}")
        stt_queue.put(text)

    print("[STT] Detenido.")


# ---------------------------------------------------------------------------
# ETAPA 3 — TRANSLATE: stt_queue → (GoogleTranslator) → translated_queue
# ---------------------------------------------------------------------------

def translate_process(
    stt_queue: mp.Queue,
    translated_queue: mp.Queue,
    input_lang: str,
    output_lang: str,
    stop_event: mp.Event,
):
    """
    Toma texto de stt_queue.
    - Si input_lang == output_lang: pasa el texto sin modificar.
    - Si difieren: traduce con GoogleTranslator.
    Coloca el resultado en translated_queue.
    """
    needs_translation = input_lang != output_lang
    if needs_translation:
        print(f"[TRANSLATE] Modo traducción activo: {input_lang} → {output_lang}\n")
    else:
        print(f"[TRANSLATE] Sin traducción (entrada=salida={input_lang})\n")

    while not stop_event.is_set() or not stt_queue.empty():
        try:
            text = stt_queue.get(timeout=1)
        except queue.Empty:
            continue

        if needs_translation:
            t0 = time.time()
            try:
                text = GoogleTranslator(
                    source=input_lang, target=output_lang
                ).translate(text)
                dt = time.time() - t0
                print(f"[TRANSLATE] ({dt:.2f}s) → translated_queue: {text!r}")
            except Exception as exc:
                print(f"[TRANSLATE] Error en traducción: {exc} — propagando original")
        else:
            print(f"[TRANSLATE] Passthrough → translated_queue: {text!r}")

        translated_queue.put(text)

    print("[TRANSLATE] Detenido.")


# ---------------------------------------------------------------------------
# ETAPA 4 — PLAY: translated_queue → gTTS → pygame → altavoz
# ---------------------------------------------------------------------------

def play_process(
    translated_queue: mp.Queue,
    output_device_name: str,
    output_lang: str,
    stop_event: mp.Event,
):
    """
    Toma texto de translated_queue, sintetiza voz con gTTS y lo reproduce
    a través de pygame usando el dispositivo de salida indicado.
    """
    print(f"[PLAY] Listo — altavoz: {output_device_name!r}\n")

    # Inicializar pygame.mixer una sola vez en este proceso
    pygame.mixer.init(devicename=output_device_name)

    while not stop_event.is_set() or not translated_queue.empty():
        try:
            text = translated_queue.get(timeout=1)
        except queue.Empty:
            continue

        print(f"[PLAY] Sintetizando: {text!r}")

        # Síntesis gTTS
        t0 = time.time()
        try:
            tts = gtts.gTTS(text=text, lang=output_lang)
            mp3_buf = BytesIO()
            tts.write_to_fp(mp3_buf)
            mp3_buf.seek(0)
        except Exception as exc:
            print(f"[PLAY] Error en gTTS: {exc}")
            continue

        dt_tts = time.time() - t0
        print(f"[PLAY] gTTS listo ({dt_tts:.2f}s) — reproduciendo…")

        # Reproducción
        try:
            pygame.mixer.music.load(mp3_buf)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(20)
        except Exception as exc:
            print(f"[PLAY] Error reproduciendo: {exc}")

    pygame.mixer.quit()
    print("[PLAY] Detenido.")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    config = select_devices_and_languages()

    print("\n=== Configuración seleccionada ===")
    print(f"  Micrófono índice  : {config['input_device_index']}")
    print(f"  Altavoz           : {config['output_device_name']}")
    print(f"  Idioma entrada    : {config['input_lang']}")
    print(f"  Idioma salida     : {config['output_lang']}")
    input("\nPresiona Enter para comenzar (Ctrl+C para detener)…\n")

    # ── Buffers compartidos entre etapas ──────────────────────────────────────
    audio_queue:      mp.Queue = mp.Queue(maxsize=20)   # bytes WAV
    stt_queue:        mp.Queue = mp.Queue(maxsize=50)   # texto transcripto
    translated_queue: mp.Queue = mp.Queue(maxsize=50)   # texto (traducido o no)

    # ── Señal de parada global ────────────────────────────────────────────────
    stop_event: mp.Event = mp.Event()

    # ── Definición de los 4 procesos ─────────────────────────────────────────
    p_capture = mp.Process(
        target=capture_process,
        args=(audio_queue, config["input_device_index"], stop_event),
        name="Captura",
        daemon=True,
    )

    p_stt = mp.Process(
        target=stt_process,
        args=(audio_queue, stt_queue, config["input_lang"], stop_event),
        name="STT",
        daemon=True,
    )

    p_translate = mp.Process(
        target=translate_process,
        args=(
            stt_queue,
            translated_queue,
            config["input_lang"],
            config["output_lang"],
            stop_event,
        ),
        name="Translate",
        daemon=True,
    )

    p_play = mp.Process(
        target=play_process,
        args=(
            translated_queue,
            config["output_device_name"],
            config["output_lang"],
            stop_event,
        ),
        name="Play",
        daemon=True,
    )

    # ── Arranque ──────────────────────────────────────────────────────────────
    for p in (p_capture, p_stt, p_translate, p_play):
        p.start()

    print("[MAIN] Pipeline activo (4 procesos). Ctrl+C para detener.\n")
    print("       CAPTURA → audio_queue → STT → stt_queue → TRANSLATE → translated_queue → PLAY\n")

    try:
        while any(p.is_alive() for p in (p_capture, p_stt, p_translate, p_play)):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupción recibida. Deteniendo pipeline…")
        stop_event.set()

    # ── Esperar terminación limpia (máx. 10 s cada proceso) ──────────────────
    for p in (p_capture, p_stt, p_translate, p_play):
        p.join(timeout=10)
        if p.is_alive():
            print(f"[MAIN] Forzando terminación de {p.name}…")
            p.terminate()

    print("[MAIN] ¡Hasta luego!")


if __name__ == "__main__":
    # Necesario en macOS/Windows para multiprocessing con 'spawn'
    mp.set_start_method("spawn", force=True)
    main()
