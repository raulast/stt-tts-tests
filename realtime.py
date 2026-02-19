"""
realtime.py
-----------
Dos procesos simultáneos:
  - Proceso 1 (captura): escucha el micrófono de forma continua y mete
    chunks de audio en una cola compartida.
  - Proceso 2 (procesamiento): saca chunks de la cola, los transcribe con
    Faster-Whisper, los traduce si hace falta con deep-translator y
    reproduce el resultado con gTTS + pygame.

Uso:
    python realtime.py

Al arrancar, el script pide al usuario que seleccione:
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

def select_devices_and_languages():
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
# Proceso 1: Captura de audio
# ---------------------------------------------------------------------------

def capture_process(audio_queue: mp.Queue, input_device_index: int, stop_event: mp.Event):
    """
    Escucha el micrófono de forma continua.
    Cada vez que detecta silencio (fin de frase) encola los bytes WAV del chunk.
    """
    print(f"[CAPTURA] Iniciando con dispositivo índice {input_device_index}")

    recognizer = sr.Recognizer()
    # Ajuste de sensibilidad: menor energy_threshold → más sensible
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    # Tiempo mínimo de silencio para considerar fin de frase (segundos)
    recognizer.pause_threshold = 0.7

    microphone = sr.Microphone(device_index=input_device_index)

    with microphone as source:
        print("[CAPTURA] Ajustando al ruido ambiente...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[CAPTURA] Listo. Escuchando...")

        while not stop_event.is_set():
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=30)
                wav_bytes = audio.get_wav_data()
                audio_queue.put(wav_bytes)
                print(f"[CAPTURA] Chunk encolado ({len(wav_bytes)} bytes)")
            except sr.WaitTimeoutError:
                # No se detectó audio en los últimos 5 s → volver a escuchar
                continue
            except Exception as exc:
                print(f"[CAPTURA] Error: {exc}")
                time.sleep(0.5)

    print("[CAPTURA] Detenido.")


# ---------------------------------------------------------------------------
# Proceso 2: Procesamiento (STT + traducción + TTS)
# ---------------------------------------------------------------------------

def process_process(
    audio_queue: mp.Queue,
    output_device_name: str,
    input_lang: str,
    output_lang: str,
    stop_event: mp.Event,
):
    """
    Saca chunks de la cola, los transcribe, traduce si es necesario y los
    reproduce con gTTS.
    """
    print("[PROCESO] Cargando modelo Faster-Whisper (small)...")
    model = WhisperModel("small", device="cpu", compute_type="float32")
    print("[PROCESO] Modelo cargado. Esperando audio...")

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            wav_bytes = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        # -- Transcripción --
        t0 = time.time()
        try:
            segments, _ = model.transcribe(
                BytesIO(wav_bytes),
                language=input_lang,
                vad_filter=True,           # filtra silencios con VAD
                vad_parameters={"min_silence_duration_ms": 500},
            )
            text = "".join(seg.text for seg in segments).strip()
        except Exception as exc:
            print(f"[PROCESO] Error en transcripción: {exc}")
            continue

        if not text:
            print("[PROCESO] Transcripción vacía, ignorando.")
            continue

        print(f"[PROCESO] Transcripción ({time.time()-t0:.2f}s): {text!r}")

        # -- Traducción --
        if input_lang != output_lang:
            t1 = time.time()
            try:
                text = GoogleTranslator(source=input_lang, target=output_lang).translate(text)
                print(f"[PROCESO] Traducción ({time.time()-t1:.2f}s): {text!r}")
            except Exception as exc:
                print(f"[PROCESO] Error en traducción: {exc}")

        # -- Síntesis gTTS --
        t2 = time.time()
        try:
            tts = gtts.gTTS(text=text, lang=output_lang)
            mp3_buf = BytesIO()
            tts.write_to_fp(mp3_buf)
            mp3_buf.seek(0)
            print(f"[PROCESO] gTTS ({time.time()-t2:.2f}s) → reproduciendo...")
        except Exception as exc:
            print(f"[PROCESO] Error en gTTS: {exc}")
            continue

        # -- Reproducción con pygame --
        try:
            _play_audio(mp3_buf, output_device_name)
        except Exception as exc:
            print(f"[PROCESO] Error reproduciendo audio: {exc}")

    print("[PROCESO] Detenido.")


def _play_audio(audio_buf: BytesIO, device_name: str):
    """Reproduce un buffer MP3/WAV con pygame en el dispositivo indicado."""
    if not pygame.mixer.get_init():
        pygame.mixer.init(devicename=device_name)
    pygame.mixer.music.load(audio_buf)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


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
    input("\nPresiona Enter para comenzar (Ctrl+C para detener)...\n")

    # Cola y evento compartidos entre procesos
    audio_queue: mp.Queue = mp.Queue(maxsize=50)
    stop_event: mp.Event = mp.Event()

    # Proceso 1: captura
    p_capture = mp.Process(
        target=capture_process,
        args=(audio_queue, config["input_device_index"], stop_event),
        name="Captura",
        daemon=True,
    )

    # Proceso 2: STT + traducción + TTS
    p_process = mp.Process(
        target=process_process,
        args=(
            audio_queue,
            config["output_device_name"],
            config["input_lang"],
            config["output_lang"],
            stop_event,
        ),
        name="Procesamiento",
        daemon=True,
    )

    p_capture.start()
    p_process.start()

    print("[MAIN] Ambos procesos activos. Ctrl+C para detener.")

    try:
        # Mantenemos el proceso principal vivo mientras los hijos corren
        while p_capture.is_alive() or p_process.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupción recibida. Deteniendo procesos...")
        stop_event.set()

    # Esperar a que terminen limpiamente (máx. 10 s cada uno)
    p_capture.join(timeout=10)
    p_process.join(timeout=10)

    # Forzar terminación si siguen vivos
    if p_capture.is_alive():
        p_capture.terminate()
    if p_process.is_alive():
        p_process.terminate()

    print("[MAIN] ¡Hasta luego!")


if __name__ == "__main__":
    # Necesario en macOS/Windows para multiprocessing con 'spawn'
    mp.set_start_method("spawn", force=True)
    main()
