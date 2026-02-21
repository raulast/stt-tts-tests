"""
realtime.py
-----------
Cinco procesos completamente separados:

  CAPTURA → audio_queue → STT → stt_queue → TRANSLATE → translated_queue
          → PREPROCES OUTPUT → output_queue → PLAY

  - [CAPTURA]          : escucha el micrófono; encola chunks WAV.
  - [STT]              : transcribe con Faster-Whisper; encola el texto.
  - [TRANSLATE]        : traduce si input_lang ≠ output_lang; encola texto final.
  - [PREPROCES OUTPUT] : sintetiza voz con gTTS o Piper y acelera el audio un 30%;
                         encola los bytes WAV listos para reproducir.
  - [PLAY]             : reproduce los bytes WAV recibidos con pygame.

Uso:
    python realtime.py [flags]

Flags disponibles:
  --list-mics          : Lista los micrófonos disponibles y sale.
  --list-speakers      : Lista los altavoces disponibles y sale.
  --list-langs         : Lista los idiomas soportados y sale.
  --list-tts           : Lista los proveedores de TTS y sale.
  --mic INDEX          : Índice del micrófono a usar.
  --speaker INDEX/NAME : Índice o nombre del altavoz a usar.
  --input-lang LANG    : Idioma de entrada (es, en, etc).
  --output-lang LANG   : Idioma de salida (es, en, etc).
  --tts PROVIDER       : Proveedor de TTS (gtts, piper).
  --skip-enter         : Salta el mensaje de "Presione Enter para comenzar".

Ctrl+C para detener.
"""

import os
import sys
import argparse
import time
import queue
import multiprocessing as mp
from io import BytesIO
import wave

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import pygame._sdl2.audio as sdl2_audio
import speech_recognition as sr
import gtts
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from pydub import AudioSegment

# Intentar importar piper. Si no está, el proveedor 'piper' fallará en ejecución.
try:
    from piper.voice import PiperVoice
    from piper.download import download_voice
except ImportError:
    PiperVoice = None


# ---------------------------------------------------------------------------
# Configuración y Constantes
# ---------------------------------------------------------------------------

LANGS = {"es": "Español", "en": "English"}
TTS_PROVIDERS = ["gtts", "piper"]
PIPER_MODELS_DIR = "piper-models"

# Mapeo de idiomas a modelos de Piper (basado en la estructura de archivos vista)
PIPER_VOICES = {
    "es": os.path.join(PIPER_MODELS_DIR, "es", "es_ES-davefx-medium.onnx"),
    "en": os.path.join(PIPER_MODELS_DIR, "en", "en_US-joe-medium.onnx"),
}


# ---------------------------------------------------------------------------
# Helpers de audio y sistema
# ---------------------------------------------------------------------------

def _speed_up_audio(audio_bytes: bytes, speed: float = 1.30, format="wav") -> bytes:
    """
    Acelera el audio (WAV o MP3) `speed` veces usando resampling.
    """
    if format == "mp3":
        audio = AudioSegment.from_mp3(BytesIO(audio_bytes))
    else:
        audio = AudioSegment.from_wav(BytesIO(audio_bytes))
        
    faster = audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)},
    ).set_frame_rate(audio.frame_rate)
    
    out = BytesIO()
    faster.export(out, format="wav")
    return out.getvalue()


def list_microphones():
    print("\n=== Micrófonos disponibles ===")
    mic_names = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mic_names):
        print(f"  [{i}] {name}")


def list_speakers():
    print("\n=== Altavoces disponibles ===")
    pygame.mixer.init()
    output_device_names = tuple(sdl2_audio.get_audio_device_names(False))
    pygame.mixer.quit()
    for i, name in enumerate(output_device_names):
        print(f"  [{i}] {name}")


def list_languages():
    print("\n=== Idiomas soportados ===")
    for code, name in LANGS.items():
        print(f"  [{code}] {name}")


def list_tts_providers():
    print("\n=== Proveedores de TTS ===")
    for provider in TTS_PROVIDERS:
        status = "disponible"
        if provider == "piper" and PiperVoice is None:
            status = "NO INSTALADO (instale 'piper-tts')"
        print(f"  - {provider} ({status})")


# ---------------------------------------------------------------------------
# Selección de dispositivos e idiomas
# ---------------------------------------------------------------------------

def get_config(args) -> dict:
    """Combina flags de CLI con selección interactiva."""
    
    config = {}

    # --- Micrófono ---
    # mic_names = sr.Microphone.list_microphone_names()
    if args.mic is not None:
        config["input_device_index"] = args.mic
    else:
        list_microphones()
        config["input_device_index"] = int(input("Seleccione índice del micrófono: ").strip())

    # --- Altavoz ---
    pygame.mixer.init()
    speakers = tuple(sdl2_audio.get_audio_device_names(False))
    pygame.mixer.quit()
    
    if args.speaker is not None:
        try:
            # Intentar como índice
            idx = int(args.speaker)
            config["output_device_name"] = speakers[idx]
        except (ValueError, IndexError):
            # Intentar como nombre
            config["output_device_name"] = args.speaker
    else:
        list_speakers()
        idx = int(input("Seleccione índice del altavoz: ").strip())
        config["output_device_name"] = speakers[idx]

    # --- Idiomas ---
    if args.input_lang and args.input_lang in LANGS:
        config["input_lang"] = args.input_lang
    else:
        list_languages()
        config["input_lang"] = input("Idioma de entrada (código): ").strip().lower()

    if args.output_lang and args.output_lang in LANGS:
        config["output_lang"] = args.output_lang
    else:
        list_languages()
        config["output_lang"] = input("Idioma de salida (código): ").strip().lower()

    # --- TTS Provider ---
    if args.tts and args.tts in TTS_PROVIDERS:
        config["tts_provider"] = args.tts
    else:
        list_tts_providers()
        config["tts_provider"] = input("Proveedor de TTS (gtts/piper): ").strip().lower()

    return config


# ---------------------------------------------------------------------------
# ETAPA 1 — CAPTURA: Micrófono → audio_queue
# ---------------------------------------------------------------------------

def capture_process(
    audio_queue: mp.Queue,
    input_device_index: int,
    stop_event: mp.Event,
):
    print(f"[CAPTURA] Iniciando — micrófono índice {input_device_index}")
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    try:
        microphone = sr.Microphone(device_index=input_device_index)
        with microphone as source:
            print("[CAPTURA] Ajustando al ruido ambiente (1 s)…")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[CAPTURA] Listo. Escuchando…\n")

            while not stop_event.is_set():
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    wav_bytes = audio.get_wav_data()
                    audio_queue.put(wav_bytes)
                    print(f"[CAPTURA] Chunk → audio_queue  ({len(wav_bytes):,} bytes)")
                except sr.WaitTimeoutError:
                    continue
                except Exception as exc:
                    print(f"[CAPTURA] Error en escucha: {exc}")
                    time.sleep(0.5)
    except Exception as e:
        print(f"[CAPTURA] Error fatal: {e}")
        stop_event.set()

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
    needs_translation = input_lang != output_lang
    if needs_translation:
        print(f"[TRANSLATE] Modo traducción: {input_lang} → {output_lang}\n")
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
# ETAPA 4 — PREPROCES OUTPUT: translated_queue → TTS + speed_up → output_queue
# ---------------------------------------------------------------------------

def preproces_output_process(
    translated_queue: mp.Queue,
    output_queue: mp.Queue,
    output_lang: str,
    tts_provider: str,
    stop_event: mp.Event,
    speed: float = 1.30,
):
    print(f"[PREPROCES] Proveedor: {tts_provider} | Velocidad ×{speed:.2f}\n")
    
    # Cargar Piper si es necesario
    piper_voice = None
    if tts_provider == "piper":
        if PiperVoice is None:
            print("[PREPROCES] ERROR: Piper no está instalado. Usando gTTS como fallback.")
            tts_provider = "gtts"
        else:
            model_path = PIPER_VOICES.get(output_lang)
            if not model_path or not os.path.exists(model_path):
                print(f"[PREPROCES] ERROR: No se encontró modelo Piper para '{output_lang}'.")
                print(f"            Buscado en: {model_path}")
                folder = os.path.join(PIPER_MODELS_DIR, output_lang)
                onxFile = model_path.split(os.sep)[-1].replace(".onnx", "")
                os.makedirs(folder, exist_ok=True)
                print(f"[PREPROCES] Carpeta creada: {folder}")
                download_voice(output_lang, folder)
                model_path = os.path.join(folder, onxFile + ".onnx")
            
            print(f"[PREPROCES] Cargando voz Piper: {os.path.basename(model_path)}")
            piper_voice = PiperVoice.load(model_path)

    while not stop_event.is_set() or not translated_queue.empty():
        try:
            text = translated_queue.get(timeout=1)
        except queue.Empty:
            continue

        print(f"[PREPROCES] Sintetizando ({tts_provider}): {text!r}")
        t0 = time.time()
        
        try:
            if tts_provider == "piper" and piper_voice:
                # Piper genera audio directamente en un archivo o buffer WAV
                wav_file = BytesIO()
                piper_voice.synthesize_wav(text, wave.Wave_write(wav_file))
                wav_file.seek(0)
                raw_audio = wav_file.getvalue()
                fmt = "wav"
            else:
                # gTTS (default)
                tts = gtts.gTTS(text=text, lang=output_lang)
                mp3_buf = BytesIO()
                tts.write_to_fp(mp3_buf)
                raw_audio = mp3_buf.getvalue()
                fmt = "mp3"
        except Exception as exc:
            print(f"[PREPROCES] Error en TTS ({tts_provider}): {exc}")
            continue

        dt_tts = time.time() - t0

        # Acelerar audio
        t1 = time.time()
        try:
            wav_bytes = _speed_up_audio(raw_audio, speed=speed, format=fmt)
        except Exception as exc:
            print(f"[PREPROCES] Error acelerando audio: {exc} — usando original")
            wav_bytes = raw_audio

        dt_speed = time.time() - t1
        print(
            f"[PREPROCES] TTS {dt_tts:.2f}s | acelerar {dt_speed:.2f}s "
            f"→ output_queue ({len(wav_bytes):,} bytes)"
        )
        output_queue.put(wav_bytes)

    print("[PREPROCES] Detenido.")


# ---------------------------------------------------------------------------
# ETAPA 5 — PLAY: output_queue → pygame → altavoz
# ---------------------------------------------------------------------------

def play_process(
    output_queue: mp.Queue,
    output_device_name: str,
    stop_event: mp.Event,
):
    print(f"[PLAY] Listo — altavoz: {output_device_name!r}\n")
    pygame.mixer.init(devicename=output_device_name)

    while not stop_event.is_set() or not output_queue.empty():
        try:
            audio_bytes = output_queue.get(timeout=1)
        except queue.Empty:
            continue

        print(f"[PLAY] Reproduciendo ({len(audio_bytes):,} bytes)…")
        try:
            pygame.mixer.music.load(BytesIO(audio_bytes))
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
    parser = argparse.ArgumentParser(description="Traductor de voz en tiempo real con 5 etapas.")
    parser.add_argument("--list-mics", action="store_true", help="Lista micrófonos y sale.")
    parser.add_argument("--list-speakers", action="store_true", help="Lista altavoces y sale.")
    parser.add_argument("--list-langs", action="store_true", help="Lista idiomas soportados y sale.")
    parser.add_argument("--list-tts", action="store_true", help="Lista proveedores de TTS y sale.")
    
    parser.add_argument("--mic", type=int, help="Índice del micrófono.")
    parser.add_argument("--speaker", type=str, help="Índice o nombre del altavoz.")
    parser.add_argument("--input-lang", type=str, help="Idioma de entrada (código, ej: es).")
    parser.add_argument("--output-lang", type=str, help="Idioma de salida (código, ej: en).")
    parser.add_argument("--tts", type=str, choices=TTS_PROVIDERS, help="Proveedor de TTS.")
    parser.add_argument("--skip-enter", action="store_true", help="No esperar Enter para comenzar.")
    
    args = parser.parse_args()

    # Comandos de listado
    if args.list_mics:
        list_microphones()
        sys.exit(0)
    if args.list_speakers:
        list_speakers()
        sys.exit(0)
    if args.list_langs:
        list_languages()
        sys.exit(0)
    if args.list_tts:
        list_tts_providers()
        sys.exit(0)

    # Configuración (flags + interactivo)
    config = get_config(args)

    print("\n=== Configuración seleccionada ===")
    print(f"  Micrófono índice  : {config['input_device_index']}")
    print(f"  Altavoz           : {config['output_device_name']}")
    print(f"  Idioma entrada    : {config['input_lang']}")
    print(f"  Idioma salida     : {config['output_lang']}")
    print(f"  TTS Provider      : {config['tts_provider']}")
    
    if not args.skip_enter:
        input("\nPresiona Enter para comenzar (Ctrl+C para detener)…\n")
    else:
        print("\nIniciando automáticamente…\n")

    # Buffers compartidos
    audio_queue:      mp.Queue = mp.Queue(maxsize=20)   # bytes WAV (captura)
    stt_queue:        mp.Queue = mp.Queue(maxsize=50)   # texto transcripto
    translated_queue: mp.Queue = mp.Queue(maxsize=50)   # texto (traducido o no)
    output_queue:     mp.Queue = mp.Queue(maxsize=10)   # bytes WAV (audio final)

    stop_event: mp.Event = mp.Event()

    # Procesos
    p_capture = mp.Process(
        target=capture_process,
        args=(audio_queue, config["input_device_index"], stop_event),
        name="Captura", daemon=True
    )
    p_stt = mp.Process(
        target=stt_process,
        args=(audio_queue, stt_queue, config["input_lang"], stop_event),
        name="STT", daemon=True
    )
    p_translate = mp.Process(
        target=translate_process,
        args=(stt_queue, translated_queue, config["input_lang"], config["output_lang"], stop_event),
        name="Translate", daemon=True
    )
    p_preproces = mp.Process(
        target=preproces_output_process,
        args=(translated_queue, output_queue, config["output_lang"], config["tts_provider"], stop_event),
        name="PreprocesOutput", daemon=True
    )
    p_play = mp.Process(
        target=play_process,
        args=(output_queue, config["output_device_name"], stop_event),
        name="Play", daemon=True
    )

    # Arranque
    for p in (p_capture, p_stt, p_translate, p_preproces, p_play):
        p.start()

    print("[MAIN] Pipeline activo (5 procesos). Ctrl+C para detener.\n")

    try:
        while any(p.is_alive() for p in (p_capture, p_stt, p_translate, p_preproces, p_play)):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupción recibida. Deteniendo pipeline…")
        stop_event.set()

    # Join
    for p in (p_capture, p_stt, p_translate, p_preproces, p_play):
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    print("[MAIN] ¡Hasta luego!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
