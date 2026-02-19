import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
from faster_whisper import WhisperModel
import speech_recognition as sr
import wave
from piper.voice import PiperVoice
from piper.download_voices import download_voice
import gtts
import time
import pygame
import pygame._sdl2.audio as sdl2_audio
from io import BytesIO
from deep_translator import GoogleTranslator
from pathlib import Path
import sys

model_faster_small = WhisperModel("small", device="cpu", compute_type="float32")
# model_faster_medium = WhisperModel("medium", device="cpu", compute_type="float32")
# model_faster_large = WhisperModel("large", device="cpu", compute_type="float32")

input_device = ""
output_device = ""
input_language = ""
output_language = ""
def init_devices():
    global input_device, output_device, input_language, output_language
    mic_dic = sr.Microphone.list_microphone_names()
    print("Selecciona un microfono: ")
    for i, mic in enumerate(mic_dic):
        print(f"[{i}]: {mic}")
    input_device = int(input("Input device: "))

    print("Selecciona un altavoz: ")
    init_by_me = not pygame.mixer.get_init()
    if init_by_me:
        pygame.mixer.init()
    devices = tuple(sdl2_audio.get_audio_device_names(False))
    for i, device in enumerate(devices):
        print(f"[{i}]: {device}")
    output_device = int(input("Altavoz: "))
    output_device = devices[output_device]
    pygame.mixer.quit()
    langs = {
        "es":"Español",
        "en":"English"
    }
    print("Selecciona un idioma de entrada: ")
    for lang in langs:
        print(f"[{lang}]: {langs[lang]}")
    input_language = input("Idioma de entrada: ")
    print("Selecciona un idioma de salida: ")
    for lang in langs:
        print(f"[{lang}]: {langs[lang]}")
    output_language = input("Idioma de salida: ")
    

def audio_to_bytes(audio):
    wav_file = BytesIO()
    wav_data = audio.get_wav_data()
    wav_file.write(wav_data)
    wav_file.seek(0)
    return wav_file.read()

def play_audio_bytes(audio_bytes):
    wav_file = BytesIO(audio_bytes)
    playFile(wav_file)       

def listen():
    global input_device
    print(f"listening with device: {input_device} ...")
    microphone = sr.Microphone(device_index=input_device)
    r = sr.Recognizer()
    with microphone as source:
        audio = r.listen(source)
    return audio

def piper_talk(text, lang="es"):
    start_time = time.time()
    langs = {
        "es": "./piper-models/es/es_ES-davefx-medium.onnx",
        "en": "./piper-models/en/en_US-joe-medium.onnx"
    }
    if lang not in langs:
        raise ValueError(f"Language {lang} not supported")
    if not os.path.exists(langs[lang]):
        voices = {
            "es": "es_ES-davefx-medium",
            "en": "en_US-joe-medium"
        }
        os.makedirs(f"./piper-models/{lang}", exist_ok=True)
        download_voice(voices[lang], Path(f"./piper-models/{lang}"))
    modelPath = langs[lang]
    voice = PiperVoice.load(modelPath)
    wav_file = BytesIO()
    voice.synthesize_wav(text, wave.Wave_write(wav_file))
    wav_file.seek(0)
    playFile(wav_file)
    end_time = time.time()
    print(f"Piper ({lang}) Time taken: {end_time - start_time} seconds")
    return wav_file

def gtts_talk(text, lang="es"):
    start_time = time.time()
    tts = gtts.gTTS(text=text, lang=lang)
    wav_file = BytesIO()
    tts.write_to_fp(wav_file)
    wav_file.seek(0)
    playFile(wav_file)
    end_time = time.time()
    print(f"gTTS ({lang}) Time taken: {end_time - start_time} seconds")
    return wav_file

def audio_to_text(audio_bytes, lang="none"):
    # faster whisper
    segments, info = model_faster_small.transcribe(BytesIO(audio_bytes), language="es",)
    text = "".join([segment.text for segment in segments])
    return text    
    

def playFile(filename):
    global output_device
    init_by_me = not pygame.mixer.get_init()
    if init_by_me:
        pygame.mixer.init(devicename=output_device)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def piper_tts(text, lang="es"):
    start_time = time.time()
    langs = {
        "es": "./piper-models/es/es_ES-davefx-medium.onnx",
        "en": "./piper-models/en/en_US-joe-medium.onnx"
    }
    modelPath = langs[lang]
    voice = PiperVoice.load(modelPath)
    with wave.open(f"hola_{lang}_piper.wav", "w") as wav_file:
        voice.synthesize_wav(text, wav_file)
    end_time = time.time()
    print(f"Piper ({lang}) Time taken: {end_time - start_time} seconds")
    return f"hola_{lang}_piper.wav"

def gtts_tts(text, lang="es"):
    start_time = time.time()
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(f"hola_{lang}_gtts.mp3")
    end_time = time.time()
    print(f"gTTS ({lang}) Time taken: {end_time - start_time} seconds")
    return f"hola_{lang}_gtts.mp3"

def deep_translate(text, lang="en"):
    start_time = time.time()
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    end_time = time.time()
    print(f"DeepL ({lang}) Time taken: {end_time - start_time} seconds")
    print(translated)
    return translated

def main():
    #args
    flow = 0
    if len(sys.argv) > 1:
        flow = int(sys.argv[1])

    if flow== 1:
        init_devices()
        print("\n\nResumen de configuracion:")
        print(f"Input device: {input_device}")
        print(f"Output device: {output_device}")
        print(f"Input language: {input_language}")
        print(f"Output language: {output_language}")
        print("\n\nPresiona enter para comenzar")
        input()
        audio = listen()
        audio_bytes = audio_to_bytes(audio)
        text = audio_to_text(audio_bytes, lang=input_language)
        print("text_listen: ", text)
        if(input_language != output_language):
            text_translated = deep_translate(text, lang=output_language)
            print("text_translated: ", text_translated)
        else:
            text_translated = text
        # piper_talk(f"{text_translated}", lang=output_language)
        gtts_talk(f"{text_translated}", lang=output_language)
    if flow == 0:
        wav_file_es = piper_talk(text="Hola, ¿cómo estás?. Quisiera saber si me puedes ayudar con algo.", lang="es")
        wav_file_es.seek(0)
        audio_bytes = wav_file_es.read()
        _ = piper_talk(text="Hello, how are you? I would like to know if you can help me with something.", lang="en")
        _ = gtts_talk(text="Hola, ¿cómo estás?. Quisiera saber si me puedes ayudar con algo.", lang="es")
        _ = gtts_talk(text="Hello, how are you? I would like to know if you can help me with something.", lang="en")
    

        # SpeechRecognition
        start_time = time.time()
        r = sr.Recognizer()
        audio = sr.AudioFile(BytesIO(audio_bytes))
        with audio as source:
            audio_data = r.record(source)
        result_speechrecognition = r.recognize_google(audio_data, language="es")
        end_time = time.time()
        print(result_speechrecognition)
        print(f"SpeechRecognition Time taken: {end_time - start_time} seconds")
        

        # Faster Whisper
        start_time = time.time()
        model_faster = WhisperModel("tiny", device="cpu", compute_type="float32")
        segments, info = model_faster.transcribe(BytesIO(audio_bytes), language="es",)
        end_time = time.time()
        text = "".join([segment.text for segment in segments])
        print(text)
        print(f"Faster Whisper tiny Time taken: {end_time - start_time} seconds")

        start_time = time.time()
        model_faster = WhisperModel("small", device="cpu", compute_type="float32")
        segments, info = model_faster.transcribe(BytesIO(audio_bytes), language="es",)
        end_time = time.time()
        text = "".join([segment.text for segment in segments])
        print(text)
        print(f"Faster Whisper small Time taken: {end_time - start_time} seconds")

        start_time = time.time()
        model_faster = WhisperModel("medium", device="cpu", compute_type="float32")
        segments, info = model_faster.transcribe(BytesIO(audio_bytes), language="es",)
        end_time = time.time()
        text = "".join([segment.text for segment in segments])
        print(text)
        print(f"Faster Whisper medium Time taken: {end_time - start_time} seconds")

        # Piper
        file_piper = piper_tts(text=text, lang="es")

        # gTTS
        file_gtts = gtts_tts(text=text, lang="es")

        playFile(file_piper)
        playFile(file_gtts)

        # DeepL
        translated = deep_translate(text=text)

        # Piper
        file_piper = piper_tts(translated, lang="en")

        # gTTS
        file_gtts = gtts_tts(translated, lang="en")

        playFile(file_piper)
        playFile(file_gtts)

if __name__ == "__main__":
    main()
