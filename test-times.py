import whisper
from faster_whisper import WhisperModel
import speech_recognition as sr
import wave
from piper.voice import PiperVoice
import gtts
import time
import pygame
from deep_translator import GoogleTranslator

def playFile(filename):
    pygame.mixer.init()
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
    # Whisper
    start_time = time.time()
    model = whisper.load_model("tiny", device="cpu")
    result_whisper = model.transcribe("hola.mp3", language="es", verbose=False, fp16=False)
    end_time = time.time()
    print(result_whisper["text"])
    print(f"Whisper Time taken: {end_time - start_time} seconds")

    # Faster Whisper
    start_time = time.time()
    model_faster = WhisperModel("tiny", device="cpu", compute_type="float32")
    segments, info = model_faster.transcribe("hola.mp3", language="es",)
    end_time = time.time()
    text = "".join([segment.text for segment in segments])
    print(text)
    print(f"Faster Whisper Time taken: {end_time - start_time} seconds")

    # SpeechRecognition
    start_time = time.time()
    r = sr.Recognizer()
    with sr.AudioFile("hola.wav") as source:
        audio = r.record(source)
    result_speechrecognition = r.recognize_google(audio, language="es")
    end_time = time.time()
    print(result_speechrecognition)
    print(f"SpeechRecognition Time taken: {end_time - start_time} seconds")

    # Piper
    file_piper = piper_tts(result_whisper["text"])

    # gTTS
    file_gtts = gtts_tts(result_whisper["text"])

    playFile(file_piper)
    playFile(file_gtts)

    # DeepL
    translated = deep_translate(result_whisper["text"])

    # Piper
    file_piper = piper_tts(translated, lang="en")

    # gTTS
    file_gtts = gtts_tts(translated, lang="en")

    playFile(file_piper)
    playFile(file_gtts)
if __name__ == "__main__":
    main()
