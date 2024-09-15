from transformers import pipeline
from gtts import gTTS
import os
import speech_recognition as sr

# Load pre-trained transformer model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0)


def get_gpt_response(prompt):
    response = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)
    return response[0]['generated_text']


def speak(text):
    tts = gTTS(text=text, lang='bn')
    tts.save("response.mp3")

    # Play the audio response
    if os.name == 'posix':  # For Unix-like systems (macOS, Linux)
        if 'darwin' in os.uname().sysname.lower():  # macOS
            os.system("afplay response.mp3")
        else:  # Linux
            os.system("mpg321 response.mp3")
    elif os.name == 'nt':  # For Windows
        os.system("start response.mp3")


def transcribe_and_respond():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        while True:
            try:
                audio = recognizer.listen(source)
                user_input = recognizer.recognize_google(audio)
                print("User:", user_input)
                response_text = get_gpt_response(user_input)
                print("JARVIS:", response_text)
                speak(response_text)
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
            except sr.RequestError:
                print("Sorry, there was an issue with the speech recognition service.")
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    transcribe_and_respond()
