import streamlit as st
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import librosa
import cv2
from PIL import Image
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wavfile

# Spotify API credentials (replace with your credentials)
SPOTIPY_CLIENT_ID = '1a332a9873bc4d739adfe877e87e790f'
SPOTIPY_CLIENT_SECRET = 'f1ec74c0bb1f471097b4631bbf19df72'

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# Load the trained models
facial_emotion_model = load_model('CNN_emotion_model_VGG16.keras')
speech_emotion_model = load_model('Speech_emotion_model.keras')

# Load the music dataset
music_data = pd.read_csv("data_moods.csv")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to map detected emotions to music moods
def map_emotion_to_mood(emotion):
    mood_mapping = {
        'happy': 'Happy',
        'sad': 'Sad',
        'angry': 'Energetic',
        'neutral': 'Calm',
        'fearful': 'Calm',
        'disgusted': 'Happy',
        'surprised': 'Energetic'
    }
    return mood_mapping.get(emotion, 'Calm')

# Function to recommend music based on detected emotion
def recommend_music(emotion):
    mood = map_emotion_to_mood(emotion)
    filtered_tracks = music_data[music_data['mood'] == mood]
    if not filtered_tracks.empty:
        selected_track = filtered_tracks.sample(n=1).iloc[0]
        return selected_track['name'], selected_track['artist']
    else:
        return None, None

# Function to play music using Spotify API
def play_music_on_spotify(track_name, artist_name):
    query = f"track:{track_name} artist:{artist_name}"
    results = sp.search(q=query, limit=1)
    if results['tracks']['items']:
        track_uri = results['tracks']['items'][0]['uri']
        st.write(f"Playing: {track_name} by {artist_name}")
        st.write(f"[Play on Spotify](https://open.spotify.com/track/{track_uri.split(':')[-1]})")
    else:
        st.write("Track not found on Spotify.")

# Function to predict emotion from an image
def predict_emotion_from_image(image):
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = facial_emotion_model.predict(img_array)
    predicted_emotion_index = np.argmax(prediction)
    emotion_labels = ['happy', 'sad', 'angry', 'neutral', 'fearful', 'disgusted', 'surprised']
    return emotion_labels[predicted_emotion_index]

# Function to extract features from an audio file
def extract_features(audio_path, sr=22050):
    audio, sample_rate = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Function to predict emotion from an audio file
def predict_emotion_from_audio(audio_file):
    features = extract_features(audio_file)
    features = features.reshape(1, -1)
    prediction = speech_emotion_model.predict(features)
    predicted_emotion_index = np.argmax(prediction)
    emotion_labels = ['happy', 'sad', 'angry', 'neutral', 'fearful', 'disgusted', 'surprised']
    return emotion_labels[predicted_emotion_index]

# Function to detect emotion from webcam image
def detect_emotion_from_webcam():
    st.write("Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.write("Error opening webcam.")
        return None

    detected_emotion = None
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_color = frame[y:y+h, x:x+w]
                face_img = cv2.resize(roi_color, (224, 224))
                face_img = img_to_array(face_img)
                face_img = np.expand_dims(face_img, axis=0)
                face_img /= 255.0
                prediction = facial_emotion_model.predict(face_img)
                predicted_emotion_index = np.argmax(prediction)
                emotion_labels = ['happy', 'sad', 'angry', 'neutral', 'fearful', 'disgusted', 'surprised']
                detected_emotion = emotion_labels[predicted_emotion_index]
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')
                st.write(f"Detected Emotion: {detected_emotion}")
                cap.release()
                cv2.destroyAllWindows()
                return detected_emotion
        else:
            st.write("No face detected.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return detected_emotion

# Function to record audio and predict emotion
def record_and_predict_emotion():
    st.write("Recording...")
    fs = 22050  # Sample rate
    duration = 5  # seconds

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    wavfile.write("temp_audio.wav", fs, (recording * 32767).astype(np.int16))

    # Predict emotion from the recorded audio
    detected_emotion = predict_emotion_from_audio("temp_audio.wav")
    st.write(f"Detected Emotion from Speech: {detected_emotion}")
    return detected_emotion

# Streamlit UI
st.title("Emotion-aware Smart Music Recommendation System")

st.header("Choose an option:")

# Initialize detected_emotion
detected_emotion = None

# Option to use webcam
if st.button("Use Webcam for Facial Emotion Detection"):
    detected_emotion = detect_emotion_from_webcam()

# Option to use microphone for real-time speech emotion detection
elif st.button("Record Audio for Speech Emotion Detection"):
    detected_emotion = record_and_predict_emotion()

else:
    # File uploader for image or audio
    uploaded_file = st.file_uploader("Upload a photo or audio file...", type=["jpg", "jpeg", "png", "wav", "mp3"])

    if uploaded_file is not None:
        if uploaded_file.type.startswith("image/"):
            # Predict emotion from image
            detected_emotion = predict_emotion_from_image(uploaded_file)
            st.write(f"Detected Emotion: {detected_emotion}")

        elif uploaded_file.type.startswith("audio/"):
            # Predict emotion from audio
            detected_emotion = predict_emotion_from_audio(uploaded_file)
            st.write(f"Detected Emotion: {detected_emotion}")

# Recommend and play music based on detected emotion
if detected_emotion:
    track_name, artist_name = recommend_music(detected_emotion)
    if track_name and artist_name:
        st.write(f"Recommended Track: {track_name} by {artist_name}")
        play_music_on_spotify(track_name, artist_name)
    else:
        st.write("No suitable track found for the detected emotion.")
