from flask import Flask, request, jsonify, send_file
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import requests
from datetime import datetime
import re
from keras.layers import Layer

# Flask app initialization
app = Flask(__name__)

# Paths to model and tokenizer files
MODEL_DIR = os.getcwd()
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_model.keras")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# Load models
encoder_model = tf.keras.models.load_model(ENCODER_PATH)
decoder_model = tf.keras.models.load_model(DECODER_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH, 'rb') as file:
    tokenizer = pickle.load(file)

# Load latitude and longitude dataset
CSV_URL = "https://raw.githubusercontent.com/ajisakarsyi/CulTour/refs/heads/main/asset/province_latitude_longitude_data.csv"

try:
    df = pd.read_csv(CSV_URL, skiprows=1, names=["country", "province", "latitude", "longitude"])
    print("Latitude and longitude data loaded successfully.")
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Function to fetch weather data
def fetch_weather_data(latitude, longitude, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation"
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to process weather data
def process_weather_data(data):
    times = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in data["hourly"]["time"]]
    precipitation = np.array(data["hourly"]["precipitation"])

    daily_precipitation = {}
    for time, rain in zip(times, precipitation):
        date = time.date()
        if date not in daily_precipitation:
            daily_precipitation[date] = 0
        daily_precipitation[date] += rain

    rainy_days = {date: rain > 0.1 for date, rain in daily_precipitation.items()}

    total_days = len(rainy_days)
    rainy_count = sum(rainy_days.values())
    if rainy_count / total_days > 0.5:
        weather_summary = "Mostly rainy"
    elif rainy_count / total_days > 0.25:
        weather_summary = "Partly rainy"
    else:
        weather_summary = "Mostly sunny"

    return rainy_days, weather_summary

# Function to generate weather plot
def plot_weather_with_annotations(data, rainy_days):
    times = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in data["hourly"]["time"]]
    temperatures = data["hourly"]["temperature_2m"]
    precipitation = data["hourly"]["precipitation"]

    plt.figure(figsize=(12, 6))
    plt.plot(times, temperatures, label="Temperature (°C)", color="orange", linewidth=2)
    plt.fill_between(times, precipitation, label="Precipitation (mm)", color="blue", alpha=0.3)

    for date, is_rainy in rainy_days.items():
        if is_rainy:
            plt.axvspan(datetime.combine(date, datetime.min.time()),
                        datetime.combine(date, datetime.max.time()),
                        color="gray", alpha=0.2, label="Rainy Day" if "Rainy Day" not in plt.gca().get_legend_handles_labels()[1] else None)

    plt.title("Weather Forecast (Easy to Interpret)")
    plt.xlabel("Time")
    plt.ylabel("Weather Metrics")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "static", "weather_plot.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    return plot_path


# Function to load data from a URL
def load_data_from_url(file_url):
    response = requests.get(file_url)
    response.raise_for_status()

    lines = response.text.splitlines()
    input_data = []
    output_data = []
    mode = None 

    for line in lines:
        line = line.strip()
        if line == "[Input]":
            mode = "input"
            continue
        elif line == "[Output]":
            mode = "output"
            continue
        elif not line:
            continue

        if mode == "input":
            input_data.append(line)
        elif mode == "output":
            output_data.append(line)

    return {"Input": input_data, "Output": output_data}

file_url = "https://raw.githubusercontent.com/ajisakarsyi/CulTour/refs/heads/main/asset/weather_response_data.txt"  

data = load_data_from_url(file_url)

input_sequences = tokenizer.texts_to_sequences(data["Input"])
output_sequences = tokenizer.texts_to_sequences(data["Output"])

# Padding
max_input_len = max(len(seq) for seq in input_sequences)
max_output_len = max(len(seq) for seq in output_sequences)

start_token = "[START]"
end_token = "[END]"

# Function to generate chatbot response
def generate_response(input_text):
    input_seq = pad_sequences(tokenizer.texts_to_sequences([input_text]), maxlen=max_input_len, padding='post')
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index[start_token]

    response = []
    for _ in range(max_output_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.random.choice(range(len(output_tokens[0, -1])), p=output_tokens[0, -1])
        sampled_word = tokenizer.index_word.get(sampled_token_index, "<OOV>")

        if sampled_word.strip() == end_token.strip():
            break

        if sampled_word not in {"<OOV>", end_token, start_token}:
            response.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    response = [
        word for word in response if word.strip() not in {start_token.strip(), end_token.strip()}
    ]

    response = ' '.join(response).replace("[end]", "").replace("[start]", "").strip()

    sentences = re.split(r'[.!]', response)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    sentences = sentences[:3]

    formatted_sentences = []
    for sentence in sentences:
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        if not sentence.endswith('.') and not sentence.endswith('!'):
            sentence += '.'

        formatted_sentences.append(sentence)

    final_response = ' '.join(formatted_sentences)

    return final_response



@app.route('/')
def home():
    return "Welcome to the Weather Chatbot API!"


# Flask route for weather chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    input = request.json
    province = input.get("province").strip().lower()
    start_date = input.get("start_date")
    end_date = input.get("end_date")

    if not all([province, start_date, end_date]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Get coordinates for the province
    
    province_row = df[df['province'].apply(lambda x: isinstance(x, str) and x.lower() == province)]
    if province_row.empty:
        return jsonify({"error": "Province not found"}), 404

    latitude = province_row.iloc[0]['latitude']
    longitude = province_row.iloc[0]['longitude']

    # Fetch weather data
    weather_data = fetch_weather_data(latitude, longitude, start_date, end_date)
    if "error" in weather_data:
        return jsonify({"error": "Failed to fetch weather data"}), 500

    # Process weather data
    rainy_days, weather_summary = process_weather_data(weather_data)

    # Generate weather plot
    plot_path = plot_weather_with_annotations(weather_data, rainy_days)

    # Generate chatbot response
    chatbot_response = generate_response(weather_summary)

    return jsonify({
        "weather_summary": weather_summary,
        "chatbot_response": chatbot_response,
        "plot_url": f"{request.url_root.rstrip('/')}/static/weather_plot.png"
    })

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
