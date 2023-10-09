#!/usr/bin/env python
# coding: utf-8

# In[13]:


### random forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("weather.csv")
description_mapping = {
    'Partly cloudy throughout the day.': 0,
    'Partly cloudy throughout the day with rain.': 1,
    'Partly cloudy throughout the day with rain clearing later.': 2,
    'Cloudy skies throughout the day.': 3,
    'Cloudy skies throughout the day with late afternoon rain.': 4,
    'Partly cloudy throughout the day with a chance of rain throughout the day.': 5,
    'Partly cloudy throughout the day with rain in the morning and afternoon.': 6,
    'Partly cloudy throughout the day with morning rain.': 7,
    'Cloudy skies throughout the day with rain clearing later.': 8,
    'Partly cloudy throughout the day with late afternoon rain.': 9,
    'Cloudy skies throughout the day with early morning rain.': 10,
    'Cloudy skies throughout the day with a chance of rain throughout the day.': 11,
    'Cloudy skies throughout the day with rain.': 12,
    'Partly cloudy throughout the day with early morning rain.': 13,
    'Partly cloudy throughout the day with afternoon rain.': 14,
    'Becoming cloudy in the afternoon with early morning rain.': 15,
    'Becoming cloudy in the afternoon.': 16,
    'Becoming cloudy in the afternoon with late afternoon rain.': 17,
    'Clear conditions throughout the day.': 18,
    'Clearing in the afternoon with a chance of rain throughout the day.': 19,
    'Clearing in the afternoon.': 20,
    'Clearing in the afternoon with morning rain.': 21,
    'Cloudy skies throughout the day with rain in the morning and afternoon.': 22,
    'Clear conditions throughout the day with rain clearing later.': 23,
    'Clear conditions throughout the day with morning rain.': 24,
    'Becoming cloudy in the afternoon with rain in the morning and afternoon.': 25,
    'Becoming cloudy in the afternoon with rain.': 26,
    'Cloudy skies throughout the day with morning rain.': 27,
    'Cloudy skies throughout the day with afternoon rain.': 28,
    'Clearing in the afternoon with rain in the morning and afternoon.': 29,
    'Clearing in the afternoon with rain.': 30,
    'Clearing in the afternoon with rain clearing later.': 31,
    'Becoming cloudy in the afternoon with rain clearing later.': 32,
    'Clear conditions throughout the day with rain.': 33,
    'Becoming cloudy in the afternoon with afternoon rain.': 34,
    'Clear conditions throughout the day with late afternoon rain.': 35,
    'Clear conditions throughout the day with afternoon rain.': 36,
    'Clear conditions throughout the day with early morning rain.': 37,
    'Clearing in the afternoon with early morning rain.': 38,
    'Becoming cloudy in the afternoon with morning rain.': 39,
    'Clearing in the afternoon with afternoon rain.': 40,
    'Clearing in the afternoon with late afternoon rain.': 41,
    'Becoming cloudy in the afternoon with a chance of rain throughout the day.': 42,
    'Clear conditions throughout the day with a chance of rain throughout the day.': 43,
    'Clear conditions throughout the day with rain in the morning and afternoon.': 44,
    'nan': -1  # Handle 'nan' value
}
label_encoder = LabelEncoder()
df['description'] = df['description'].map(description_mapping).fillna(-1)  # Use map and handle 'nan' with -1
df['datetime'] = pd.to_datetime(df['datetime'])
# Feature Engineering (if needed)
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_year'] = df['datetime'].dt.dayofyear
df['hour'] = df['datetime'].dt.hour
# Specify your feature columns
feature_columns = ['day_of_week', 'day_of_year', 'hour', 'temp', 'humidity', 'precip', 'windspeed']
# Select features and target variable
X = df[feature_columns]
y = df['description']  # Updated target variable
# Split the data into training and testing sets (80% train, 20% test in this example)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a Random Forest classification model
random_forest = RandomForestClassifier()
# Train the model on the training data
random_forest.fit(X_train, y_train)
# Make predictions on the test data
y_pred = random_forest.predict(X_test)
# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
# Print the metrics
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))


# In[14]:


import pickle

with open('Weather_model.pkl', 'wb') as model_file:
    pickle.dump(random_forest, model_file,protocol=4)


# In[15]:


import streamlit as st
import requests
import pandas as pd
import pickle
import pyttsx3

# Load the model
with open('Weather_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# API key for OpenWeatherMap
api_key = '0306b8c765c8842c17662737cedb83ab'  # Replace with your API key

# List of allowed localities
allowed_localities = ['nalgonda', 'Warangal', 'Adilabad', 'Medchal', 'Mancherial', 'Mahabubabad', 'khammam', 'hyderabad', 'nizamabad', 'medak', 'Nirmal', 'siddipet', 'vikarabad', 'KARIMNAGAR']

# Streamlit app configuration
st.set_page_config(
    page_title="Weather Forecast App",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Text-to-speech engine setup
engine = pyttsx3.init()

# Sidebar for user input
st.sidebar.title("Weather Forecast")
location = st.sidebar.text_input("Enter the city (e.g., 'Hyderabad'): ")

# Button to trigger weather forecast
if st.sidebar.button("Get Weather Forecast"):
    # Check if the entered location is in the allowed list
    if location.lower() not in allowed_localities:
        st.sidebar.error("Invalid location. Please enter one of the allowed districts.")
    else:
        # Make API request
        api_url = f'http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}'
        response = requests.get(api_url)

        # Display results if API request is successful
        if response.status_code == 200:
            forecast_data = response.json()

            # Process weather data
            weather_data = []
            for entry in forecast_data['list']:
                date_time = entry['dt_txt']
                temperature_kelvin = entry['main']['temp']
                humidity = entry['main']['humidity']
                precip = 0
                windspeed = entry['wind']['speed']

                temperature_celsius = temperature_kelvin - 273.15
                day_of_week = pd.to_datetime(date_time).dayofweek
                day_of_year = pd.to_datetime(date_time).dayofyear
                hour = pd.to_datetime(date_time).hour

                weather_data_point = {
                    'day_of_week': day_of_week,
                    'day_of_year': day_of_year,
                    'hour': hour,
                    'temp': temperature_celsius,
                    'humidity': humidity,
                    'precip': precip,
                    'windspeed': windspeed,
                    'date_time': date_time
                }
                weather_data.append(weather_data_point)

            weather_df = pd.DataFrame(weather_data)

            # Filter data for the first two days
            weather_df_2_days = weather_df[weather_df['day_of_year'] <= (weather_df['day_of_year'].min() + 1)]

            # Make predictions
            predictions = model.predict(weather_df_2_days[['day_of_week', 'day_of_year', 'hour', 'temp', 'humidity', 'precip', 'windspeed']])

            # Reverse the mapping to create a dictionary of {value: key}
            inverse_description_mapping = {v: k for k, v in description_mapping.items()}

            # Initialize previous prediction
            prev_prediction = None

            # Display predictions with corresponding date_time
            for i, (prediction, timestamp) in enumerate(zip(predictions, weather_df_2_days['date_time'])):
                predicted_description = inverse_description_mapping.get(prediction, 'Unknown')
                st.write(f' {pd.to_datetime(timestamp)}, Description: {predicted_description}', font=('Algerian', 14))

                # Check for sudden weather changes and display a text-to-speech alert
                if prev_prediction is not None and prediction != prev_prediction:
                    st.warning(f"Sudden weather change detected at {pd.to_datetime(timestamp)}!")

                    # Convert the warning message to speech
                    engine.say(f"Sudden weather change detected at {pd.to_datetime(timestamp)}!")
                    engine.runAndWait()

                # Update previous prediction
                prev_prediction = prediction

        else:
            st.error('Error fetching weather forecast data. Please check your input and API key.')


# In[ ]:




