

# Telangana Weather Forecast Project

## Overview

This project focuses on building a weather Prediciitng system for districts in Telangana, India.The goal is to predict weather conditions based on various features like temperature, humidity, precipitation, and windspeed.

## Project structure
datavisualisation.ipynb: Handles data preprocessing tasks, including handling missing values, encoding categorical variables, and creating new features.Explores and visualizes the data to gain insights. Includes various plots for class distribution, average temperature changes by season, precipitation by season, and more.

MLmodel.ipynb: Applies machine learning models to predict weather conditions. Includes XGBoost, Random Forest Classifier, Linear Regression, Logistic Regression, KNeighborsClassifier, SVM, and Decision Tree.

app.py: Selects the best-performing model (Random Forest Classifier) and develops a simple web app using Streamlit. The web app allows users to input a city and get a weather warnings.


## Model:
  The prediciitve model uses Random forest. The model has 76% accuracy. It takes temperature, precipiton, humidity and windspeed as input from OpenWeather API and predicts the weather conditions. 

## Getting Started
To use the Weather-prediction model:

1.Clone this GitHub repository to your local machine.

2. Install the required libraries and dependencies by running:

pip install -r requirements.txt

3. **Run Scripts:**
   - Execute the scripts in the following order:
1. datavisualization.ipynb
2. MLmodel.ipynb
3. app.py



## Author
Oladri Renuka (renukareddy.oladhri@gmail.com) Avula Jhansy (avulajhansy6@gmail.com) Bodapatla Sindhu Priya (bsindhupriya03@gmail.com) Suchitra Sekhar(suchitrasekhar2004@gmail.com) Sai sri laxmi(saisrilaximikommineni@gmail.com)

## Additional Notes

- Ensure you have the required dataset (`weather.csv`) in the specified location.
- API key: The web app uses the OpenWeatherMap API. Make sure to replace the placeholder API key in the script with your own key.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
---
