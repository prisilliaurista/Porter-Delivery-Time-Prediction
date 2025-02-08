# Porter-Delivery-Time-Prediction
Predicting delivery time using machine learning

## Project Overview
This project aims to estimate the delivery time for Porter services using machine learning techniques. By predicting the delivery time accurately, the company can optimize delivery processes and improve customer satisfaction.

## Problem Statement
Predicting the time it takes for deliveries based on historical data and various influencing factors (e.g., order details, delivery distance, traffic).

## Methodology
  1. Data Cleaning: Cleaned missing values and outliers.
  2. Feature Engineering: Created relevant features based on the data.
  3. Model Training: Used regression algorithms like Random Forest and Linear Regression.
  4. Evaluation: Evaluated models based on RMSE and accuracy.

## Technologies Used
Python; pandas; scikit-learn; Jupyter Notebooks

## Instalation
1. Clone this repository
```bash
git clone <repository-url>
```   
2. Install dependencies
```bash
pip install -r requirements.txt
```

## How To Run
Run the model training script:
```bash
python train_model.py
```
  
## Model Files (pkl files)
Model files are the saved versions of machine learning models that you've trained. These files store the trained model (i.e., the algorithms, weights, coefficients) so that you don’t have to retrain them every time.

.pkl files (short for Pickle) are a common file format used to save and load machine learning models in Python. When you train a machine learning model, you can "pickle" the model, which means you're serializing it (converting it into a format that can be saved to disk) so you can load it later without having to retrain it.

Why use .pkl files:
1. Save Time: Loading a saved model is much faster than retraining it from scratch.
2. Reproduce Results: You can share the model file with others, so they can use it without retraining.
3. Deployment: You can deploy the model (e.g., in a web app) without needing the original training script.

## How to save and load a model in Python using Pickle:
Saving a model:
```python
import pickle
from sklearn.ensemble import RandomForestRegressor

# Assume 'model' is your trained machine learning model
model = RandomForestRegressor()
model.fit(X_train, y_train)  # Example training

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

Loading a model:
```python
import pickle

# Load the model from a file
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use the loaded model to make predictions
predictions = loaded_model.predict(X_test)

## Dataset / Model File
[Download my_model.pkl from Google Drive](https://drive.google.com/file/d/12FtCJ51aQ2xjQMEHOVh4jEoZ3G1G3ebc/view?usp=drive_link)

```
