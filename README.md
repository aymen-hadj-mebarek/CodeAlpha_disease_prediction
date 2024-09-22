# Disease Prediction from Medical Data

This project is a part of my internship with **CodeAlpha**, focusing on building an AI model to predict disease prognosis based on symptom severity. The model is integrated into a Flask-based web application that allows users to select symptoms, and it predicts the most probable disease outcomes.

## Overview

- **AI Model**: A Neural Network model built using TensorFlow and Keras to predict disease prognosis based on symptoms and their severity.
- **Web Application**: A Flask web app where users can input symptoms, and the model predicts possible diagnoses, providing more information about prognosis and potential treatment when needed.
- **Dataset**: The dataset used contains 4,900 instances. While the model works well on this data, it is expected to perform even better with larger datasets.

## Project Structure

### Training

- **`label_encoder.pkl`**: The label encoder used for encoding disease labels.
- **`model.pkl`**: The trained model file for predicting disease prognosis.
- **`symptom_Description.csv`**: Contains detailed descriptions of symptoms.
- **`symptom_precaution.csv`**: Contains precautionary measures based on the predicted diseases.
- **`Symptom-severity.csv`**: Contains the severity of each symptom, which is used to weight the symptoms during the training and inference process.
- **`Training.csv`**: The dataset used for training the model.
- **`Testing.csv`**: The dataset used for testing the model.
- **`train.ipynb`**: A Jupyter notebook for training the AI model and visualizing results.

### Web App

- **`app.py`**: The Flask web app that provides an interface for users to input symptoms and get prognosis predictions.
- **Model Folder (`model/`)**:
    - **`best_model.pkl`**: The trained AI model file used by the web app for predictions.
    - **`best_label_encoder.pkl`**: The label encoder for the model, used to decode predicted labels.
- **Static Files (`static/`)**:
    - **`style.css`**: Contains the CSS styling for the web app.
- **Templates (`templates/`)**:
    - **`index.html`**: The main page for user input.
    - **`result.html`**: Displays the results of the prognosis predictions.
- **`Symptom-severity.csv`**: A copy of the CSV file used in the web app to associate symptoms with their severity.

## AI Model

The AI model uses a simple neural network architecture:

- Input layer: Processes 132 features (symptoms).
- Two hidden layers: With 64 and 32 units, respectively, using ReLU activation.
- Output layer: Predicts one of the possible diagnoses using softmax activation.

### Model Training

The model was trained using the following steps:

1. **Data Preprocessing**: Each symptom's severity is multiplied by its weight, and the labels (diseases) are encoded.
2. **Model Training**: The data was split into 80% training and 20% testing. The model was trained for 50 epochs.
3. **Evaluation**: The model achieved a good accuracy, but further improvements could be made with a larger dataset.

### Accuracy & Performance

- **Test Accuracy**: Approximately 80% on the available test data.
- **Visualization**: The project includes visualizations of symptom frequency, correlations, and model performance (accuracy, loss plots, confusion matrix).

## Flask Web Application

The web app allows users to:

1. Select symptoms from a list (based on `Symptom-severity.csv`).
2. Submit the form to get a list of the top 3 possible diagnoses with their probabilities.
3. Display additional information about the prognosis and treatment options for the predicted diseases.

### App Routes

- **Home (`/`)**: The main page where users can select symptoms and submit them for diagnosis.
- **Result (`/result`)**: Displays the top 3 predicted diseases along with their probabilities.

### Key Dependencies

- Flask
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- Seaborn
- Matplotlib

## Installation & Usage

### Prerequisites

- Python 3.x
- TensorFlow
- Flask

### Installation Steps

1. Clone the repository:
    
    ```bash
    git clone https://github.com/aymen-hadj-mebarek/CodeAlpha_disease_prediction.git
    cd CodeAlpha_disease_prediction
    ```
    
2. Install required dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. Run the Flask app:
    
    ```bash
    python app.py
    ```
    
4. Access the app in your browser at `http://127.0.0.1:5000/`.
    

## Future Improvements

- The model will benefit from a **larger dataset** to improve performance.
- Extend the web app to display **more detailed information** about the prognosis and treatment based on the predicted diagnosis.

## Credits

This project is developed as part of my internship at **CodeAlpha**.