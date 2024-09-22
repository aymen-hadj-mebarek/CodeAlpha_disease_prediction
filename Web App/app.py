from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
app = Flask(__name__)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Load the pre-trained model
with open('model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('model/best_label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Load the symptoms CSV
symptoms_df = pd.read_csv('Symptom-severity.csv')
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the severity values from the form (input names should match)
        symptoms_list = []
        for symptom in symptoms_df['Symptom']:
            value = request.form.get(symptom, 0)  # Default 0 for unselected
            symptoms_list.append(int(value))
        
        print("================================================================")
        print(symptoms_list)
        
        # Convert to numpy array and reshape for the model
        # symptoms_array = np.array(symptoms_list).reshape(1, -1)

        # Predict the diagnosis using the pre-trained model
        symptoms_list = np.array(symptoms_list)
        # symptoms_list = symptoms_list.values
        symptoms_list = symptoms_list.reshape((1,132))
        diagnosis = model.predict(symptoms_list)
        
        top_3_values, top_3_indices = tf.nn.top_k(diagnosis, k=3)
        
        # Convert top 3 indices to numpy arrays
        top_3_indices = top_3_indices.numpy()[0]
        top_3_values = top_3_values.numpy()[0]
        
        # Use LabelEncoder to get the string labels for the top 3 predicted classes
        top_3_labels = encoder.inverse_transform(top_3_indices)

        # Zip the top 3 labels with their corresponding probabilities
        top_3_predictions = list(zip(top_3_labels, top_3_values))
        print(top_3_predictions)
        
        # Render the result page with the diagnosis
        return render_template('result.html', diagnosis=top_3_predictions)

    # Render the symptom selection form
    
    symptoms = dict(zip(symptoms_df['Symptom'], symptoms_df['weight']))
    return render_template('index.html', symptoms=symptoms)

if __name__ == '__main__':
    app.run(debug=True)
