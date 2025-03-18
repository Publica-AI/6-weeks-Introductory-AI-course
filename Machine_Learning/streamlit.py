import pickle
import streamlit as st
import pandas as pd
import numpy as np


training_data = pd.read_csv("Training.csv")
# Create a list of features used during training (exclude the 'prognosis' column)
trained_features = list(training_data.columns)
trained_features.remove("prognosis")

my_model = pickle.load(open("naive_bayes_model.pkl", "rb"))

prognosis_mapping = {
    0: "(vertigo) Paroymsal Positional Vertigo",
    1: "AIDS",
    2: "Acne",
    3: "Alcoholic hepatitis",
    4: "Allergy",
    5: "Arthritis",
    6: "Bronchial Asthma",
    7: "Cervical spondylosis",
    8: "Chicken pox",
    9: "Chronic cholestasis",
    10: "Common Cold",
    11: "Dengue",
    12: "Diabetes",
    13: "Dimorphic hemmorhoids(piles)",
    14: "Drug Reaction",
    16: "GERD",
    17: "Gastroenteritis",
    18: "Heart attack",
    19: "Hepatitis B",
    20: "Hepatitis C",
    21: "Hepatitis D",
    22: "Hepatitis E",
    23: "Hypertension",
    24: "Hyperthyroidism",
    25: "Hypoglycemia",
    26: "Hypothyroidism",
    27: "Impetigo",
    28: "Jaundice",
    29: "Malaria",
    30: "Migraine",
    31: "Osteoarthristis",
    32: "Paralysis (brain hemorrhage)",
    33: "Peptic ulcer diseae",
    34: "Pneumonia",
    35: "Psoriasis",
    36: "Tuberculosis",
    37: "Typhoid",
    38: "Urinary tract infection",
    39: "Varicose veins",
    40: "hepatitis A"
}


def predict_disease(selected_symptoms):
    # Create binary input vector for all trained features
    input_vector = [1 if feature in selected_symptoms else 0 for feature in trained_features]
    # The model expects a 2D array as input
    prediction_code = my_model.predict([input_vector])[0]
    disease = prognosis_mapping.get(prediction_code, "Unknown")
    return disease


st.title("Disease Prediction Based on Symptoms")
st.write("Select the symptoms you are experiencing:")

selected_symptoms = st.multiselect("Select Symptoms", trained_features)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        predicted_disease = predict_disease(selected_symptoms)
        st.success("Predicted Disease: " + predicted_disease)