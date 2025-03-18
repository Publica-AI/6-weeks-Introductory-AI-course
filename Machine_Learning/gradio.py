import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


URL = "student_score_prediction.csv"
student_data = pd.read_csv(URL)

# Prepare data
X = student_data[['Hours']]
y = student_data['Scores']

# Train model
model = LinearRegression()
model.fit(X, y)

# Define prediction function
def predict_score(hours):
    pred_score = model.predict(np.array(hours).reshape(-1,1))
    return np.round(pred_score[0], 2)

# Create Gradio Interface
gr.Interface(fn=predict_score,
             inputs=gr.Number(label="Hours Studied"),
             outputs=gr.Textbox(label="Predicted Score")).launch(share=True)

# st.title("Student Score Predictor")

# # User input
# hours = st.number_input("Enter the number of hours studied:", min_value=0.0, step=0.5)

# # Button to make predictions
# if st.button("Predict Score"):
#     result = predict_score(hours)
#     st.write(f"Predicted Score: {result}")