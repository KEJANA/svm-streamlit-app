import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC

st.title("🌸 Iris Flower Prediction using SVM")

# Load Dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train Model
model = SVC()
model.fit(X, y)

st.subheader("Enter Flower Features:")

sepal_length = st.number_input("Sepal Length", min_value=4.0, max_value=8.0, value=5.1)
sepal_width = st.number_input("Sepal Width", min_value=2.0, max_value=4.5, value=3.5)
petal_length = st.number_input("Petal Length", min_value=1.0, max_value=7.0, value=1.4)
petal_width = st.number_input("Petal Width", min_value=0.1, max_value=2.5, value=0.2)

if st.button("Predict Flower"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    flower_name = data.target_names[prediction]
    st.success(f"Predicted Flower: {flower_name}")