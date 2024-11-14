import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the models
@st.cache(allow_output_mutation=True)
def load_models():
    model = load_model("vulnerability_model.h5")
    word2vec_model = KeyedVectors.load("word2vec_model_only_vectors.kv") 
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, word2vec, label_encoder

model, word2vec_model, label_encoder = load_models()

# Function to get Word2Vec vector for input text
def get_word2vec_vector(tokens, word2vec_model, vector_size):
    vec = np.zeros(vector_size)
    count = 0
    for token in tokens:
        if token in word2vec_model.key_to_index:  # Check against word2vec key_to_index instead of wv
            vec += word2vec_model[token]
            count += 1
    if count > 0:
        vec /= count
    return vec

# Streamlit App
st.title("Vulnerability Assessment Tool")
st.write("Predict CVSS score and CWE name based on vulnerability description.")

# User input
description = st.text_area("Enter vulnerability description:")

if st.button("Predict"):
    if description:
        # Preprocess input description
        tokens = description.lower().split()
        description_vector = get_word2vec_vector(tokens, word2vec_model, word2vec_model.vector_size)
        description_vector = description_vector.reshape(1, -1)

        # Make predictions
        predicted_cvss, predicted_cwe = model.predict(description_vector)
        predicted_cwe_name = label_encoder.inverse_transform([np.argmax(predicted_cwe)])

        # Display results
        st.write(f"Predicted CVSS Score: {predicted_cvss[0][0]:.2f}")
        st.write(f"Predicted CWE Name: {predicted_cwe_name[0]}")
    else:
        st.warning("Please enter a description.")
