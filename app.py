import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import MeanSquaredError

# Load the model, ignoring the `reduction` argument
custom_objects = {'MeanSquaredError': MeanSquaredError}

# Load the models
@st.cache(allow_output_mutation=True)
def load_models():
    
    model = load_model("vulnerability_model.h5",  custom_objects=custom_objects)
    word2vec_model = Word2Vec.load("word2vec_model.bin") 
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, word2vec_model, label_encoder

model, word2vec_model, label_encoder = load_models()

def get_word2vec_vector(tokens, word2vec_model, vector_size):
    vec = np.zeros(vector_size)
    count = 0
    for token in tokens:
        if token in word2vec_model.wv.key_to_index:
            vec += word2vec_model.wv[token]
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
