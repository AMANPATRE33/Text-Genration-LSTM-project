# LSTM Text Generation – Full GenAI Project

#CHECK DEPLOYED STEAMLIT VERSION :


This project demonstrates an end-to-end Generative AI pipeline using a character-level LSTM model trained on Shakespeare's works.

## Contents
- LSTM model training & evaluation
- Temperature-based text generation
- Streamlit web application
- Model performance statistics
- Dataset included

## Dataset
Shakespeare corpus from TensorFlow public mirror (Project Gutenberg).

## Model Architecture
- Embedding Layer (128)
- LSTM Layer (256 units)
- Dense Softmax Output

## Performance
- Training Loss: 1.18
- Validation Loss: 1.46
- Token Accuracy: 60.72%
- Top-5 Accuracy: 88.66%
- Top-10 Accuracy: 95.80%

## Streamlit App
Run locally using:
streamlit run app.py

## Requirements
See requirements.txt
