
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="LSTM Text Generator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 LSTM Text Generator")
st.markdown(
    "Generate Shakespeare-style text using a trained **LSTM language model**."
)

# -----------------------------
# Load Model & Tokenizers
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lstm_text_generator.h5")

model = load_model()

with open("char_mappings.pkl", "rb") as f:
    char_to_idx, idx_to_char = pickle.load(f)

SEQ_LEN = 40

# -----------------------------
# Sampling Function
# -----------------------------
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# -----------------------------
# Text Generation
# -----------------------------
def generate_text(seed, length, temperature):
    seed = seed.lower()
    generated = seed

    for _ in range(length):
        seq = [char_to_idx.get(c, 0) for c in seed]
        seq = np.array(seq).reshape(1, -1)

        preds = model.predict(seq, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_char = idx_to_char[next_idx]

        generated += next_char
        seed = seed[1:] + next_char

    return generated

# -----------------------------
# UI Controls
# -----------------------------
seed_text = st.text_input(
    "✍️ Enter seed text",
    value="to be or not to be that "
)

col1, col2 = st.columns(2)

with col1:
    text_length = st.slider(
        "📏 Text Length",
        min_value=100,
        max_value=600,
        value=300,
        step=50
    )

with col2:
    temperature = st.slider(
        "🔥 Temperature",
        min_value=0.2,
        max_value=1.5,
        value=0.8,
        step=0.1
    )

# -----------------------------
# Generate Button
# -----------------------------
if st.button("🚀 Generate Text"):
    if len(seed_text) < SEQ_LEN:
        st.warning(f"Seed text must be at least {SEQ_LEN} characters long.")
    else:
        with st.spinner("Generating text..."):
            output = generate_text(
                seed_text[-SEQ_LEN:],
                text_length,
                temperature
            )

        st.subheader("📜 Generated Text")
        st.text_area("", output, height=300)
