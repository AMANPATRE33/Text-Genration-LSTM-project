import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="LSTM Text Generator",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🧠 LSTM Text Generator</h1>
    <p style="text-align:center; font-size:16px;">
    Generate Shakespeare-style text using a trained <b>LSTM language model</b>
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Load Model & Character Mappings
# -------------------------------------------------
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("lstm_text_generator.h5")
    with open("char_mappings.pkl", "rb") as f:
        char_to_idx, idx_to_char = pickle.load(f)
    return model, char_to_idx, idx_to_char

model, char_to_idx, idx_to_char = load_resources()
SEQ_LEN = 40

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def prepare_seed(seed):
    seed = seed.lower()
    if len(seed) < SEQ_LEN:
        seed = (" " * (SEQ_LEN - len(seed))) + seed
    return seed[-SEQ_LEN:]

def generate_text(seed, length, temperature):
    seed = prepare_seed(seed)
    generated = seed.strip()

    for _ in range(length):
        seq = [char_to_idx.get(c, 0) for c in seed]
        seq = np.array(seq).reshape(1, -1)

        preds = model.predict(seq, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_char = idx_to_char[next_idx]

        generated += next_char
        seed = seed[1:] + next_char

    return generated

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Generation Settings")

    text_length = st.slider(
        "📏 Generated Text Length",
        min_value=100,
        max_value=600,
        value=300,
        step=50
    )

    temperature = st.slider(
        "🔥 Creativity (Temperature)",
        min_value=0.2,
        max_value=1.5,
        value=0.8,
        step=0.1
    )

    st.markdown(
        """
        **Temperature Guide**
        - 🔹 0.2 – 0.5 → Safer, repetitive
        - 🔸 0.6 – 1.0 → Balanced
        - 🔥 1.1 – 1.5 → Creative, risky
        """
    )

# -------------------------------------------------
# Main Input
# -------------------------------------------------
st.subheader("✍️ Enter Seed Text")
seed_text = st.text_area(
    "",
    value="to be or not to be",
    height=80,
    help="You can enter even a short phrase (5 words is enough)."
)

# -------------------------------------------------
# Generate Button
# -------------------------------------------------
if st.button("🚀 Generate Text", use_container_width=True):
    if seed_text.strip() == "":
        st.warning("Please enter some seed text to start generation.")
    else:
        with st.spinner("🧠 Generating Shakespearean text..."):
            output = generate_text(
                seed_text,
                text_length,
                temperature
            )

        st.subheader("📜 Generated Text")
        st.text_area(
            "",
            output,
            height=350
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:13px; color:gray;">
    Built with using LSTM, TensorFlow & Streamlit<br>
    Generative AI Project
    </p>
    """,
    unsafe_allow_html=True
)
