import json
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="News Classifier", page_icon="📰")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


def clean(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def scores(model, text: str) -> pd.Series:
    """Return the predicted probability for every category."""
    p = model.predict_proba([clean(text)])[0]
    return pd.Series(p, index=model.classes_).sort_values(ascending=False)


model = load_model()
metrics = json.load(open("metrics.json"))

st.title("📰 News Classification Model")
st.write("Paste a news headline or article below and the model will predict its category.")

examples = {
    "Sports": "India beat Australia by five wickets in a thrilling final over",
    "Business": "Sensex jumps 400 points as banking and IT stocks rally",
    "Technology": "Apple launches new iPhone with faster chip and better camera",
    "Entertainment": "Shah Rukh Khan's new film breaks box office records on opening day",
}
choice = st.selectbox("Try an example (optional)", ["—"] + list(examples))
text = st.text_area("Enter news text", value=examples.get(choice, ""), height=160)

if st.button("Classify", type="primary"):
    if len(clean(text)) < 3:
        st.warning("Please enter some news text.")
    else:
        s = scores(model, text)
        st.success(f"Predicted category: **{s.index[0]}**  ({s.iloc[0]:.0%} confidence)")
        st.bar_chart(s)

with st.expander("About this model"):
    st.write(f"**Algorithm:** TF-IDF (1–2 grams) + {metrics['best_model']}")
    st.write(f"**Test accuracy:** {metrics['results'][metrics['best_model']]['accuracy']:.1%}")
    st.write("**Categories:** " + ", ".join(metrics["labels"]))
