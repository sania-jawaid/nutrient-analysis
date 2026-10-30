"""Serve the saved HTML demo inside Streamlit, without running inference."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "main_output" / "index.html"

st.set_page_config(page_title="Dashboard 2 · HTML viewer", layout="wide")
st.markdown(
    "<style>.stApp,[data-testid='stHeader']{background:#000;color:#E8EEF5}"
    ".block-container{max-width:none;padding:1rem}"
    "</style>", unsafe_allow_html=True,
)
if HTML.is_file():
    st.caption("Saved HTML demo · interactive charts · frozen results, not live inference")
    components.html(HTML.read_text(encoding="utf-8"), height=1100, scrolling=True)
else:
    st.error("The saved HTML demo was not found in main_output.")