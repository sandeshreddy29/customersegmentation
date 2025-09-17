import streamlit as st
from pathlib import Path
st.set_page_config(page_title="Hello Streamlit", layout="wide")

st.title("✅ Streamlit is rendering")
st.write("If you can see this, the frontend is fine.")

BASE_DIR = Path(__file__).resolve().parent
st.caption(f"Working dir: {BASE_DIR}")
st.caption(f"Files in ./artifacts/model: {[p.name for p in (BASE_DIR/'artifacts'/'model').glob('*')]}")

