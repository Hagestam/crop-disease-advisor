"""
Streamlit frontend for the Crop Disease Detection & Advisory System.
Run with: streamlit run frontend/app.py
"""

import streamlit as st
import requests
from PIL import Image
import io

API_BASE = "http://localhost:8000"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Disease Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌱 Crop Disease Advisor")
    st.markdown(
        "Upload a photo of a crop leaf to detect diseases and receive "
        "AI-generated treatment recommendations."
    )
    st.divider()
    st.caption("**Model:** EfficientNet-B0 (INT8 TFLite)")
    st.caption("**Classes:** 38 disease/healthy classes")
    st.caption("**RAG:** LangChain + ChromaDB + Claude")
    st.divider()

    # API health check
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            st.success("API Online ✅")
        else:
            st.error("API Error")
    except Exception:
        st.error("API Offline — start the FastAPI server")

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("Crop Disease Detection & Advisory System")

tab1, tab2 = st.tabs(["📷 Analyze Image", "💬 Ask Advisory"])

# ── TAB 1: Image Analysis ──────────────────────────────────────────────────────
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a leaf image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Clear, well-lit photos of individual leaves work best",
    )

    if uploaded_file:
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            st.caption(f"Size: {image.size[0]}×{image.size[1]}px | "
                       f"Format: {image.format or 'unknown'}")

        with col_result:
            st.subheader("🔍 Analysis")
            analyze_btn = st.button("Analyze Image", type="primary", use_container_width=True)

            if analyze_btn:
                with st.spinner("Running detection and generating advisory..."):
                    try:
                        uploaded_file.seek(0)
                        response = requests.post(
                            f"{API_BASE}/advisory/analyze",
                            files={"file": (uploaded_file.name,
                                            uploaded_file.getvalue(),
                                            uploaded_file.type or "image/jpeg")},
                            timeout=60,
                        )

                        if response.status_code == 200:
                            result = response.json()
                            top = result["top_prediction"]

                            # ── Detection Result ──
                            if top["is_healthy"]:
                                st.success(f"✅ **Healthy** — {top['crop']}")
                            else:
                                st.error(f"⚠️ **Disease Detected**")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Crop", top["crop"])
                                with col_b:
                                    st.metric("Disease", top["disease"])

                            # ── Confidence Bar Chart ──
                            st.markdown("**Confidence Scores**")
                            for pred in result["predictions"]:
                                label = f"{pred['crop']} — {pred['disease']}"
                                st.progress(
                                    pred["confidence"],
                                    text=f"{label}: {pred['confidence']:.1%}",
                                )

                            # ── Advisory ──
                            st.divider()
                            st.subheader("🌿 Treatment Advisory")
                            st.markdown(result["advisory"])

                            # ── Sources ──
                            if result.get("sources"):
                                with st.expander("📚 Source Documents"):
                                    for src in result["sources"]:
                                        st.caption(f"• {src}")

                        else:
                            st.error(f"API error: {response.status_code} — {response.text}")

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API. Make sure the FastAPI server is running.")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

# ── TAB 2: Free-form Advisory Chat ────────────────────────────────────────────
with tab2:
    st.subheader("💬 Ask the Advisory System")
    st.markdown(
        "Ask any question about crop disease management. "
        "The system searches agricultural extension documents to answer."
    )

    crop_col, disease_col = st.columns(2)
    with crop_col:
        crop_input = st.text_input("Crop (e.g. Tomato, Apple, Corn)")
    with disease_col:
        disease_input = st.text_input("Disease (e.g. Late blight, Leaf spot)")

    if st.button("Get Advisory", type="primary") and crop_input and disease_input:
        with st.spinner("Searching extension documents and generating advisory..."):
            try:
                response = requests.post(
                    f"{API_BASE}/advisory/",
                    json={"crop": crop_input, "disease": disease_input},
                    timeout=60,
                )
                if response.status_code == 200:
                    result = response.json()
                    st.markdown(result["advisory"])
                    if result.get("sources"):
                        with st.expander("📚 Sources"):
                            for src in result["sources"]:
                                st.caption(f"• {src}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.caption("Advisory generated from agricultural extension documents via RAG. "
               "Always consult a local agronomist for critical decisions.")