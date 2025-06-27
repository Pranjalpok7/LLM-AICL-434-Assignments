# frontend/app.py (Improved Layout Version)

import streamlit as st
import requests
from PIL import Image
import io

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000" # URL of your FastAPI backend

# --- UI Setup ---
st.set_page_config(layout="wide")
st.title("🧠 Simple Multimodal AI Application")
st.write("Powered by Hugging Face BLIP and built with FastAPI & Streamlit.")

# --- Sidebar for Mode Selection ---
st.sidebar.title("Choose a Task")
app_mode = st.sidebar.selectbox("Select the task you want to perform:",
                                ["🖼️ Image Captioning", "❓ Visual Question Answering (VQA)"])

# --- Main Page Content ---

# Shared UI element for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_bytes = uploaded_file.getvalue()

    # --- NEW: Create a two-column layout ---
    col1, col2 = st.columns(2)

    # --- In the first column, display the image ---
    with col1:
        st.subheader("Your Image")
        # Use a fixed width to control the size and prevent it from being too tall.
        st.image(image, caption="Your Uploaded Image", width=400)

    # --- In the second column, display controls and results ---
    with col2:
        # This placeholder will hold the results
        result_placeholder = st.empty()

        if app_mode == "🖼️ Image Captioning":
            st.subheader("Generate Image Caption")
            if st.button("Generate Caption", use_container_width=True):
                with st.spinner("Thinking..."):
                    files = {'image_file': (uploaded_file.name, image_bytes, uploaded_file.type)}
                    try:
                        response = requests.post(f"{BACKEND_URL}/generate-caption", files=files)
                        response.raise_for_status()
                        caption = response.json().get("caption")
                        result_placeholder.success(f"**Generated Caption:** {caption}")
                    except requests.exceptions.RequestException as e:
                        result_placeholder.error(f"Error connecting to backend: {e}")

        elif app_mode == "❓ Visual Question Answering (VQA)":
            st.subheader("Ask a Question About the Image")
            question = st.text_input("Enter your question:", "What is in this image?")

            if st.button("Get Answer", use_container_width=True):
                if not question:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Finding the answer..."):
                        files = {'image_file': (uploaded_file.name, image_bytes, uploaded_file.type)}
                        data = {'question': question}
                        try:
                            response = requests.post(f"{BACKEND_URL}/answer-question", files=files, data=data)
                            response.raise_for_status()
                            answer = response.json().get("answer")
                            # We can display the question permanently and the answer in the placeholder
                            st.write(f"**Your Question:** {question}")
                            result_placeholder.info(f"**Answer:** {answer}")
                        except requests.exceptions.RequestException as e:
                            result_placeholder.error(f"Error connecting to backend: {e}")
else:
    st.info("Please upload an image to get started.")