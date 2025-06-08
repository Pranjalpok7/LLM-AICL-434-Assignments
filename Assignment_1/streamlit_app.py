
import streamlit as st
import requests 
import json     

# --- Configuration ---
FASTAPI_BACKEND_URL = "http://127.0.0.1:8000/"
PROCESS_ALL_ENDPOINT = f"{FASTAPI_BACKEND_URL}/process/all" # Construct the full endpoint URL

st.set_page_config(page_title="NLP Preprocessor", layout="wide")
st.title("📝 NLP Text Preprocessor")
st.markdown("Enter text below to perform various NLP preprocessing tasks using our powerful API!")

st.subheader("Input Text")
default_text = "Dr. Jane Smith from New York visited London on March 10th, 2023, for a conference on Artificial Intelligence by Apple Inc."
user_text = st.text_area("Paste your text here:", value=default_text, height=150, key="user_input_text") 

if st.button("Process Text", type="primary", key="process_button"): 
    if user_text.strip():
        st.info("🚀 Sending text to the NLP API for processing...")

        
        api_payload = {"text": user_text.strip()}

        try:
            
            response = requests.post(PROCESS_ALL_ENDPOINT, json=api_payload, timeout=30)
            response.raise_for_status()
            nlp_results = response.json()

            st.success("✅ Processing Complete!")
            st.subheader("Processed Results:")

            

            
            if "original_text" in nlp_results:
                st.markdown("---") # Horizontal line
                st.markdown(f"**Original Processed Text:** {nlp_results['original_text']}")

            # Display Tokens
            if "tokens" in nlp_results and nlp_results["tokens"]:
                st.markdown("---")
                st.markdown("**Tokens:**")
                # Join tokens into a string for display, or display as a list
                st.text(", ".join(nlp_results["tokens"]))
                

            # Display Lemmas
            if "lemmas" in nlp_results and nlp_results["lemmas"]:
                st.markdown("---")
                st.markdown("**Lemmas (Text -> Lemma):**")
                for item in nlp_results["lemmas"]:
                    st.markdown(f"- `{item.get('text')}` ➔ `{item.get('lemma')}`")

            # Display Stems
            if "stems" in nlp_results and nlp_results["stems"]:
                st.markdown("---")
                st.markdown("**Stems (Text -> Stem):**")
                for item in nlp_results["stems"]:
                    st.markdown(f"- `{item.get('text')}` ➔ `{item.get('stem')}`")

            # Display POS Tags
            if "pos_tags" in nlp_results and nlp_results["pos_tags"]:
                st.markdown("---")
                st.markdown("**Part-of-Speech Tags:**")

                for tag_info in nlp_results["pos_tags"]:
                    explanation = f"({tag_info.get('explanation')})" if tag_info.get('explanation') else ""
                    st.markdown(
                        f"- `{tag_info.get('text')}` ➔ **{tag_info.get('pos_tag')}** "
                        f"(`{tag_info.get('tag')}`) {explanation}"
                    )

            # Display Named Entities
            if "named_entities" in nlp_results and nlp_results["named_entities"]:
                st.markdown("---")
                st.markdown("**Named Entities:**")
                for ent_info in nlp_results["named_entities"]:
                    explanation = f"({ent_info.get('explanation')})" if ent_info.get('explanation') else ""
                    st.markdown(
                        f"- `{ent_info.get('text')}` ➔ **{ent_info.get('label')}** {explanation}"
                    )
            st.markdown("---") # Final horizontal line
           
        except requests.exceptions.HTTPError as http_err:
            # Handle HTTP errors (like 400, 422, 500 from FastAPI)
            st.error(f"API Error: {http_err}")
            try:
                # Try to get more detail from the API's JSON error response
                error_detail = http_err.response.json()
                st.json(error_detail)
            except json.JSONDecodeError:
                st.error("Could not parse error response from API.")
        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"Connection Error: Could not connect to the API at {PROCESS_ALL_ENDPOINT}. Is the backend server running?")
        except requests.exceptions.Timeout as timeout_err:
            st.error(f"Timeout Error: The API request timed out. The server might be too busy or the request too large.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    else:
        st.warning("⚠️ Please enter some text to process.")

st.markdown("---")

st.header("Word Embeddings & Neighbors")

embedding_query_word = st.text_input(label="Enter a word to query embeddings:")
top_n_neighbors = st.number_input("Number of neighbors to find:", min_value=1, max_value=50, value=5, step=1)

if st.button("🔍 Get Embedding & Find Neighbors", ...):


    if embedding_query_word.strip():
        processed_query_word = embedding_query_word.strip()

        # --- Call API to get embedding ---
        st.subheader(f"Embedding for '{processed_query_word}'")
        try:
            embedding_url = f"{FASTAPI_BACKEND_URL}/embedding/{processed_query_word}"
            embedding_response = requests.get(embedding_url, timeout=10) # Add timeout
            embedding_response.raise_for_status()
            embedding_data = embedding_response.json()

            if embedding_data.get("found"):
                st.success(f"Embedding found!")

            else:
                st.warning(f"Embedding for '{processed_query_word}' not found in GloVe vocabulary.")
        
        except requests.exceptions.HTTPError as http_err:
            st.error(f"API Error (getting embedding for '{processed_query_word}'): {http_err.response.status_code}")
            try:
                st.json(http_err.response.json())
            except json.JSONDecodeError:
                st.error("Could not parse API error response.")
        except Exception as e:
            st.error(f"Error getting embedding: {e}")


        
        st.subheader(f"Nearest Neighbors for '{processed_query_word}' (Top {top_n_neighbors})")
        try:
            neighbors_url = f"{FASTAPI_BACKEND_URL}/nearest-neighbors/{processed_query_word}"
            params = {"top_n": top_n_neighbors}
            neighbors_response = requests.get(neighbors_url, params=params, timeout=20) 
            neighbors_response.raise_for_status()
            neighbors_data = neighbors_response.json() 

            if neighbors_data:
                st.markdown("Found neighbors:")
                for neighbor in neighbors_data:
                    
                    st.markdown(f"- `{neighbor.get('word')}` (Similarity: {neighbor.get('similarity'):.4f})")
            else:
                st.info("No distinct neighbors found (word might not be in vocabulary or is unique in its context).")
        
        except requests.exceptions.HTTPError as http_err:
            st.error(f"API Error (getting neighbors for '{processed_query_word}'): {http_err.response.status_code}")
            try:
                st.json(http_err.response.json())
            except json.JSONDecodeError:
                st.error("Could not parse API error response.")
        except Exception as e:
            st.error(f"Error getting neighbors: {e}")

    else:
        st.warning("Please enter a word for embedding/neighbor search.")


st.markdown("---")