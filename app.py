import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import HfApi
import time

# Streamlit app setup - This must be the first Streamlit command
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")

# Title and header
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Hugging Face API Token and URL
with st.sidebar:
    hf_api_key = st.text_input("Hugging Face API Token", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Validate Hugging Face API Key
if hf_api_key:
    api = HfApi()
    try:
        api.whoami(token=hf_api_key)  # Simple API call to test token validity
    except Exception as e:
        st.error("Invalid Hugging Face API Token. Please check and try again.")
        st.stop()

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_api_key)

# Define prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def retry_request(chain, docs, retries=5, delay=30):
    for attempt in range(retries):
        try:
            return chain.run(docs)
        except Exception as e:
            if '429' in str(e):  # Check if rate limit error
                st.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
                delay *= 2  # Exponential backoff
            else:
                raise e

# Handle summarization
if st.button("Summarize the Content from YT or Website"):
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or a website URL.")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load data from URL
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                docs = loader.load()

                # Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                
                # Retry logic with exponential backoff
                output_summary = retry_request(chain, docs)
                st.success(output_summary)
        except Exception as e:
            st.error(f"Error occurred: {e}")