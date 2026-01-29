import streamlit as st
from transformers import pipeline
from newspaper import Article
import nltk
import time

# 1. Essential Background Setup
@st.cache_resource
def setup_nltk():
    try:
        # These are needed for the 'newspaper' library to read web articles
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass

setup_nltk()

# 2. Page Interface Configuration
st.set_page_config(page_title="AI News Brief", page_icon="📰")

# 3. Load the AI Model
# We use T5-Small because it's fast and doesn't crash the server.
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small")

summarizer = load_summarizer()

# 4. App UI Elements
st.title("📰 AI News Summarizer")
st.write("Enter a news article link to get a concise summary.")

url = st.text_input("🔗 Paste News URL here:", placeholder="https://www.bbc.com/news/...")

if st.button("Summarize Now"):
    if url:
        try:
            with st.spinner('AI is reading and condensing...'):
                start_time = time.time()
                
                # Fetch and extract text from the URL
                article = Article(url)
                article.download()
                article.parse()
                
                # Safety: Summarize only the first 2000 characters to prevent memory errors
                input_text = article.text[:2000]
                
                if len(input_text) < 100:
                    st.error("Text is too short to summarize. Try another link.")
                else:
                    # Execute AI Summarization
                    output = summarizer(input_text, max_length=120, min_length=40, do_sample=False)
                    summary = output[0]['summary_text']
                    
                    # Display the Results
                    st.subheader(f"📄 {article.title}")
                    st.success(summary)
                    
                    # Performance Metrics
                    st.write("---")
                    st.info(f"Summary generated in {round(time.time() - start_time, 2)} seconds.")
                    
        except Exception as e:
            st.error("Could not process this link. Some websites block AI access.")
    else:
        st.warning("Please enter a URL first.")