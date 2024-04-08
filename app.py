from transformers import pipeline
import streamlit as st

# Initialize the sentiment analysis model
analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Customize page configuration with a unique title, icon, and layout
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="wide")

# Use columns to create a more interesting layout
col1, col2 = st.columns([1, 3])

with col1:
    st.image("https://source.unsplash.com/random/800x800", caption="Explore the sentiment of your text")

with col2:
    st.title("Explore Text Sentiments")
    st.caption("Powered by AI and HuggingFace")

    # Setup session state to manage the app's state
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = ""

    input_text = st.text_area("Enter your text", height=200, help="Type the text you want to analyze here")

    if st.button("Analyze Sentiment"):
        if input_text:
            analysis_result = analyzer(input_text)
            st.session_state["analysis_result"] = analysis_result[0]['label']
        else:
            st.warning("Please enter some text to proceed with the analysis.")

    # Display the sentiment analysis result in a visually appealing way
    st.subheader("Sentiment Analysis Result")
    if st.session_state["analysis_result"] == "POSITIVE":
        st.markdown(f"<h3 style='color:green;'>{st.session_state['analysis_result']}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red;'>{st.session_state['analysis_result']}</h3>", unsafe_allow_html=True)

# Add a footer for a more professional look
st.markdown("---")
st.markdown("👩‍💻 Sentiment Analyzer App by [Your Name](https://www.yourwebsite.com)")
