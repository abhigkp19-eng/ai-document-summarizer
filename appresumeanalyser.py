import streamlit as st
import PyPDF2
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Azure OpenAI setup
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment = os.getenv("DEPLOYMENT_NAME")

# Function to extract text from PDF
def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# UI starts here 👇 IMPORTANT
st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

if uploaded_file:
    text = extract_text(uploaded_file)

    st.subheader("📜 Resume Preview")
    st.write(text[:1000])

    if st.button("Analyze Resume"):
        with st.spinner("Analyzing..."):

            prompt = f"""
            Analyze the following resume:

            Give:
            1. Resume Score (out of 100)
            2. Key Skills
            3. Missing Skills
            4. Improvement Suggestions

            Resume:
            {text}
            """

            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are an expert HR."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700
            )

            result = response.choices[0].message.content

            st.subheader("📊 Analysis Result")
            st.write(result)