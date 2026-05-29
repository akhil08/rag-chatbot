import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
# import google.generativeai as genai
# from PIL import Image

from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")
# Gemini API setup
# genai.configure(api_key="")

# Create Groq client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Streamlit title
st.title("PDF RAG Chatbot")

website_url = st.text_input(
    "Enter website URL:"
)
website_text = ""

if website_url:
    # Clear old chat history
    st.session_state.messages = []

    try:

        # Download webpage
        response = requests.get(website_url)

        # Parse HTML
        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Extract webpage text
        website_text = soup.get_text()

        st.success("Website loaded successfully!")

    except:
        st.error("Failed to load website.")

# Upload image
# uploaded_image = st.file_uploader(
#     "Upload an image",
#     type=["png", "jpg", "jpeg"]
# )
# if uploaded_image:

#     st.image(
#         uploaded_image,
#         caption="Uploaded Image",
#         use_container_width=True
#     )

#     image_question = st.text_input(
#         "Ask something about the image:"
#     )

#     if image_question:

        # Open image
        # image = Image.open(uploaded_image)

        # # Load Gemini vision model
        # model = genai.GenerativeModel(
        #     "gemini-2.0-flash"
        # )

        # Generate response
        # response = model.generate_content(
        #     [image_question, image]
        # )

        # st.write("### Image Answer:")
        # st.write(response.text)

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)
if uploaded_file:
    st.success("PDF uploaded successfully!")

# Display old messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# Run only if PDF uploaded
if uploaded_file or website_url:

    text = ""

    # Read PDF only if uploaded
    if uploaded_file:

        pdf_reader = PdfReader(uploaded_file)

        for page in pdf_reader.pages:
            text += page.extract_text()

    # @st.cache_resource
    def create_vectorstore(text):

        # Split text
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        if uploaded_file:
            combined_text = text
        else:
            combined_text = website_text

        chunks = text_splitter.split_text(
            combined_text
        )

        # Create embeddings
        embeddings = HuggingFaceEmbeddings()

        # Create FAISS vector store
        vectorstore = FAISS.from_texts(
            chunks,
            embeddings
        )

        return vectorstore

    # Load vectorstore
    vectorstore = create_vectorstore(text)

    # User question
    user_question = st.chat_input(
        "Ask a question:"
    )

    if user_question:

        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(
            user_question,
            k=3
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        # Prompt
        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{user_question}
"""

        with st.spinner("Thinking..."):
            # LLM response
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content
            # Save user message
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )

            # Save assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            # Display latest user message
            with st.chat_message("user"):
                st.write(user_question)

            # Display latest assistant message
            with st.chat_message("assistant"):
                st.write(answer)

            
