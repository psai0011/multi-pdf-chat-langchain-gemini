import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectore_store(text_chunks):
    if not text_chunks:
        st.error("No text found to create a FAISS index.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")  # Corrected model name
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    os.makedirs("faiss-index", exist_ok=True)  # Ensure directory exists
    vector_store.save_local("faiss-index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in 
    the provided context, just say, \"The answer is not available in the context.\" Don't provide a wrong answer.
    
    Context:
    {context}?
    
    Question:
    {question}
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")  # Corrected model name

        # Check if FAISS index exists before loading
        if not os.path.exists("faiss-index/index.faiss"):
            st.error("No FAISS index found. Please upload PDFs and process them first.")
            return "No FAISS index found. Please upload PDFs and process them first."

        vector_store = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)

        if not docs:
            st.warning("No relevant information found in the PDFs.")
            return "No relevant information found."

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]

    except Exception as e:
        st.error(f"Error processing query: {e}")
        return "An error occurred."

def main():
    st.set_page_config(page_title="Multi-Language Invoice Extractor using Gemini Pro", layout="wide")
    st.sidebar.title("📂 Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    process_pdfs = st.sidebar.button("Submit")  # Added Submit button
    
    if process_pdfs and uploaded_files:
        with st.spinner("Processing PDFs..."):
            text = get_pdf_text(uploaded_files)
            if text.strip():
                text_chunks = get_text_chunks(text)
                get_vectore_store(text_chunks)
                st.sidebar.success("PDFs processed successfully!")
            else:
                st.sidebar.error("No extractable text found in the uploaded PDFs.")
    
    st.title("Multi-Language Invoice Extractor using Gemini Pro")
    user_question = st.text_input("Ask a question about the PDFs:")
    
    if user_question:
        response = user_input(user_question)
        st.write("**Reply:**", response)

if __name__ == "__main__":
    main()