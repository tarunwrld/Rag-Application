import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize
import google.generativeai as genai
import faiss
import numpy as np
import os

# Ensure NLTK resources are downloaded
nltk.download("punkt")

# Configure Gemini API (use environment variable or Streamlit secrets for API key)

GEMINI_API_KEY = ""  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Function to extract text from the uploaded PDF using PyMuPDF (fitz)
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Function to split text into overlapping chunks using NLTK tokenization
def split_text_into_chunks(text, chunk_size=500, overlap=100):
    try:
        words = word_tokenize(text)
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")
        return []

# Function to generate embeddings for a list of text chunks
def generate_embeddings(chunks, title="PDF Document"):
    embeddings = []
    for chunk in chunks:
        try:
            embedding = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document",
                title=title
            )
            embeddings.append(embedding["embedding"])
        except Exception as e:
            st.error(f"Error generating embedding for chunk: {e}")
    return embeddings

# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings):
    try:
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        return index
    except Exception as e:
        st.error(f"Error storing embeddings in FAISS: {e}")
        return None

# Function to retrieve relevant chunks using FAISS
def retrieve_relevant_chunks(query_embedding, index, chunks, top_k=3):
    try:
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        relevant_chunks = [chunks[i] for i in indices[0]]
        return relevant_chunks
    except Exception as e:
        st.error(f"Error retrieving relevant chunks: {e}")
        return []

# Function to generate an answer using Gemini API
def generate_answer(query, context_chunks):
    try:
        context = "\n".join(context_chunks)
        prompt = f"""
        Context:
        {context}
        Question:
        {query}
        Answer the question based on the context provided above.
        """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Unable to generate an answer due to an error."

# Streamlit UI
with st.sidebar:
    st.title("Navigation")
    hide_st_style = '''
        <style>
        MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    '''
    st.markdown(hide_st_style, unsafe_allow_html=True)
    page = st.radio("Options", ["Home", "Privacy Policy"], label_visibility="collapsed")

if page == "Home":
    st.title("Gemini RAG Application")
    st.markdown("Upload a PDF document and ask questions to get answers using Google's Gemini API.")

    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

    if pdf_file is not None:
        with st.spinner("Extracting text..."):
            extracted_text = extract_text_from_pdf(pdf_file)
        
        if extracted_text:
            with st.spinner("Splitting text into overlapping chunks..."):
                chunks = split_text_into_chunks(extracted_text, chunk_size=500, overlap=100)
            
            if chunks:
                with st.status(f"Total chunks: {len(chunks)}"):
                    for i, chunk in enumerate(chunks):
                        st.subheader(f"Chunk {i + 1}")
                        st.text_area(f"Chunk {i + 1} Text", chunk, height=200, key=f"chunk_{i}")
                
                with st.spinner("Generating embeddings..."):
                    embeddings = generate_embeddings(chunks)
                
                if embeddings:
                    with st.spinner("Storing embeddings in FAISS..."):
                        index = store_embeddings_in_faiss(embeddings)
                    
                    if index:
                        st.success("Embeddings have been successfully stored in the FAISS vector database.")
                        
                        query = st.text_input("Enter your question:")
                        if query:
                            with st.spinner("Generating query embedding..."):
                                query_embedding = genai.embed_content(
                                    model="models/embedding-001",
                                    content=query,
                                    task_type="retrieval_query"
                                )["embedding"]
                            
                            with st.spinner("Retrieving relevant chunks..."):
                                relevant_chunks = retrieve_relevant_chunks(query_embedding, index, chunks, top_k=3)
                            
                            if relevant_chunks:
                                with st.status("### Relevant Context Chunks:"):
                                    for i, chunk in enumerate(relevant_chunks):
                                        st.subheader(f"Chunk {i + 1}")
                                        st.text_area(f"Relevant Chunk {i + 1} Text", chunk, height=200, key=f"relevant_chunk_{i}")
                                
                                with st.spinner("Generating answer..."):
                                    answer = generate_answer(query, relevant_chunks)
                                    st.write("### Answer:")
                                    st.write(answer)
                            else:
                                st.warning("No relevant chunks found.")
                    else:
                        st.error("Failed to store embeddings in FAISS.")
                else:
                    st.error("Failed to generate embeddings.")
            else:
                st.error("No chunks generated from the text.")
        else:
            st.error("No text extracted. The document might be image-based or corrupted.")
