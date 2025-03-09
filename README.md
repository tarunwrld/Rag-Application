# Rag-Application

This is a Streamlit-based application that allows users to upload a PDF document, extract text from it, and ask questions using Google's Gemini API. The application uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant chunks of text from the document and generate answers using the Gemini model.

## Features

- **PDF Text Extraction**: Extract text from uploaded PDF documents using PyMuPDF (fitz).
- **Text Chunking**: Split the extracted text into overlapping chunks using NLTK tokenization.
- **Embedding Generation**: Generate embeddings for text chunks using Google's Gemini Embedding API.
- **FAISS Vector Storage**: Store embeddings in a FAISS vector database for efficient similarity search.
- **Question Answering**: Retrieve relevant chunks based on a user query and generate answers using the Gemini model.

## Prerequisites

Before running the application, ensure you have the following:

- Python 3.8 or higher.
- **Google Gemini API Key**: Obtain an API key from Google Cloud Console.
- **NLTK Data**: The punkt tokenizer data must be downloaded.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/tarunwrld/Rag-Application.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up the environment:

    - Create a `.env` file in the root directory and add your Gemini API key:

      ```
      GEMINI_API_KEY=your_api_key_here
      ```

    - Alternatively, you can set the API key:

      ```bash
      GEMINI_API_KEY=your_api_key_here
      ```

4. Download NLTK data:

    ```python
    import nltk
    nltk.download("punkt")
    ```

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2. Open your browser and navigate to `http://localhost:8501`.
3. Upload a PDF document using the file uploader.
4. Once the text is extracted and processed, enter your question in the input box.
5. The application will display the relevant chunks of text and generate an answer using the Gemini model.


## Dependencies

- **Streamlit**: For building the web application.
- **PyMuPDF (fitz)**: For extracting text from PDFs.
- **NLTK**: For tokenizing text into chunks.
- **Google Generative AI**: For generating embeddings and answers.
- **FAISS**: For storing and searching embeddings efficiently.
- **NumPy**: For handling numerical operations.

## Configuration

- **Gemini API Key**: Set your API key in the `.env` file or as an environment variable.
- **NLTK Data Path**: The application uses `/tmp/nltk_data` as the default directory for NLTK data. You can change this by modifying the `nltk_data_dir` variable in `app.py`.

## Troubleshooting

### NLTK Data Not Found

If you encounter errors related to NLTK data (e.g., `punkt` not found), ensure that:

1. The `punkt` tokenizer is downloaded:

    ```python
    import nltk
    nltk.download("punkt")
    ```

2. The `NLTK_DATA` environment variable is set correctly:

    ```bash
    export NLTK_DATA=/tmp/nltk_data
    ```

### API Key Errors

If the application fails to authenticate with the Gemini API:

1. Verify that the `GEMINI_API_KEY` is set correctly in the `.env` file or environment variables.
2. Ensure the API key has the necessary permissions.

### PDF Extraction Issues

If the application cannot extract text from a PDF:

1. Ensure the PDF is not image-based or scanned.
2. Use a different PDF file to test.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


