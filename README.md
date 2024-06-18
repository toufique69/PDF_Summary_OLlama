# Real-Time PDF Summarization Chatbot using OLlama

## Overview
The goal of this project is to develop a "Real-Time PDF Summarization Web Application Using the open-source model Ollama". This application enables users to upload PDF files and query their contents in real-time, providing summarized responses in a conversational style akin to ChatGPT.

## Features
- **PDF Upload**: Users can upload multiple PDF files.
- **Real-Time Summarization**: The application processes and summarizes the content of the PDFs in real-time.
- **Chatbot Interface**: Users can ask questions about the uploaded PDFs and receive detailed answers.
- **Stylish Interface**: Custom CSS for an enhanced user experience.

### Steps
1. **Clone the repository:**

2. **Download and Setup Ollama:**
- Visit the Ollama GitHub page.
- Ensure you have the required model such as "mistral" by running: ollama pull mistral

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**
    ```bash
    streamlit run main.py
    ```
   
## Usage
1. Open your web browser and go to the local server URL displayed in the terminal (usually `http://localhost:8501`).
2. Upload your PDF files using the upload section.
3. Once the PDFs are processed, you can ask questions about the content in the chat section.
4. The bot will respond to your questions based on the content of the uploaded PDFs.

## File Descriptions
- **main.py**: The main Streamlit application file that sets up the web interface and handles user interactions.
- **requirements.txt**: List of required Python packages.

## Technologies Used
- **Streamlit**: For building the web interface.
- **LangChain**: For handling text splitting, embeddings, and question-answering chains.
- **UnstructuredPDFLoader**: For loading and processing PDF files.
- **RecursiveCharacterTextSplitter**: For splitting the text into chunks.
- **OllamaEmbeddings**: For generating embeddings for the document chunks.
- **Chroma**: For creating and querying the vector database.
- **ChatOllama**: For generating responses using the Ollama language model.

© TOUFIQUE HASAN - 2024
