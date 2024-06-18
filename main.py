import streamlit as st
import tempfile
import shutil
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize LLM from Ollama
local_model = "mistral"
llm = ChatOllama(model=local_model)

# Template for creating multi-queries from the question
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def process_pdf(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
            shutil.copyfileobj(file, tmpfile)
            tmpfile_path = tmpfile.name
        loader = UnstructuredPDFLoader(file_path=tmpfile_path)
        data = loader.load()
        os.remove(tmpfile_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        logging.info("PDF processed successfully.")
        return chunks
    except Exception as e:
        logging.error(f"Failed to process PDF: {e}")
        st.error("Failed to process PDF. Please try again.")
        return None

def setup_retriever(chunks):
    try:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
            collection_name="local-rag"
        )
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT
        )
        logging.info("Retriever setup successfully.")
        return retriever
    except Exception as e:
        logging.error(f"Failed to setup retriever: {e}")
        st.error("Failed to setup information retrieval. Please try again.")
        return None

def main():
    st.title("PDF Content Query Application")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner('Processing PDF...'):
            chunks = process_pdf(uploaded_file)
            if chunks:
                st.success('PDF processed successfully!')
                retriever = setup_retriever(chunks)
                if retriever:
                    query = st.text_input("Ask a question about the PDF content")
                    if st.button("Answer"):
                        if query:
                            with st.spinner("Fetching answer..."):
                                try:
                                    # Using the chain to answer the question
                                    chain = (
                                        {"context": retriever, "question": RunnablePassthrough()}
                                        | prompt
                                        | llm
                                        | StrOutputParser()
                                    )
                                    response = chain.invoke(query)
                                    st.write(response)
                                except Exception as e:
                                    st.error(f"An error occurred: {str(e)}")
                        else:
                            st.error("Please enter a question to get an answer.")

if __name__ == "__main__":
    main()
