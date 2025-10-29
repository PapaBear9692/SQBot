# SQBot

Medical RAG Web Application

This project is a Python-based web application for a Retrieval-Augmented Generation (RAG) chatbot, built with a modular MVC-like pattern. It uses Flask for the web server, Pinecone as the vector store, and supports swappable components for embedding and language models.

Project Structure

/core: Contains the core logic, structured around interfaces for modularity.

    interfaces.py: Abstract Base Classes (interfaces) for LLMs, Embedders, etc.

    embedders.py: Concrete embedder classes (e.g., SentenceTransformer, Google).

    llms.py: Concrete LLM classes (e.g., Gemini, OpenAI).

    vector_stores.py: Concrete vector store classes (e.g., Pinecone).

    loaders_and_splitters.py: Data loading and text splitting logic.

/utils:

    app.py: The "Controller" - the main Flask application entry point.

    rag_service.py: The "Model" - orchestrates the RAG logic.

    data_pipeline.py: A script to process and upload your data to Pinecone.

    app_config.py: Central configuration to manage API keys and model choices.

    environment.yml: Conda environment file.

.env: Your file for secret API keys (create from .env.example).




Setup Instructions

Create Conda Environment:
    conda env create -f environment.yml
    conda activate medical-rag-app


Set Up API Keys:
    Rename .env.example to .env
    Open the .env file and add your secret keys for Pinecone, Google (Gemini), and OpenAI.

Configure Your Models:
    Open config.py
    Set LLM_PROVIDER to "gemini" or "openai".
    Set EMBEDDER_PROVIDER to "default" (for all-miniLm-l6) or "google" (for text-embedder-small).
    Set your PINECONE_INDEX_NAME. You must create this index in your Pinecone account first.

Prepare Your Data:
    Create a directory named data in the root of the project.
    Place your source text files (e.g., .txt files) inside the data directory.

Run the Data Pipeline:
    This script will load, chunk, embed, and upload your data to Pinecone.
    This only needs to be run once (or when your data changes).
    python data_pipeline.py

Run the Web Application:
    gunicorn app:app

    Or for development:
    flask run


Access the App:
    Open your web browser and go to http://127.0.0.1:5000 
    (or http://127.0.0.1:8000 if using gunicorn).