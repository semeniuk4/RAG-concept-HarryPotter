# Harry Potter RAG Question Answering System

This application uses the Retriever-Augmented Generation (RAG) architecture to provide detailed answers to questions about the Harry Potter series. It integrates the OpenAI API for generating responses and features a user-friendly Streamlit interface. The system is specifically trained on Harry Potter-related information and maintains a history of all previous questions asked.

## Features

- **Question Answering**: Interactively ask questions related to the Harry Potter books and receive informative, context-aware answers.
- **RAG Architecture**: Utilizes both a powerful retriever and a generative model to deliver precise and comprehensive responses.
- **Question History**: Tracks and displays a history of previously asked questions for easy reference.

## Prerequisites

An API key from OpenAI is required to run this application. This key is necessary to authenticate API requests and should be set as an environment variable.

## Setup

1. **Running the Application**  

   To run the application, execute the following command:

   ```bash
   streamlit run main.py

## Retraining the Model

If you wish to retrain the system with another content:

- Provide new links containing relevant information in the `scrapper.py` file.
- Run the scrapping script to collect new data and retrain the RAG model with the updated dataset.
