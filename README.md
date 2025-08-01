# Talent-Acquisition-RAG-Application

## Overview
This project is a GenAI Resume Screening ChatBot that helps talent acquisition teams streamline candidate selection using RAG (Retrieval-Augmented Generation).

It enables:

- ✅ Automated CV processing (extract, chunk, and store in Milvus Vector DB)

- ✅ Natural Language Search for candidates using job descriptions

- ✅ AI-Powered Recommendations based on candidate profiles

- ✅ Simple UI for uploading CVs and chatting with the bot

- ✅ Fully containerized deployment via Docker

## How to Run the Application

1. Install Dependencies

   Run the following command inside the project directory:

   ```bash
   pip install -r requirements.txt
    ```

2. Start the Streamlit Chatbot
    ```bash
     streamlit run app/app.py
    ```
    This starts the chatbot UI, accessible at `http://localhost:8501`.
  
3. Upload CVs & Search for Candidates

    - Upload DOCX/PDF CVs via the UI
  
    - Enter a job description or required skills
  
    - Get recommended candidates!
  
## 🐳 Docker Deployment

To run the application inside a Docker container, follow these steps:
1. Build the Docker Image
   ```bash
     docker build -t my-streamlit-app .
    ```
3. Run the Docker Container
   ```bash
     docker run -p 8501:8501 my-streamlit-app
    ```
   This will start the chatbot inside a container, accessible at `http://localhost:8501`.
