# Ask PDF

**Ask PDF** is a web application designed to streamline document understanding by providing instant answers to user queries within PDF documents. Whether you're a student, researcher, or professional, **Ask PDF** simplifies the process of extracting valuable insights from lengthy documents.

## How It Works

- **Upload PDF**: Users can upload their PDF documents directly through the file input menu.
- **Text Extraction**: **Ask PDF** extracts text from the uploaded PDF, ensuring no information is missed during the process.
- **Vector Database**: The extracted text is stored in a vector database using embeddings, enabling efficient storage and retrieval of document content.
- **Text Similarity Search**: When a user asks a question, **Ask PDF** performs a text similarity search within the vector database to find relevant content.
- **Output Generation**: Leveraging the Language Model (LLM), **Ask PDF** generates and displays the final output, providing users with accurate and contextually relevant answers to their queries.

## Technologies Used

- **Streamlit**: Provides an intuitive interface for user interaction.
- **EasyOCR**: Ensures accurate text extraction from PDF documents.
- **Langchain**: Empowers document processing with features like text splitting and retrieval-based question answering.
- **Hugging Face Models**: Utilizes state-of-the-art transformer models for precise and context-aware answers.
- **FAISS**: Enables fast and efficient document retrieval, enhancing user experience.

## Getting Started

To get started with **Ask PDF**, simply clone this repository and install the necessary dependencies. Run the application using Streamlit and upload your PDF documents to begin extracting insights instantly.
```bash
streamlit run app.py
```

## Contribution

Contributions to **Ask PDF** are welcome! If you have any ideas for improvement or would like to report a bug, feel free to open an issue or submit a pull request.
