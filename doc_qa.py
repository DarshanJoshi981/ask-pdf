from dotenv import load_dotenv
load_dotenv()

import os
from ocr import convert_pdf_to_images, extract_text_with_easyocr
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI


class AskPDF:
    def __init__(self):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.75, google_api_key=google_api_key)
        self.gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        self.prompt = self.create_prompt_template()


    def prepare_data(self, file_path):
        pdf_loader = PyPDFLoader(file_path)
        docs = pdf_loader.load()
        raw_text = ''
        for doc in docs:
            raw_text += doc.page_content

        if len(raw_text) < 10:
            raw_text = extract_text_with_easyocr(convert_pdf_to_images(file_path))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=15000, chunk_overlap=200
        )
        texts = text_splitter.split_text(raw_text)
        # # Create multiple documents
        docs = [Document(page_content=t) for t in texts]
        return docs

    def create_vector_db(self, docs):
        vectorstore_faiss = FAISS.from_documents(
            documents=docs,
            embedding=self.gemini_embeddings,
        )
        return vectorstore_faiss

    def create_prompt_template(self):
        prompt_template = """
        Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, don't try to make up an answer.
        <context>
        {context}
        </context>
        Question: {question}
        Assistant:"""
        prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )
        return prompt

    def create_conversation_chain(self, vector_store, prompt_template):
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 6}
            ),
            # return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
            # verbose = True
        )

        return qa


    def run_pdf_assistant(self, query, vector_store, assistant):
        conversation_chain = assistant.create_conversation_chain(vector_store, self.prompt)
        answer = conversation_chain({'query': query})
        return answer['result']

