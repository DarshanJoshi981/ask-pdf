from dotenv import load_dotenv
load_dotenv()

import pickle
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
import os
from ocr import convert_pdf_to_images, extract_text_with_easyocr
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFaceHub

load_dotenv()

# @st.cache_resource
def create_vector_store(file_path):
    pdf_loader = PyPDFLoader(file_path)
    docs = pdf_loader.load()
    raw_text = ''
    for doc in docs:
        raw_text += doc.page_content

    if len(raw_text) < 10:
        raw_text = extract_text_with_easyocr(convert_pdf_to_images(file_path))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=200
    )
    texts = text_splitter.split_text(raw_text)
    # # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    vectorstore_faiss = FAISS.from_documents(
        documents=docs,
        embedding=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base"),
    )
    return vectorstore_faiss

def create_prompt_template():
    prompt_template = """
    Human: Answer the question as a full sentence from the context provided. If you don't know the answer, don't try to make up an answer.
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:"""
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template
    )
    return prompt


# @st.cache_resource
def create_retrieval_chain(vector_store, prompt_template):
    qa = RetrievalQA.from_chain_type(
        llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature": 0.5, "max_new_tokens": 2000}),
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        ),
        chain_type_kwargs={"prompt": prompt_template},
    )

    return qa


def generate_response(chain, input_question):
    answer = chain({"query": input_question})
    return answer["result"]


def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size_bytes = file.tell()
    file_size_mb = file_size_bytes / (1024 * 1024)  # Convert bytes to megabytes
    file.seek(0)
    return file_size_mb


# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        if len(history["generated"][i]) == 0:
            message("Please reframe your question properly", key=str(i))
        else:
            message(history["generated"][i],key=str(i))


def create_folders_if_not_exist(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def main():
    st.set_page_config(
        page_title="Ask PDF",
        page_icon=":mag_right:",
        layout="wide"
    )

    st.title("Ask PDF")
    st.subheader("Unlocking Answers within Documents, Your Instant Query Companion!")

    # Sidebar for file upload
    st.sidebar.title("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("", label_visibility='collapsed', type=["pdf"])

    create_folders_if_not_exist("data", "data/pdfs", "data/vectors")

    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.generated = [f"Ask me a question about {uploaded_file.name}" if uploaded_file else ""]
        st.session_state.past = ["Hey there!"]
        st.session_state.last_uploaded_file = uploaded_file.name if uploaded_file else None

    if uploaded_file is not None:
        filepath = "data/pdfs/" + uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        vector_file = os.path.join('data/vectors/', f'vector_store_{uploaded_file.name}.pkl')

        # Display the uploaded file name in the sidebar
        st.sidebar.markdown(f"**Uploaded file:** {uploaded_file.name}")

        if not os.path.exists(vector_file) or "ingested_data" not in st.session_state:
            with st.spinner('Embeddings are in process...'):
                ingested_data = create_vector_store(filepath)
                with open(vector_file, "wb") as f:
                    pickle.dump(ingested_data, f)
                st.session_state.ingested_data = ingested_data
                st.success('Embeddings are created successfully! ✅✅✅')
        else:
            ingested_data = st.session_state.ingested_data

        prompt = create_prompt_template()
        chain = create_retrieval_chain(ingested_data, prompt)

        user_input = st.chat_input(placeholder="Ask a question")

        if user_input:
            answer = generate_response(chain, user_input)
            st.session_state.past.append(user_input)
            response = answer
            st.session_state.generated.append(response)

        # Display conversation history using Streamlit messages
        if st.session_state.generated:
            display_conversation(st.session_state)

if __name__ == "__main__":
    main()

