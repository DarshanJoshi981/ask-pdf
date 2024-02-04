import base64
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
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI


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
        chunk_size=20000, chunk_overlap=200
    )
    texts = text_splitter.split_text(raw_text)
    # # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    vectorstore_faiss = FAISS.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
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
        llm= GoogleGenerativeAI(model="gemini-pro", temperature=0.9),
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        ),
        # return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
        # verbose = True
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


@st.cache_data
#function to display the PDF of a given file
def display_pdf(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
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

    uploaded_file = st.file_uploader("", label_visibility='collapsed', type=["pdf"])

    create_folders_if_not_exist("data", "data/pdfs", "data/vectors")

    # Check if the uploaded file has changed and reset session state variables accordingly
    if st.session_state.get("uploaded_file") != uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state["generated"] = [f"Ask me a question about {uploaded_file.name}"]
        st.session_state["past"] = ["Hey there!"]

    if uploaded_file is not None:
        file_size_mb = get_file_size(uploaded_file)

        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{file_size_mb:0.2f} MB"
        }

        filepath = "data/pdfs/" + uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        vector_file = os.path.join('data/vectors/', f'vector_store_{uploaded_file.name}.pkl')

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            if file_size_mb > 2:
                st.warning("The uploaded PDF is too large to preview. Please proceed with your questions.")
            else:
                pdf_view = display_pdf(filepath)

        with col2:
            with st.spinner('Embeddings are in process...'):
                if os.path.exists(vector_file):
                    with open(vector_file, "rb") as f:
                        ingested_data = pickle.load(f)
                else:
                    ingested_data = create_vector_store(filepath)
                    with open(vector_file, "wb") as f:
                        pickle.dump(ingested_data, f)

            prompt = create_prompt_template()
            chain = create_retrieval_chain(ingested_data, prompt)
            st.success('Embeddings are created successfully! ✅✅✅')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

            user_input = st.text_input("")
            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = [f"Ask me a question about {uploaded_file.name}"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]

            # Search the database for a response based on user input and update session state
            if user_input:
                answer = generate_response(chain, user_input)
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)

if __name__ == "__main__":
    main()
