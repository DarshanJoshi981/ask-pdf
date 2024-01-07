import os
import pickle
import streamlit as st
from doc_qa import AskPDF
import warnings
warnings.filterwarnings('ignore')


def main():
    st.set_page_config(
        page_title="Ask PDF",
        page_icon=":mag_right:",
        layout="wide"
    )

    st.title("Ask PDF")
    st.subheader("Unlocking Answers within Documents, Your Instant Query Companion!")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    processing_text = st.empty()
    ask_pdf = AskPDF()  # Move this line outside the if block

    # Check if a new file is uploaded
    if uploaded_file:
        filename = uploaded_file.name
        with open(f'{filename}', 'wb') as f:
            f.write(uploaded_file.getbuffer())

        @st.cache(allow_output_mutation=True)
        def process_data():
            processing_text.text("Splitting text into smaller chunks...⌛️⌛️⌛️")
            docs = ask_pdf.prepare_data(filename)

            # Create a vector database
            processing_text.text("Storing data into the vector database...🗃️")
            vector_store = ask_pdf.create_vector_db(docs)
            with open(f'vector_store_{uploaded_file.name}.pkl', "wb") as f:
                pickle.dump(vector_store, f)

            processing_text.text('Data Loading Completed...✅✅✅')
            return vector_store

        vector_store = process_data()

    with st.form("Question", clear_on_submit=True):
        question = st.text_input("Ask a question:")
        submitted = st.form_submit_button("Submit")

        # Check if a PDF file is uploaded before submitting a question
        if submitted and not uploaded_file:
            st.warning("Please upload a PDF file before submitting a question.")
        elif submitted and not question.strip():  # Check if the question is not empty
            st.warning("Please enter a question before submitting.")
        elif submitted:
            if os.path.exists(f'vector_store_{uploaded_file.name}.pkl'):
                with open(f'vector_store_{uploaded_file.name}.pkl', "rb") as f:
                    vector_store = pickle.load(f)
                with st.spinner("Processing... Please wait."):
                    answer = ask_pdf.run_pdf_assistant(question, vector_store, ask_pdf)
                st.header("Answer")
                st.success(f"{answer}")

if __name__ == "__main__":
    main()
