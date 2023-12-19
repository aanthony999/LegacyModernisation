import streamlit as st
from dotenv import load_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
import os
 
# Sidebar contents
with st.sidebar:
    st.title('MainFrame Moderniser')
    
load_dotenv()

software_engineering_context = """
You're a software engineer specialising in modernising systems. I want you to act as a COBOL code analyser and provide suggestions on transforming a monolithic service into a microservice architecture. Analyse the provided COBOL code/file and consider factors like code structure, dependencies, and functionality to generate insightful recommendations. Emphasise the benefits of microservices, scalability, modularity, and maintainability. Provide specific guidance on breaking down the monolith into smaller services, identifying potential service boundaries, and proposing communication mechanisms between them.
"""

def readCBLfile(COBOLFile):
    try:
        return COBOLFile.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading File: {e}")
        return None
 
def main():
    st.header("COBOL Moderniser")
 
    # upload a COBOL file
    COBOLFile = st.file_uploader("Upload your File", type='CBL')
 
    # st.write()
    if COBOLFile is not None:
        cobol_text = readCBLfile(COBOLFile)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=cobol_text)

        # embeddings
        store_name = COBOLFile.name[:-4]
        

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your COBOL file:")

        if query:
        # Combine the user query with the software engineering context
            full_query = software_engineering_context + "\nQuestion: " + query

            # Process the query through the language model
            docs = VectorStore.similarity_search(query=full_query, k=3)

            llm = ChatOpenAI(model_name='gpt-4')  # Specifying the model
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=full_query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()