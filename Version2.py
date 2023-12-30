import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from graphviz import Source
import streamlit.components.v1 as components
import hashlib
from streamlit_option_menu import option_menu

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = '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'
api_key = os.getenv('OPENAI_API_KEY')

def read_cbl_file(cbl_file):
    try:
        return cbl_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading File: {e}")
        return None

def process_cobol_file(cbl_file):
    if cbl_file is not None:
        cobol_text = read_cbl_file(cbl_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=cobol_text)
        store_name = cbl_file.name[:-4]
        vector_store = load_or_create_vector_store(store_name, chunks)
        return vector_store
    return None

def load_or_create_vector_store(store_name, chunks):
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            return pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
        return vector_store

def process_query(vector_store, query_context):
    if vector_store and query_context:
        docs = vector_store.similarity_search(query=query_context, k=3)
        llm = OpenAI(model_name= 'gpt-4')
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query_context)
            st.write(response)

def ask_query_and_process(cbl_file, query_context):
    vector_store = process_cobol_file(cbl_file)
    process_query(vector_store, query_context)

def render_mermaid_diagram(mermaid_code):
    if not mermaid_code or mermaid_code.strip() == "":
        st.error("Mermaid code is empty. Please ensure the syntax is correct.")
        return

    # Create a unique hash for the diagram
    unique_id = hashlib.md5(mermaid_code.encode()).hexdigest()

    # HTML and JavaScript to render the Mermaid diagram
    mermaid_html_template = f"""
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid@8/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{startOnLoad: true}});
            </script>
        </head>
        <body>
            <div id="{unique_id}" class="mermaid">
                {mermaid_code}
            </div>
            <script>
                document.addEventListener("DOMContentLoaded", function() {{
                    mermaid.init(undefined, "#{unique_id}");
                }});
            </script>
        </body>
    </html>
    """
    components.html(mermaid_html_template, height=1000)

def HomePage():
    st.text("Welcome to the COBOL assistant")
    st.text(""" Instructions:
            1) Upload your Cobol File
            2) use the side bar to find the Navigation and Navigate through the different options
            """)

def Overview_Page():
        query_context = """I want you to act as a Software Engineer specialising in COBOL code analysis. Given the COBOL file give me a high level overview of the code"""
        ask_query_and_process(COBOLFile, query_context)

def AsIs_ArchitecturePage():
    global COBOLFile
    st.title("As-Is Architecture")

    query_context = "give an overview of the code and analyse the structure of the program, identifying key sections, paragraphs and their relationships. This will help you create a high-level overview of the programs flow and organisation. Given the structure, create a basic mermaid diagram code to visually represent these elements. ensure the code primarilly focusses on the paragraphs as just create a simplified representation nothing too complex."
    ask_query_and_process(COBOLFile, query_context)

    if AsIs_ArchitecturePage is True:
        # Your prompt
        query_context = "Generate a Mermaid diagram code for a simple project management flowchart..."
        # Assume COBOLFile is defined
        mermaid_code = ask_query_and_process(COBOLFile, query_context)
    else:
        # Text area for manually entering or editing the Mermaid code
        mermaid_code = st.text_area("Copy the code from above into this box to visualise the diagram", height=300)

    if mermaid_code and st.button("Render Diagram", key="render_diagram"):
        render_mermaid_diagram(mermaid_code)

def ToBe_ArchitecturePage():
    global COBOLFile
    st.title("As-Is Architecture")
    query_context = "give an overview of how the structure of the program can be migrated to a microservice, identifying key sections, paragraphs and their relationships. This will help you create a high-level overview of the programs flow and organisation. Given the structure, create a basic mermaid diagram code to visually represent these elements. Just create a simplified representation nothing too complex."
    ask_query_and_process(COBOLFile, query_context)

    if AsIs_ArchitecturePage is True:
        # Your prompt
        query_context = "Generate a Mermaid diagram code for a simple project management flowchart..."
        # Assume COBOLFile is defined
        mermaid_code = ask_query_and_process(COBOLFile, query_context)
    else:
        # Text area for manually entering or editing the Mermaid code
        mermaid_code = st.text_area("Copy the code above to visualise the As-Is", height=300)

    if mermaid_code and st.button("Render Diagram", key="render_diagram"):
        render_mermaid_diagram(mermaid_code)

def QA_Page():
    global COBOLFile
    st.title("Chat")
    user_query = st.text_input("Ask questions about your cobol code: ")

    if user_query:
        # Create a unique key for each query
        query_key = f"response_{hash(user_query)}"

        context = "You're a software engineer specialising in modernising systems and specifically in COBOL."
        query_context = context + "\nQuestion: " + user_query
        ask_query_and_process(COBOLFile, query_context)
        # Display the response
    else:
        st.write("Enter a question to get started.")

def main():
    global COBOLFile
    st.title("Invent MainFrame Moderniser POC")
    with st.sidebar:
        selected = option_menu(
            menu_title= "Main Menu",
            options= ["Home", "Overview", "As-Is Architecture Assistant", "To-Be Architecture Assistant", "COBOL File Q&A"]
        )
    
    COBOLFile = st.file_uploader("Upload your File", type='CBL', key="unique_cobol_file_uploader")

    if selected == "Home":
        HomePage()
    elif selected == "Overview":
        Overview_Page()
    elif selected == "As-Is Architecture Assistant":
        AsIs_ArchitecturePage()
    elif selected == "To-Be Architecture Assistant":
        ToBe_ArchitecturePage()
    elif selected == "COBOL File Q&A":
        QA_Page()

if __name__ == "__main__":
    main()
