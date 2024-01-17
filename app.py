import streamlit as st
from streamlit_chat import message
from PIL import Image
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from sentence_transformers import SentenceTransformer
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile
import base64



def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('5053309.jpg')

load_dotenv()



def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hola hazme preguntas sobre tu documento 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hola IDGTA! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Pregunta:", placeholder="Realiza preguntas sobre tu documento", key='input')
            #user_input+ ", en español"
            submit_button = st.form_submit_button(label='PREGUNTAR')

        if submit_button and user_input:
            if user_input.strip():  # Verifica si la pregunta no está en blanco
                with st.spinner('Generando respuesta....'):
                    output = conversation_chat(user_input, chain, st.session_state['history'])
                    
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
            else:
                # Muestra una pantalla emergente si la pregunta está en blanco
                st.warning('¡Por favor, ingresa una pregunta!')

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="black")
                message(st.session_state["generated"][i], avatar_style="black")

##TENGO QUE ELIMINAR LOS AVATARES


def create_conversational_chain(vector_store):
    load_dotenv()
    # Create llm
    llm = CTransformers(model="Llama-2-7b-ft-instruct-es.gguff",
                        streaming=True, 
                        callbacks=[StreamingStdOutCallbackHandler()],
                        model_type="llama",
                        config={'max_new_tokens': 500, 'temperature': 0.2})
        
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("GOBIERNO AUTONOMO MUNICIPAL DE LA PAZ")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.header("Chatea con tus documentos a través de llama2 tipo PDF, docs :books:")
    
    # Initialize Streamlit
   # st.sidebar.title("Procesa aqui tus Documentos")
    
    
  
    #st.sidebar
    uploaded_files = st.file_uploader("Subir archivos", accept_multiple_files=True)


    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)
        
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="jaimevera1107/all-MiniLM-L6-v2-similarity-es", 
                                           model_kwargs={'device': 'cpu'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        
        display_chat_history(chain)


if __name__ == "__main__":
    main()



