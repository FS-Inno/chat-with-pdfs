# UI comes here
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def init():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None        
    # if "llm" not in st.session_state:
    #     st.session_state.llm = ChatOpenAI(temperature=0, model_name='gpt-4', openai_api_key=openai_key)
    # if "embeddings" not in st.session_state:
    #     st.session_state.embeddings = OpenAIEmbeddings()    
        #st.session_state.embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

def init_openai_components(openai_key):
    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(temperature=0, model_name='gpt-4', openai_api_key=openai_key)
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)    

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdfReader=PdfReader(pdf)
        for page in pdfReader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    textsplitter=CharacterTextSplitter(separator="\n",chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks=textsplitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    vectorstore = FAISS.from_texts(texts=chunks,embedding=st.session_state.embeddings)
    return vectorstore
    
def get_conversation(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vectorstore.as_retriever(),
        memory = memory    
    )
    return conversation_chain

def handle_user_input(question):
    response = st.session_state.conversation({'question':question})
    st .session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
#            st.write(user_template.replace(
#                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
#           st.write(bot_template.replace(
#                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    #load_dotenv()
    init()

    st.set_page_config(page_title="Förderrichtlinien-Assistent", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header(":books: Förderrichtlinien-Assistent ")
    user_input = st.chat_input("Stellen Sie Ihre Frage hier")
    if user_input:
        with st.spinner("Führe Anfrage aus ..."):        
            handle_user_input(user_input)


    with st.sidebar:
        st.subheader("Konfiguration")
        openai_key=st.text_input("OpenAI API Key")
        st.subheader("Förderrichtlinien")
        pdf_docs=st.file_uploader("Dokumente hier hochladen", accept_multiple_files=True)
        if st.button("Hochladen"):
            with st.spinner("Analysiere Dokumente ..."):
                init_openai_components(openai_key)
                #pdf_docs.name
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #get text chunks
                text_chunks = get_text_chunks(raw_text)
                #put to vectorstore
                vectorstore = get_vectorstore(text_chunks)
                #create conversation chain
                st.session_state.conversation = get_conversation(vectorstore) 

        #st.write('<div style="position: fixed;bottom:16px;left:16px;"><a href="https://www.flaticon.com/free-icons/bot" title="bot icons">Bot icons created by Freepik - Flaticon</a></div>',unsafe_allow_html=True)


if __name__ == "__main__":
    main()