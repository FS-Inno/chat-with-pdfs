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
from streaming import StreamHandler
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import PyPDFDirectoryLoader


gpt_model = 'gpt-4-1106-preview'
embedding_model = 'text-embedding-3-small'


def init():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []        
    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(temperature=0, model_name=gpt_model, streaming=True)
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(model=embedding_model) 


def get_pdf_text():
    loader = PyPDFDirectoryLoader("samples/")
    docs = loader.load_and_split()
    return docs


def get_text_chunks(text):
    textsplitter=CharacterTextSplitter(separator="\n",chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks=textsplitter.split_documents(text)
    #for chunk in chunks:
       # chunk.metadata['source'] = 'test123'
    return chunks


def get_vectorstore(chunks):
    vectorstore = FAISS.from_documents(documents=chunks,embedding=st.session_state.embeddings)
    return vectorstore



def get_conversation(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True,input_key="question", output_key="answer")
    conversation_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=st.session_state.llm,
        retriever=vectorstore.as_retriever(),
        memory = memory    
    )
    return conversation_chain

def handle_user_input(question):
	
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
                           
    with st.chat_message("user"):
        st.write(question)
        
    st_cb = StreamHandler(st.chat_message("assistant").empty())
    print(st_cb)
    response = st.session_state.conversation({'question':question},callbacks=[st_cb])
    st.session_state.chat_history = response['chat_history']
	



def main():
    load_dotenv()
    init()

    st.set_page_config(page_title="Förderrichtlinien-Assistent", page_icon=":books:")

    st.header(":books: Förderrichtlinien-Assistent ")
    user_input = st.chat_input("Stellen Sie Ihre Frage hier")
    if user_input:
        with st.spinner("Führe Anfrage aus ..."):        
            handle_user_input(user_input)


    with st.sidebar:
        st.subheader("Förderrichtlinien")
        if st.button("Analysieren"):
            with st.spinner("Analysiere Dokumente ..."):
                raw_text = get_pdf_text()
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation(vectorstore) 


if __name__ == "__main__":
    main()