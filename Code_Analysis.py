import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
 
#load_dotenv
#os.environ['OPENAI_API_KEY'] = ''
# Clone
repo_path = "."
# repo = Repo.clone_from("https://github.com/langchain-ai/langchain", to_path=repo_path)
# Load
loader = GenericLoader.from_filesystem(
    repo_path + ".",
    glob="**/*",
    suffixes=[".java"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.JAVA, parser_threshold=500),
)
documents = loader.load()
len(documents)
 
 
 
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
print(texts[0])
 
 
 
db = FAISS.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)
 
 
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
#llm = {"model": "gpt-4-1106-preview"}
#chain = configurable_runnable.with_config({"configurable": {"llm": llm}})chain.invoke({})

memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
 
questions = [
    "was passiert in der POST response?",
    "du bist beauftragt worden die software zu testen.schreibe einen Junit5 test der das Programm bezüglich der umwandlung von | zu &  testet.",
    "schreibe das programm bitte nach python",
]
 
for question in questions:
    result = qa(question)
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")