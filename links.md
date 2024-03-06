# Linksammlung

ChatPDF-Beispiel: https://github.com/Anil-matcha/ChatPDF/blob/main/pdfquery.py

## Beispiel für Mehrfach-Import von PDFs
Youtube-Video: https://www.youtube.com/watch?v=dXxQ0LR-3Hg&t=334s 
Sourcecode: https://github.com/alejandro-ao/ask-multiple-pdfs/blob/main/app.py

## Beispiel mit abgespeichertem Vectorstore: 
Youtube-Video: https://www.youtube.com/watch?v=RIWbalZ7sTo&t=38s
Sourcecode: https://pastebin.com/mcHG4cY4

## Streamlit
- Streamlit-Doku: 
- Icons: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
- CallbackHandler verwenden: https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/chat_with_documents.py
- Neue Stream-Komponente in Streamlit: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming

## Langchain
- Doku allgemein: https://python.langchain.com/docs
- ChatModels: https://python.langchain.com/docs/integrations/chat/
- Chains allgemein: https://python.langchain.com/docs/modules/chains/
- ConversationalRetrievalChain: https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html#

- Cookbook: https://python.langchain.com/cookbook

## PDF-Loader
https://unstructured.io/

## Textsplitter
- Beschreibung verschiedener Textsplitter: https://medium.com/@cronozzz.rocks/splitting-large-documents-text-splitters-langchain-7c7bfa899267
- Satzweises Splitten: https://pypi.org/project/textsplitter/

## Embedding
- Alternatives Embedding-Model: https://huggingface.co/hkunlp/instructor-xl

## Sentence Transformers
- Splitting, Embedding, Encoding:  https://www.sbert.net/

## Meta-Data
- relevante Infos: Seitenzahlen, Dokumenttitel, Dokumentzusammenfassung?
- Metadaten erstellen: https://python.langchain.com/docs/integrations/document_transformers/openai_metadata_tagger

### Retriever
- Was ist Retrieval? https://python.langchain.com/docs/modules/data_connection/
- ParentDocumentRetriever: https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever
- Child vs. ParentChildRetriever: https://github.com/mneedham/LearnDataWithMark/blob/main/parent-child-retriever/notebooks/Child-vs-Parent-Child-Retriever-Tutorial.ipynb

## Small-To-Big-Retrieval
- One-Sentence-Child-Chunks mit größerem Parent-Chunk. Kleiner Chunk für Embedding und Suche, großer Chunk für LLM-Synthese.

## Selfquerying
https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/

## Gut zum abgucken
PDF splitten mit Metadaten:
https://github.com/mmz-001/knowledge_gpt/tree/main

Textsummarization mit summarizationchain:
https://github.com/dataprofessor/langchain-text-summarization/tree/master

## How to build a good dashboard
https://blog.streamlit.io/crafting-a-dashboard-app-in-python-using-streamlit/?utm_campaign=Streamlit%20Releases&utm_medium=email&_hsmi=292874685&_hsenc=p2ANqtz-_WcnZ7GT9tnOHnKiyAfIUIB_jAnGFedBbj6VXJJg2lkwXF9W4unqCTy4UbaabfgAPbKfung2UFzkTegArc619yhF5Vdg&utm_content=292874685&utm_source=hs_email#what%E2%80%99s-inside-the-dashboard
