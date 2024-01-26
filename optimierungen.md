# Mögliche Optimierungen
## Small-To-Big-Retrieval
One-Sentence-Child-Chunks mit größerem Parent-Chunk. Kleiner Chunk für Embedding und Suche, großer Chunk für LLM-Synthese.

## Meta-Data
- Seitenzahlen, Dokumenttitel, Dokumentzusammenfassung
- Metadaten erstellen: https://python.langchain.com/docs/integrations/document_transformers/openai_metadata_tagger
 
## Systemmessage
-  Allgemeine Beschreibung

## Textsplitting
- Satzweises Splitten: https://pypi.org/project/textsplitter/

## Selfquerying
https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/