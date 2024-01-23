# Chat-With-PDFs
Ein einfacher Demonstrator zum Chatten mit PDF-Dateien. Nach Eingabe eines OpenAI-API-Keys zur Verwendung von GPT-4-Turbo und dem Hochladen eines PDF ist es möglich den Chatbot zu Inhalten des PDF zu befragen.

## 0. Voraussetzungen
* ein API-Key von [OpenAI](https://www.openai.com) (siehe auch: https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key)
* die Berechtigung Python auf dem Zielrechner zu installieren (wenn nicht vorhanden) Empfehlung Python-Version > 3.10

Zur erfolgreichen Installation sind die folgenden Schritte auszuführen:

## 1. Python installieren
Ein aktuelle Python-Version von https://www.python.org/downloads/ herunterladen und installieren. GUI, IDEs und Debugging-Erweiterungen können weggelassen werden. Wichtig ist, dass Python zum Umgebungspfad hinzugefügt wird!

## 2. Repository clonen
Mit 

    git clone https://github.com/ag-jil/chat-with-pdfs.git

wird das Repo von Github gecloned. Danach in das Verzeichnis wechseln!

## 3. Virtuelles Environment erzeugen und aktivieren
Um in Python Anwendungen voneinander zu separieren, gibt es virtuelle Umgebungen.
Zum Erzeugen der Umgebung `venv` wird

    python -m venv venv
  
ausgeführt.
Mit 
  
     venv\Scripts\activate

wird die virtuelle Umgebung aktiviert. Vor dem Prompt sollte dann `(venv)´ stehen.   

## 4. Abhängigkeiten installieren
Für den Chatbot werden die Bibliotheken `streamlit` und `langchain` sowie ihre Abhängigkeiten benötigt.
Diese können mit

    pip install -r requirements.txt
  
installiert werden.

Anmerkungen: 
1. pip ist der Paketmanager in Python. 
2. Die Pakete werden in die virtuelle Umgebung `venv` installiert. Sie stehen auch nur zur Verfügung, wenn die Umgebung aktiviert wurde!
3. streamlit ist eine Bibliothek zum UI bauen.
4. Langchain ist eine Abstraktion für die Arbeit mit LLMs. In unserem Fall ruft Langchain im Hintergrund die API von OpenAI auf.

## 5. Anwendung starten
Wenn noch nicht passiert, muss die virtuelle Umgebung aktiviert werden. (Muss bei jedem Neustart des Terminals/der Kommandozeile erfolgen!)
  
     venv\Scripts\activate

Danach die Anwendung starten:

    streamlit run app.py
  
Wenn der Start erfolgreich war, sieht man in der Kommandozeile die Meldungen des Webservers und es wird ein Browserfenster mit der Anwendung geöffnet.
