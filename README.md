
# Projekt Setup

## Voraussetzungen

1. **Virtuelle Umgebung erstellen**  
   Erstelle eine virtuelle Umgebung und installiere die Abhängigkeiten über die `requirements.txt`:

   ```bash
   python -m venv venv
   source venv/bin/activate  # unter Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **OpenAI API Key**  
   Für die Nutzung von GPT-4o wird ein OpenAI API Key benötigt.  
   Die Kosten betragen ca. **0,04 € für 20 Anfragen** (Stand: April 2025).

   - Erstelle im Projekt-Root eine Datei namens `.env` mit folgendem Inhalt:

     ```
     OPEN_AI_API_KEY=dein-api-key-hier
     ```

3. **MODNet-Modell herunterladen**  
   Lade das vortrainierte MODNet-Modell herunter und speichere es im Verzeichnis `checkpoints/`.  
   [Download Link](https://huggingface.co/DavG25/modnet-pretrained-models/blob/main/models/modnet_photographic_portrait_matting.ckpt)

---

## Lokaler Start

Um die Anwendung lokal zu starten, führe folgenden Befehl im Projekt-Root aus:

```bash
uvicorn app.main:app --reload
```

---

## API-Nutzung

Zum Testen und Ansprechen der API:

- Nutze die Datei `communicator.py` als Beispiel-Client:
  - Gib dort den Pfad zu einem Bild an.
  - Probiere die optionalen Parameter aus.

Alternativ:
- Analysiere den verwendeten API-Endpoint in `communicator.py`.
- Implementiere deinen eigenen Client basierend auf den Anforderungen.
