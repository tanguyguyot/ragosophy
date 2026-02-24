# ragosophy

A Retrieval-Augmented Generation (RAG) chatbot application that lets you converse with a chatbot that impersonates philosophers based on their texts. Currently supports Marcus Aurelius' *Meditations* (in French).

## Features

- **RAG Pipeline**: Embeds PDF documents using sentence-transformers and stores them in ChromaDB
- **Chat Interface**: Gradio-powered web UI for asking questions
- **Text-to-Speech**: Transforms responses into an elderly stoic-sounding voice
- **Prompt Injection Protection**: Uses delimiter-based defense to prevent prompt hacking

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- OpenAI API key (set in `.env` file as `OPENAI_API_KEY`)
- FFmpeg (for TTS audio processing)

## Project Structure

```
ragosophy/
├── app.py                 # Gradio web interface
├── rag.py                 # VectorDatabase and Chatbot classes
├── marco_chat.py          # Marc Aurele-specific preprocessor and chatbot
├── requirements.txt      # Python dependencies
├── documents/             # Source PDFs
│   ├── fr/marc_aurele/   # French texts (Marcus Aurelius)
├── cleaned/              # Preprocessed text data
├── chroma_storage/        # ChromaDB vector database
└── speeches/              # Generated audio files
```

## Usage

1. Set your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. Download the PDF we use the RAG on, and add it to documents folder
- for Marcus Aurelius : https://fr.wikisource.org/wiki/Pens%C3%A9es_pour_moi-m%C3%AAme in documents/fr/marc_aurele folder

3. Run the Gradio interface:
   ```bash
   python app.py
   ```

3. Open the URL shown in the terminal to chat with Marcus Aurelius.

## How It Works

1. **Preprocessing**: PDF documents are parsed and split into chunks
2. **Embedding**: Text chunks are embedded using sentence-transformers (French model for Marc Aurele)
3. **Retrieval**: Questions are embedded and relevant context is retrieved from ChromaDB
4. **Generation**: OpenAI GPT model generates responses based on retrieved context
5. **TTS**: Response is converted to speech with pitch/speed adjustments for an elderly tone

## Example Questions

- "Quelle est la nature de la vertu selon Marc Aurèle ?"
- "Comment gérer la colère selon Marc Aurèle ?"
- "Qu'est-ce que le contrôle personnel pour un stoïcien ?"


## Screenshots

![Chat Example]("images/marc aurele examples.png")
![Out of subject prompt example]("/images/luffy example.png")


## License

MIT
