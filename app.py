import gradio as gr
from marco_chat import MarcAureleChatbot
from rag import VectorDatabase
from sentence_transformers import SentenceTransformer

# filepath: /home/tguyot/PersonalCode/ragosophy/app.py

# Initialize chatbot
vector_db = VectorDatabase(embedding_model=SentenceTransformer("dangvantuan/sentence-camembert-base"))
chatbot = MarcAureleChatbot(vector_db)

def answer_question(question, history):
    """Process user question and return answer."""
    answer = chatbot.answer_question(question, keep_in_history=False)
    return answer

# Create Gradio interface
with gr.Blocks(title="Marc Aurèle Chatbot") as demo:
    gr.Markdown("# Marc Aurèle Chatbot")
    gr.Markdown("Ask questions about Marcus Aurelius and his Meditations")
    
    with gr.Row():
        with gr.Column():
            chatbot_interface = gr.ChatInterface(
                answer_question,
                examples=[
                    "Quelle est la nature de la vertu selon Marc Aurèle ?",
                    "Comment gérer la colère selon Marc Aurèle ?",
                    "Qu'est-ce que le contrôle personnel pour un stoïcien ?"
                ],
                title="Chat with Marc Aurèle",
                description="Discuss philosophy and the Meditations"
            )

if __name__ == "__main__":
    demo.launch()