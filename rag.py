import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
from openai import OpenAI

class VectorDatabase:
    
    def __init__(self, embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')):
        load_dotenv()
        self.client = chromadb.Client(Settings())
        self.embedding_model = embedding_model
        self.raw_chunks = []
        self.embedded_chunks = []
        self.collection = None

    def create_collection(self, collection_name):
        self.collection = self.client.create_collection(name=collection_name)

    def load_chunks(self, chunks):
        self.raw_chunks = chunks
        self.embedded_chunks = self._embed_chunks(chunks)

    def _embed_chunks(self, chunks):
        return self.embedding_model.encode(chunks, convert_to_tensor=True).cpu().numpy().tolist()

    def add_embedded_chunks(self, ids=None):
        self.collection.add(documents=self.raw_chunks, embeddings=self.embedded_chunks, ids=ids)

    def query_collection(self, query, n_results=5):
        if not hasattr(self, 'collection') or not isinstance(self.collection, chromadb.api.models.Collection):
            raise ValueError("Collection not initialized. Call create_collection first.")
        return self.collection.query(query_texts=[query], n_results=n_results)
    
    def retrieve_context(self, question, collection, top_k=5):
        embed_question = self.embedding_model.encode(question, convert_to_tensor=True)
        result = collection.query(
            query_embeddings=embed_question.cpu().numpy().tolist(),
            n_results=top_k)
        
        context = "\n".join(result['documents'][0])
        return context
    
class Chatbot:
    def __init__(self, 
                 vector_db: VectorDatabase, 
                 model: str = "gpt-3.5-turbo", 
                 developer_instructions: str = "You are a helpful assistant.",
                 history_limit: int = 10):
        load_dotenv()
        self.vector_db = vector_db
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.history = [
            {"role": "developer", "content": developer_instructions}
        ]
        self.history_limit = history_limit

    def clean_history(self):
        self.history = [msg for msg in self.history if msg['role'] != 'user']

    def answer_question(self, question, keep_in_history=True):
        if len(self.history) > self.history_limit:
            self.clean_history()
            print("History cleaned to low context window.")

        context = self.vector_db.retrieve_context(question, self.vector_db.collection)
        question_formatted = [{"role": "system", "content": f"Context:\n{context}"},
                                     {"role": "user", "content": question}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history + question_formatted
        )

        answer = response.choices[0].message.content

        q_and_a = question_formatted + [{"role": "assistant", "content": answer}]
        if keep_in_history:
                # add context to instructions
                self.history.extend(q_and_a)
        return answer