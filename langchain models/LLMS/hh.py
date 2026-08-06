from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key or not openai_api_key.startswith("sk-") or openai_api_key.startswith("sk-proj-"):
#     raise ValueError("Please set a valid OPENAI_API_KEY environment variable. The key should start with 'sk-' but not 'sk-proj-'")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# index = np.argmax(scores)
# score = scores[index]

# print(f"Query: {query}")
# print(f"Most similar document: {documents[index]}")
# print(f"Similarity score: {score}")

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1] 

print(f"Query: {query}")
print(f"Most similar document: {documents[index]}")
print(f"Similarity score: {score}")