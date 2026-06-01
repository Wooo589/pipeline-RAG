import ollama
from datasets import load_dataset

# Dataset do HugFace
dataset = load_dataset("TucanoBR/wikipedia-PT")

# Banco de dados vetoriais
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Cada elemento em VECTOR_DB vai ser uma tupla de (chunk, embedding)
VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

# Implementação da função de captura
def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]

  # Lista temporária para armazenar pares de (chunks e similaridade)
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))

  # Ordenar por similaridade em ordem descrescente
  similarities.sort(key=lambda x: x[1], reverse=True)
  
  # Retorne os chunks mais relevantes
  return similarities[:top_n]

# Geração de resultado
input_query = input('Pergunte-me alguma coisa: ')
retrieved_knowledge = retrieve(input_query)

print('Conhecimento extraído:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similaridade: {similarity:.2f}) {chunk}')

# Manter o prompt em inglês, pois em português parece que o modelo alucinou
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# Exibir a resposta do chatbot
print('Resposta do chatbot:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)