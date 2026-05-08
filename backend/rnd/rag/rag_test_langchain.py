from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader

# Load file
loader = TextLoader("context.txt")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)

# Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Store in vector DB
db = Chroma.from_documents(docs, embeddings)

# Search relevant context
query = "What is the login API?"

results = db.similarity_search(query, k=3)

context = "\n".join([doc.page_content for doc in results])

# Ask Ollama
llm = Ollama(model="llama3")

prompt = f"""
Only answer using this context.

Context:
{context}

Question:
{query}

If the answer is not in the context, say you don't know.
"""

response = llm.invoke(prompt)

print(response)
