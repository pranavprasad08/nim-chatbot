import os
import json
import chromadb
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_core.documents import Document

class VectorDatabase:
    def __init__(self, persist_directory="./chroma_db", json_folder="path/to/json/files"):
        """Initializes ChromaDB, NVIDIA Embeddings, and sets the folder containing JSON chunk files."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="document_embeddings")
        self.embedder = NVIDIAEmbeddings(
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            api_key="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",
            truncate="NONE"
        )
        self.reranker = NVIDIARerank(
            model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
            api_key="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",
        )
        self.json_folder = json_folder

    def is_document_indexed(self, filename):
        """Checks if a document's chunks are already indexed in ChromaDB."""
        results = self.collection.get(where={"filename": filename})
        return len(results["ids"]) > 0

    def embed_texts(self, texts):
        """Generates embeddings for a list of strings."""
        return self.embedder.embed_documents(texts)

    def embed_query(self, query):
        """Generates an embedding for a single query string."""
        return self.embedder.embed_query(query)

    def add_texts(self, docs):
        """Adds a list of document chunks to ChromaDB if not already indexed."""
        texts = [d.page_content for d in docs]
        metadatas = [dict(d.metadata) for d in docs]
        embeddings = self.embed_texts(texts)
        ids = [str(i) for i in range(len(docs))]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def index_new_documents(self):
        """Indexes only new documents from the folder containing JSON chunk files."""
        for filename in os.listdir(self.json_folder):
            if filename.endswith(".json") and not self.is_document_indexed(filename):
                filepath = os.path.join(self.json_folder, filename)
                
                with open(filepath, "r", encoding="utf-8") as f:
                    json_data = json.load(f)  # Load JSON file
                
                if "fileContents" not in json_data:
                    print(f"Skipping {filename}: No 'fileContents' found.")
                    continue
                
                # Extract chunks and metadata
                docs = []
                for chunk in json_data["fileContents"]:
                    if "contentBody" in chunk:
                        content = chunk["contentBody"]
                        metadata = chunk.get("contentMetadata", {})
                        metadata["filename"] = filename  # Add filename to metadata
                        docs.append(Document(page_content=content, metadata=metadata))

                if docs:
                    self.add_texts(docs)
                    print(f"Indexed: {filename}")
                else:
                    print(f"Skipping {filename}: No valid content found.")

    def retrieve(self, query):
        """Retrieves the top 5 relevant document chunks."""
        query_embedding = self.embed_query(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=20)
        
        # Rerank results
        response = self.reranker.compress_documents(query=query, documents=[Document(page_content=passage) for passage in results["documents"][0]])
        
        relevant_chunks = [res.page_content for res in response[:5] if res]
        return relevant_chunks if relevant_chunks else "No relevant chunks found"
