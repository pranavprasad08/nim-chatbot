from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.memory import ConversationBufferMemory

class ChainServer:
    def __init__(self, vector_db, model="meta/llama3-8b-instruct", base_url_nim="http://127.0.0.1:8000/v1"):
        """Initializes the ChainServer with vector search, NVIDIA NIM, and improved reasoning."""
        self.vector_db = vector_db
        self.llm = ChatNVIDIA(base_url=base_url_nim, model=model)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define tools (Agent Actions)
        self.tools = [
            Tool(
                name="Document Search",
                func=self.search_documents,
                description="Retrieves relevant chunks from documents based on a query. Provide the query as a string"
            ),
        ]

        # Initialize the LangChain Agent with enhanced CoT reasoning
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # Better for CoT reasoning
            memory=self.memory,
            max_iterations=5,  # Allow deeper reasoning
            verbose=True
        )

    def search_documents(self, query):
        """Retrieve relevant documents from ChromaDB."""
        retrieved_chunks = self.vector_db.retrieve(query)
        return "\n".join(retrieved_chunks) if retrieved_chunks else "No relevant documents found."

    def query(self, question):
        """Retrieves relevant documents and generates a response using LangChain Agent with CoT prompting."""
        prompt = (
            "Think step-by-step before answering the question. "
            "Use the tools available when needed and reason logically. "
            f"Question: {question}"
        )

        # Use LangChain Agent for structured reasoning & answer generation
        response = self.agent.invoke(prompt, return_intermediate_steps=True)

        return response if response else "❌ Failed to generate response."
