# # Understanding React Agentic RAG

# #**ReAct** (Reasoning + Acting) is an AI methodology that combines:
# - **Reasoning**: The AI thinks about problems step-by-step
# - **Acting**: The AI takes actions based on its reasoning

# **RAG** (Retrieval-Augmented Generation) enhances the AI by:
# 1. Retrieving relevant information from documents
# 2. Augmenting the AI's knowledge with this information
# 3. Generating better responses based on retrieved content

# This application demonstrates how these technologies work together:
# - The chatbot reasons about your questions
# - It retrieves information from a database about MSMEs in Nigeria
# - It can also search the web for up-to-date information
# - It combines all these sources to provide informed answers



import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_chroma import Chroma
import pandas as pd
from uuid import uuid4
import os
from langchain_together import TogetherEmbeddings, ChatTogether
import pathlib

# Load API keys from environment file
load_dotenv('credentials.env')
together_api = os.getenv("TOGETHER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Path for Chroma DB
chroma_dir = pathlib.Path().resolve()
chroma_path = f"{chroma_dir}\\chroma_store"

# Set up the page
st.title("React Agentic RAG Chatbot with Web Search Capabilities")


msme = pd.read_csv("msme.csv")
documents = []
metadatas = []
ids = []

for index, row in msme.iterrows():
    texts = str(row["Content"])
    title = row["Title"]
    sources = row["Sources"]
    content = texts
    metadata = {"Source": sources, "doc_title": title}
    id = f"{title}-{uuid4()}"
    documents.append(content)
    metadatas.append(metadata)
    ids.append(id)

with st.spinner("..."):
    embeddings = TogetherEmbeddings(
        model="togethercomputer/m2-bert-80M-8k-retrieval",
        api_key=together_api
    )
    
    # Initialize vector store
    Agent_msmevdb = Chroma(
        collection_name="MSME",
        persist_directory=chroma_path,
        embedding_function=embeddings
    )
    

    Agent_msmevdb.add_texts(
        texts=documents,
        metadatas=metadatas,
        ids=ids
    )

    # Create retriever tool
    retriever_tool = create_retriever_tool(
        Agent_msmevdb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}),
        "retrieve_msme_docs",
        "Search and return documents about MSMEs in Nigeria"
    )
    
    # Create web search function
    def search_web(query: str) -> str:
        """Search for users query and answer user question"""
        tool = TavilySearchResults(
            max_results=2,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=False
        )
        context = tool.invoke(query)
        return context
    
    # Create tools list
    tools = [search_web, retriever_tool]
    
    # Initialize language model
    model = ChatTogether(
        model="Qwen/Qwen2.5-7B-Instruct-Turbo", 
        api_key=together_api
    )
    
    # Define agent prompt
    React_agent_prompt = """
    You are an intelligent assistant equipped with two tools:
    - Use MSME retriever for Nigeria-specific business insights
    - Web Search Tool: use the Web search to look for up-to-date information.
    Provide answers based on the retrieved document context and 
    Cite specific sources or links from the retrieved documents
    * Provide clear, structured information from web search results
    """
    
    graph = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=React_agent_prompt,
    )
    
    st.session_state.agent = graph


# Streamlit App
st.write("AI Assistant for MSMEs in Nigeria and General Information")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_query = st.text_input("Enter your question:")

if st.button("Ask") and user_query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    try:
        # Run the agent
        with st.spinner("Processing your query..."):
            with st.chat_message("assistant"):
                inputs = {"messages": [("user", user_query)]}
                response = graph.invoke(inputs)
                
                # Extract content based on response type
                last_message = response["messages"][-1]
                
                if hasattr(last_message, "content"):
                    # Handle AIMessage object
                    final_response = last_message.content
                elif isinstance(last_message, tuple) and len(last_message) >= 2:
                    # Handle tuple format
                    final_response = last_message[1]
                else:
                    # Fallback
                    final_response = str(last_message)
                
                # Display and store the response
                st.write(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")