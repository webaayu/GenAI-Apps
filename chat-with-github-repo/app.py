#Running:) create github token to access repo 
import streamlit as st
import os
from github import Github
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Function to fetch repository data from GitHub
def fetch_github_repo_data(git_repo, github_token):
    try:
        g = Github(github_token)
        repo = g.get_repo(git_repo)
        contents = repo.get_contents("")
        repo_data = ""
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            else:
                file_data = repo.get_contents(file_content.path).decoded_content
                try:
                    text = file_data.decode("utf-8")
                    repo_data += f"\n\nFile: {file_content.path}\n{text}"
                except UnicodeDecodeError:
                    # Skip non-text files
                    continue
        return repo_data
    except Exception as e:
        st.error(f"Error fetching GitHub repository data: {e}")
        return None

# Function to perform RAG using MistralAI and Chroma
def perform_rag(repo_data, prompt):
    try:
        if repo_data:
            # Create embeddings
            embeddings = HuggingFaceEmbeddings()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.create_documents([repo_data])

            # Store chunks in ChromaDB
            persist_directory = 'github_repo_embeddings'
            vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
            vectordb.persist()  # Persist ChromaDB
            
            # Load persisted Chroma database
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            
            # Perform retrieval using Chroma
            docs = vectordb.similarity_search(prompt)
            if docs:
                text = docs[0].page_content
            else:
                st.warning("No relevant documents found.")
                return None
            
            # Perform generation using MistralAI
            client = MistralClient(api_key=mistral_api_key)
            messages = [
                ChatMessage(role="system", content="You are an assistant to help with GitHub repository queries."),
                ChatMessage(role="user", content=f"Context: {text}\n\nQuestion: {prompt}\n\nAnswer:")
            ]
            response = client.chat(model="mistral-large-latest", messages=messages)
            
            return response.choices[0].message.content
        
        else:
            st.warning("No repository data found or error occurred.")
            return None
        
    except Exception as e:
        st.error(f"Error performing RAG: {e}")
        return None

# Streamlit application
def main():
    st.title("Chat with GitHub Repository 💬")
    st.caption("This app allows you to chat with a GitHub Repo using MistralAI and ChromaDB")

    # Get the GitHub token from the user
    github_token = st.text_input("Enter your GitHub Token", type="password")

    # Get the GitHub repository from the user
    git_repo = st.text_input("Enter the GitHub Repo (owner/repo)", type="default")

    # Add the GitHub data to the knowledge base if the GitHub token is provided
    if github_token and git_repo:
        # Fetch GitHub repository data
        repo_data = fetch_github_repo_data(git_repo, github_token)

        if repo_data:
            st.success(f"Added {git_repo} to knowledge base!")

            # Ask a question about the repository
            prompt = st.text_input("Ask any question about the GitHub Repo")

            # Chat with the repository
            if prompt:
                answer = perform_rag(repo_data, prompt)
                if answer:
                    st.subheader("Generated Answer:")
                    st.write(answer)
        else:
            st.error(f"Failed to fetch data for {git_repo}. Please check the repository name and your token's permissions.")

if __name__ == "__main__":
    main()
