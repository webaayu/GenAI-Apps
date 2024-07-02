#Running:)
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
import tempfile
import streamlit as st
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
import base64
import pytz  # Import pytz library for timezone handling

# Load environment variables
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Function to fetch Gmail emails based on a filter
def fetch_gmail_emails(selected_subject, timezone):
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('gmail', 'v1', credentials=creds)
        
        # Calculate the datetime 24 hours ago in the user's timezone
        now = datetime.now(pytz.timezone(timezone))
        twenty_four_hours_ago = now - timedelta(hours=24)
        twenty_four_hours_ago_str = twenty_four_hours_ago.strftime('%Y/%m/%d')
        
        # Construct the query for Gmail API
        query = f"after:{twenty_four_hours_ago_str}"
        if selected_subject:
            query += f" subject:'{selected_subject}'"

        st.write(f"Debug: Query constructed: {query}")  # Debug output

        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q=query).execute()
        messages = results.get('messages', [])

        email_texts = []
        for message in messages[:10]:  # Limiting to first 10 emails for brevity
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            payload = msg['payload']
            headers = payload.get('headers', [])
            subject = None
            date = None
            for header in headers:
                if header.get('name') == 'Subject':
                    subject = header.get('value')
                if header.get('name') == 'Date':
                    date = header.get('value')
            if not subject or selected_subject not in subject:
                continue  # Skip emails that do not match the subject filter

            parts = payload.get('parts', [])
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    email_texts.append(f"Subject: {subject}\nDate: {date}\n\n{body}")

        if not email_texts:
            st.warning("No emails found matching the selected subject and timeframe.")
        
        return email_texts
    
    except Exception as e:
        st.error(f"Error fetching Gmail emails: {e}")
        return None

# Function to perform RAG using MistralAI and Chroma
def perform_rag(email_text, prompt):
    try:
        if email_text:
            # Create embeddings
            embeddings = HuggingFaceEmbeddings()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.create_documents([email_text])

            # Store chunks in ChromaDB
            persist_directory = 'gmail_summary_embeddings'
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
                ChatMessage(role="system", content="You are an assistant to help with email queries."),
                ChatMessage(role="user", content=f"Context: {text}\n\nQuestion: {prompt}\n\nAnswer:")
            ]
            response = client.chat(model="mistral-large-latest", messages=messages)
            
            return response.choices[0].message.content
        
        else:
            st.warning("No emails found or error occurred.")
            return None
        
    except Exception as e:
        st.error(f"Error performing RAG: {e}")
        return None

# Streamlit application
def main():
    st.title("Chat with your Gmail Inbox 📧")
    st.caption("This app allows you to chat with your Gmail inbox using MistralAI and ChromaDB")
    # Timezone input
    timezone = st.selectbox("Select your timezone", pytz.all_timezones, index=pytz.all_timezones.index("Asia/Kolkata"))
    # Subject choices for selectbox
    subject_choices = ["[eCHO News] ","Weekly Updates on Monitoring List 1: cloudyuga.guru, collabnix.com", "New Order", "Never miss an important message"]
    selected_subject = st.selectbox("Select a subject from your emails", subject_choices)
    # Fetch Gmail emails based on selected subject
    if selected_subject:
        email_text = fetch_gmail_emails(selected_subject,timezone)
        if email_text:
            #st.subheader("Email Content:")
            #st.write(email_text)
            text=str(email_text)
            # Button to generate summary
            if st.button("Generate Summary"):
                prompt="you are perfect in writting summary"
                summary = perform_rag(text,prompt)
                if summary:
                    st.subheader("Generated Summary:")
                    st.write(summary)
if __name__ == "__main__":
    main()
