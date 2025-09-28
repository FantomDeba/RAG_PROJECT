# Import necessary libraries
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found. Make sure it's in your .env file.")

# Configure the model
generation_config = {"temperature": 0.75}
genai.configure(api_key=google_api_key)

# Vector Embedding → HuggingFace (free, no quota issues)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_vectordb():
    # Load data from CSV file (⚠️ check your column names!)
    loader = CSVLoader(file_path='Ecommerce_FAQs.csv', encoding='cp1252')
    documents = loader.load()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory='./ChromaDB'
    )
    vectordb.persist()

def get_response(query):
    vectordb = Chroma(
        persist_directory="ChromaDB",
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever()

    # Prompt template
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Chat model → Gemini for answering
    chat = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",   # or "models/gemini-1.5-pro"
    google_api_key=google_api_key,
    temperature=0.3)


    chain = load_qa_chain(chat, chain_type="stuff", prompt=PROMPT)

    # Get response
    response = chain.invoke(
        {"input_documents": retriever.get_relevant_documents(query), "question": query},
        return_only_outputs=True
    )['output_text']

    return response

if __name__ == '__main__':
    # First run: uncomment this line to build ChromaDB
    # create_vectordb()

    print(get_response("shipping duration?"))
