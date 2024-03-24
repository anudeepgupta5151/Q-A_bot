from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Palm LLM model
llm_palm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# # Initialize instructor embeddings using the Hugging Face model
hf = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", encode_kwargs={'normalize_embeddings': True})
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='sample_faqs.csv', source_column="prompt")
    faq_data = loader.load()

    # Create a FAISS instance for vector database from 'faq_data'
    faiss_vectordb = FAISS.from_documents(documents=faq_data,
                                    embedding=hf)

    # Save vector database locally
    faiss_vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    faiss_vectordb = FAISS.load_local(vectordb_file_path, hf)

    # Create a faiss_retriever for querying the vector database
    faiss_retriever = faiss_vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """provided the following context and a question, generate an answer based on the given context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly reply back "I couldnot find the answer please try contacting customer support 0123456789." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm_palm,
                                        chain_type="stuff",
                                        retriever=faiss_retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("How to place an order?"))
