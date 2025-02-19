from src.helper import load_pdf_file, text_split, download_hugging_face_embidding
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_text = load_pdf_file(data = 'Data/')
chunked_text = text_split(extracted_text)
embedding = download_hugging_face_embidding()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medibot"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)


docsearch = PineconeVectorStore.from_documents(
    documents = chunked_text,
    index_name = index_name,
    embedding= embedding,
)