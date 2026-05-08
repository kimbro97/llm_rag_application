from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

loader = Docx2txtLoader('./tax.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)

embeddings = UpstageEmbeddings(model="embedding-query")

database = Chroma.from_documents(documents=document_list, embedding=embeddings)

query = "연봉 5천만인 직장의 소득세는 얼마인가요?"

retrieved_docs = database.similarity_search(query=query)
print(retrieved_docs)

