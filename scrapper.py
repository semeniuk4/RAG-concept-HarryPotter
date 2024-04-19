from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from bs4 import BeautifulSoup
import requests
from langchain.embeddings import HuggingFaceInstructEmbeddings


url = 'https://harrypotter.fandom.com/wiki/Special:AllPages'
response_general_page = requests.get(url)

soup = BeautifulSoup(response_general_page.text, 'html.parser')

all_pages_div = soup.find('div', class_='mw-allpages-body')
links = [f"https://harrypotter.fandom.com{link.get('href')}"  for link in all_pages_div.find_all('a') if link.get('href')]


loaders = UnstructuredURLLoader(urls=links)
data = loaders.load() 

text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=500)
docs = text_splitter.split_documents(data)

intsructor_embedding = HuggingFaceInstructEmbeddings()
vectorstore_openai = FAISS.from_documents(docs, intsructor_embedding)

vectorstore_openai.save_local("vectorstore")

