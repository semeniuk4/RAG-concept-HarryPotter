from langchain.llms import OpenAIChat
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA, ConversationChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

llm = OpenAIChat(model='gpt-3.5-turbo', temperature=0.8)

instructor_embedding = HuggingFaceInstructEmbeddings()
vector_db_file_path = 'vectorstore'

memory = ConversationBufferMemory(memory_key="history", return_messages=True)
def get_qa_chain():
    prompt_template = """Given the following context and a question, generate an answer based on this context or on the conversation history.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    CONVERSATION HISTORY: {history}

    QUERY: {question}

    """
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question"])
    vectordb = FAISS.load_local(vector_db_file_path, instructor_embedding)
    
    retriever = vectordb.as_retriever(score_threshold=0.7)


    chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": PROMPT,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
    )


    return chain

if __name__ == "__main__":
    chain = get_qa_chain()
    
    while True:
        query = input("Question: ")
        result = chain.run({"query": query})
        response = result
        print(response)
        # chat_history.append(question, response)
