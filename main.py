import streamlit as st
from langchain_helper import get_qa_chain 

st.title("FAQ")


st.session_state['chain'] = get_qa_chain()
   

chain = st.session_state['chain']

question = st.text_input('Question: ')

if question:
   
    response = chain(question)  # Direct call to chain
    
    response = chain.run({"query": question})
    st.header("Answers")
    st.write(response)
