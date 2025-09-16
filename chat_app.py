from langchain.chains.llm import LLMChain

import streamlit as st
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_milvus import Milvus
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
import logging
from dotenv import load_dotenv

load_dotenv()

def get_custom_prompt():
    """Create a custom prompt template for Warren Buffett-style responses"""

    template = """You are Warren Buffett, the legendary investor and CEO of Berkshire Hathaway. 
    Use the following pieces of context to answer the question in my authentic voice and style.

    Context: {context}

    Question: {question}

    Instructions for formulating the response:
    1. Draw primarily from the given context and my known investment principles
    2. Use my characteristic plain-spoken, Midwestern style
    3. Include relevant analogies and folksy wisdom where appropriate
    4. If discussing investments, emphasize long-term value investing principles
    5. Be direct and honest - if you don't know something, say so
    6. Reference specific experiences or examples from the context when relevant
    7. Maintain my ethical stance and emphasis on integrity

    If the question cannot be answered based on the context, use my general philosophy and principles,
    but clearly indicate when you are going beyond the provided context.

    Answer the question as I would, maintaining my voice and personality:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def initialize_chain():
    embeddings = OpenAIEmbeddings()

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": os.getenv("MILVUS_URI"),
                         "token": os.getenv("MILVUS_TOKEN")},
        collection_name="buffett_quotes",
        text_field="content",
        vector_field="embedding"
    )

    # Create the qa_chain first
    qa_chain = load_qa_chain(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.7,
            max_tokens=2000
        ),
        chain_type="stuff",
        prompt=get_custom_prompt()
    )

    # Create question generator
    template = ("Combine the chat history and follow up question into "
                "a standalone question. Chat History: {chat_history}"
                "Follow up question: {question}")

    prompt = PromptTemplate.from_template(template)
    question_generator = LLMChain(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.7,
            max_tokens=2000
        ),
        prompt=prompt
    )

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create conversation chain
    chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain=qa_chain,
        question_generator=question_generator,
        memory=memory,
        return_source_documents=False,
        output_key="answer"
    )

    return chain


def get_buffett_response(chain, question):
    """Process the question and generate Warren Buffett-style response"""
    try:
        # Get response from chain
        response = chain({"question": question})
        print(response)
        # Extract answer and source documents
        answer = response["answer"]
        source_docs = response.get("source_documents", [])

        # Format response with sources if available
        formatted_response = answer
        if source_docs:
            formatted_response += "\n\nThis wisdom draws from my following experiences and statements:"
            for i, doc in enumerate(source_docs, 1):
                formatted_response += f"\n{i}. {doc.page_content[:200]}..."

        return formatted_response

    except Exception as e:
        print(str(e))
        return f"Well, I seem to have encountered a technical glitch. As I always say, it's better to be honest about our limitations. Error: {str(e)}"


def main():
    st.set_page_config(page_title="Warren Buffett AI Assistant")
    st.title("Chat with Warren Buffett")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! I'm Warren Buffett. I'm here to share my investment wisdom and life experiences with you. What would you like to discuss?"}
        ]

    # Initialize the chain
    chain = initialize_chain()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    if user_input := st.chat_input("Ask Warren Buffett:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking like Warren Buffett..."):
                response = get_buffett_response(chain, user_input)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()