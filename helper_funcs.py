import os
from langchain import OpenAI
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()  # take environment variables from .env (especially openai api key)


def get_model(model):
    if model == 'OpenAI':
        return OpenAI(temperature=0.9, max_tokens=500)
    if model == 'Google PaLM':
        return GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.9)


def get_embeddings(model):
    if model == 'OpenAI':
        return OpenAIEmbeddings()
    if model == 'HuggingFace':
        return HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2')  # HuggingFaceInstructEmbeddings(query_instruction="Represent the query for retrieval: ")


def get_qa_chain(vectorstore, llm):
    template = """'Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
                    If you dont know the answer, just say that you dont know. Dont try to make up an answer.
                    ALWAYS return a "SOURCES" part in your answer. The "SOURCES" part should be a reference to the source of the document from which you got your answer.
                    SOURCES: 


                    QUESTION: {question}
                    =========
                    {summaries}
                    =========
                    FINAL ANSWER: """
    prompt = PromptTemplate(input_variables=['summaries', 'question'],
                            template=template)
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm,
                                                        retriever=vectorstore.as_retriever(),
                                                        chain_type_kwargs=chain_type_kwargs)
    # chain = RetrievalQAWithSourcesChain.from_llm(llm,
    #                                                     retriever=vectorstore.as_retriever(),
    #                                                     max_tokens_limit=1024)
    return chain
