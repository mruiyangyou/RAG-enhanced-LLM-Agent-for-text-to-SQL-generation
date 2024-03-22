from utils import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from schema_extraction import *
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pinconedb import Pineconedb
import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_pinecone import PineconeVectorStore


# define template and needed csv
template = """
You are an assistant for giving text to sql task.
Based on the table schema, relevant SQL documentation related to the question where you can learnt some functions, relevant example pairs of question-answer for SQL below
- database schema: {schema} 
- sql documentation: {docs} 
- sql simdilar examples: {examples} 

Write a SQL query that would answer the user's question.

Question: {question} 
SQL Query:
"""

common_schema_related_toks = ['student', 'course', 'department', 'age', 'course', 'ids', 'car', 'player', 'class', 'cities', 'member', 'employee']
docs = pd.read_csv('../data/sql-meterial/md_data.csv')


# retriever - example csv
embedd = OpenAIEmbeddings(model = 'text-embedding-ada-002')
pc = Pineconedb(os.getenv('PINECONE_API_KEY'))
example_index = pc.load_pinecone_index('sql-sample-rag-test')
example_vectordb = pc.create_pinecone_vectordb_from_index(example_index, embedd, 'masked_question')
#example_vectordb = PineconeVectorStore.from_existing_index(index_name='sql-sample-rag-test', embedding=OpenAIEmbeddings(),  text_key='masked_question')
example_retriever = pc.create_pinecone_retriever(example_vectordb, 'similarity', {'k': 10})
example_retriever_join = pc.create_pinecone_retriever(example_vectordb, 'similarity', {'k': 10, 'filter': {'join_involved':True}})
example_retriever_no_join = pc.create_pinecone_retriever(example_vectordb, 'similarity', {'k': 10, 'filter': {'join_involved':False}})

# Cohere 
compressor = CohereRerank()
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=example_retriever
)
rerank_retriever_join = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=example_retriever_join
)
rerank_retriever_no_join = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=example_retriever_no_join
)


# define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("human", template),
    ]
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

## RAG
rag_chain = (
    RunnableParallel({
        # 'schema': lambda x: get_useful_schema({'question': x['question'], 'schema': format_schema(x['schema'])}),
        'schema': itemgetter('schema'),
        'examples': itemgetter('masked') | example_retriever | RunnableLambda(lambda x: summarise_qa_from_result(x)),
        'docs': itemgetter('masked') | example_retriever | RunnableLambda(lambda x: search_in_document(docs, summarise_keywords_from_result(x))),
        'question':itemgetter('question')
    })
    | prompt 
    | llm.bind(stop=["\nSQLResult:"]) | StrOutputParser()
    
).with_config({"tags": ["rag-base"]})


## RAG-Rerank
rerank_chain = (
    RunnableParallel({
        'schema': itemgetter('schema'),
        'examples': itemgetter('masked') | rerank_retriever | RunnableLambda(lambda x: summarise_qa_from_result(x)),
        'docs': itemgetter('masked') | rerank_retriever | RunnableLambda(lambda x: search_in_document(docs, summarise_keywords_from_result(x))),
        'question':itemgetter('question')
    })
    | prompt 
    | llm.bind(stop=["\nSQLResult:"]) | StrOutputParser()
    
).with_config({"tags": ["rag-rerank"]})


## RAG - upgraded
# sql_chain = (
#     RunnableParallel({
#         # 'schema': lambda x: get_useful_schema({'question': x['question'], 'schema': format_schema(x['schema'])}),
#         'schema': itemgetter('schema'),
#         'examples': itemgetter('masked') | rerank_retriever | RunnableLambda(lambda x: summarise_qa_from_result(x)),
#         'docs': itemgetter('masked') | rerank_retriever | RunnableLambda(lambda x: search_in_document(docs, summarise_keywords_from_result(x))),
#         'question':itemgetter('question')
#     })
#     | prompt 
#     | llm.bind(stop=["\nSQLResult:"]) | StrOutputParser()
    
# ).with_config({"tags": ["rag-rerank", "schema_extration"]})

