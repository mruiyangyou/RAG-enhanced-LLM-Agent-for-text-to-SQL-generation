from utils import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
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
# template = """
# You are an assistant for giving text to sql task.
# Based on the database schema, relevant SQL documentation related to the question for syntax and function guidance, relevant example pairs of question-answer for more insights:
# - Database schema: {schema} 
# - SQL documentation: \n{docs} 
# - SQL simdilar examples: \n{examples} 

# Write a SQL query that would answer the user's question.

# Question: {question} 
# SQL Query:
# """

template = """
You are an assistant for giving text to sql task.
Based on the table schema, relevant SQL documentation related to the question where you can learnt some functions, relevant example pairs of question-answer for SQL below
- database schema: {schema} 
- sql documentation: \n{docs} 
- sql simdilar examples: \n{examples} 

Write a SQL query that would answer the user's question. Just return the SQL query with no other information.

Question: {question} 
SQL Query:
"""


unfamilair_template = """
Given the question {question}, return me the unfamiliar terms and their one sentence explanations in this format: 

1.  term 1: explanation one 
2.  term 2: explanation two 
"""
unfamilair_prompt = ChatPromptTemplate.from_template(unfamilair_template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
unfamilair_chain = unfamilair_prompt | llm | StrOutputParser()


common_schema_related_toks = ['student', 'course', 'department', 'age', 'course', 'ids', 'car', 'player', 'class', 'cities', 'member', 'employee']
docs = pd.read_csv('../data/sql-meterial/md_data.csv')


# retriever - example csv
embedd = OpenAIEmbeddings(model = 'text-embedding-ada-002')
pc = Pineconedb(os.getenv('PINECONE_API_KEY'))
example_index = pc.load_pinecone_index('sql-sample-rag-test')
# example_vectordb = pc.create_pinecone_vectordb_from_index(example_index, embedd, 'masked_question')
example_vectordb = PineconeVectorStore.from_existing_index(index_name='sql-sample-rag-test', embedding=embedd,  text_key='masked_question')
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


#define prompt
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


## RAG-chain of thought
upgrade_chain = (
    RunnableParallel({
        # 'schema': lambda x: get_useful_schema({'question': x['question'], 'schema': format_schema(x['schema'])}),
        'schema': itemgetter('schema'),
        'unfamiliar': RunnablePassthrough() | unfamilair_chain,
        'examples': itemgetter('masked') | example_retriever | RunnableLambda(lambda x: summarise_qa_from_result(x)),
        'docs': itemgetter('masked') | example_retriever | RunnableLambda(lambda x: search_in_document(docs, summarise_keywords_from_result(x))),
        'question':itemgetter('question')
    })
    | prompt 
    | llm.bind(stop=["\nSQLResult:"]) | StrOutputParser()
    
).with_config({"tags": ["chain-of-thought"]})

# df = pd.read_csv("../data/sql_test_suite_academic/masked_academic_sample.csv")
# q, m, schema = df.loc[1, 'question'], df.loc[1, 'masked'], df.loc[1, 'schema']
# inputs = {'question': q, 'masked': m, 'schema': schema}