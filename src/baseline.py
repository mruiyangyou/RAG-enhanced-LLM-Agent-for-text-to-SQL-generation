from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from utils import *

# llm
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# template
template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:""" 

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", template),
    ]
)

# baseline model
sql_chain = (
    RunnableParallel({
        'schema': itemgetter('schema'),
        'question':itemgetter('question')
    })
    | prompt
    | llm.bind(stop=["\nSQLResult:"]) 
    | StrOutputParser()
).with_config({"tags": ["baseline"]})
