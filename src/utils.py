from langsmith import Client
import pandas as pd 
from typing import List, Dict
from langchain_community.utilities import SQLDatabase
import asyncio
from dotenv import find_dotenv, load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import pickle
import re
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from langchain.schema import Document


# _ = load_dotenv(find_dotenv())

# def connect_db(schema: str, type = 'url'):
#     database_uri = f"redshift+psycopg2://{os.environ['redshift_user']}:{os.environ['redshift_pass']}@redshift-cluster-comp0087-demo.cvliubs5oipw.eu-west-2.redshift.amazonaws.com:5439/comp0087"
#     sqldb = SQLDatabase.from_uri(database_uri=database_uri, schema = schema)
#     return sqldb

def connect_db(type = 'url', **kwargs):
    if type == 'url':
        return SQLDatabase.from_uri(**kwargs)
    elif type == 'engine':
        return SQLDatabase(**kwargs)

def format_redshift_uri():
    return f"redshift+psycopg2://{os.environ['redshift_user']}:{os.environ['redshift_pass']}@redshift-cluster-comp0087-demo.cvliubs5oipw.eu-west-2.redshift.amazonaws.com:5439/comp0087"
    
async def execute_sql_async(db: SQLDatabase, query: str, executor: ThreadPoolExecutor) -> pd.DataFrame:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, db.run, query)
    return result

async def execute_all_queries(db: SQLDatabase, question_df: pd.DataFrame, input_col_name: str, output_col_name: str) -> pd.DataFrame:
    with ThreadPoolExecutor() as executor:
        tasks = [
            execute_sql_async(db, query, executor)
            for query in question_df[input_col_name]
        ]
        results = await asyncio.gather(*tasks)
    # Assuming each result is a DataFrame, concatenate them
    question_df[output_col_name] = results
    return question_df

def load_data_langsmith(df: pd.DataFrame, dataset_name: str,
                        desciption: str, answer: bool = False) -> None:
    client = Client()
    
    dataset = client.create_dataset(
    dataset_name=dataset_name,
    description=desciption
        )
    if not answer:
        client.create_examples(
            inputs=[{"question": q} for q in df.question.values],
            outputs=[{"query": q} for q in df["query"].values],
            dataset_id=dataset.id,
            )
    else:
        client.create_examples(
            inputs=[{"question": q} for q in df.question.values],
            outputs=[{"query": q, "answer": a} for q,a in zip(df["query"].values, df["answer"].values)],
            dataset_id=dataset.id,
            )
            
def load_testpkl_df(path: str) -> pd.DataFrame:
    with open(path, 'rb') as file:
        data = pickle.load(file)
        
    df = pd.DataFrame(data)[['text', 'db_id', 'db_path']].rename(columns={'text': 'question'})
    return df


def sql_parse(text: str) -> Dict[str, List[str]]:
    sql_split = text.split("\n")
    schema = {}

    for text in sql_split:
        table_match = re.search(r'CREATE\s+TABLE\s+("?)(\w+)("?)\s*\(', text)
        if table_match and table_match.group(2) not in schema:
            table_name = table_match.group(2)
            schema[table_name] = []
        
        column_match = re.search(r"^\t(.*?),", text)
        if column_match:
            term = column_match[0].split()
            if ("PRIMARY" not in term and "KEY" not in term) or ("FOREIGN" not in term and "KEY" not in term):
                col_name = term[0]
                col_name = re.sub(r'"', '', col_name)
                schema[table_name].append(col_name)
    return schema

# Function to find if any three consecutive letters in word are in a column name
def contains_consecutive(word, column_name):
    
    for i in range(len(word) - 3):
        substring = word[i:i+4]
        if substring in column_name:
            return True
    return False

# Mask the question
def mask_question(question, db_schema, common_schema_toks = None):
    words = question.split()
    
    # Create a stemmer object
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(word) for word in words]
    
    # Create a WordNetLemmatizer object
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word in the list
    words_lemmatize = [lemmatizer.lemmatize(word) for word in words]
    
    if type(db_schema) == dict:
        # for table, cols in db_schema['tables'].items():
        for table, cols in db_schema.items():
            cols.append(table)

            # Split column names with '_', for example, 'employee_id' will be split to 'employee' and 'id'
            schema_related_toks = [word.split('_') if '_' in word else word for word in cols]
            schema_related_toks = [item for element in schema_related_toks for item in (element if isinstance(element, list) else [element])]
            if common_schema_toks:
                schema_related_toks += common_schema_toks
            schema_related_toks = [lemmatizer.lemmatize(word.lower()) for word in schema_related_toks] # Lemmatise the schema related toks as well
            
            for tok in schema_related_toks:
                for i, word in enumerate(words_lemmatize):
                    if contains_consecutive(word.lower(), tok):
                        words[i] = '[MASK]'
                    elif word in schema_related_toks:
                        words[i] = '[MASK]'

        return ' '.join(words)
    
    elif type(db_schema) == list:
        # Split column names with '_', for example, 'employee_id' will be split to 'employee' and 'id'
        schema_related_toks = [word.split('_') if '_' in word else word for word in db_schema]
        schema_related_toks = [item for element in schema_related_toks for item in (element if isinstance(element, list) else [element])]
        if common_schema_toks:
            schema_related_toks += common_schema_toks
        schema_related_toks = [lemmatizer.lemmatize(word.lower()) for word in schema_related_toks] # Lemmatise the schema related toks as well

        for tok in schema_related_toks:
            for i, word in enumerate(words_lemmatize):
                if contains_consecutive(word.lower(), tok):
                    words[i] = '[MASK]'
                elif word in schema_related_toks:
                    words[i] = '[MASK]'

        return ' '.join(words)

def get_schema_related_toks(row):
    schema_toks = []
    schema_toks = [table[0] for table in row['tables']]
    schema_toks += row['columns']
    return schema_toks

def mask_question_df(row, common_schema_related_toks):
    question, db_schema = row['question'], row['schema_toks']
    return mask_question(question, db_schema, common_schema_related_toks)

def search_in_document(df: pd.DataFrame, keywords: List[str]) -> str:
    res = []
    
    for keyword in keywords:
       
        mask = df['title'].apply(lambda x: keyword in str(x).lower())
        res.extend(df[mask].content.to_list())
        
    return '\n'.join(res)
    
from collections import Counter


def summarise_keywords_from_result(retrieved_docs: List[Document]) -> str:
    
    keywords = [item.metadata['sql_keywords'] for item in retrieved_docs]
    keywords = [item.split(',') for item in keywords]
    keywords = [words for lst in keywords for words in lst]
    word_counts = dict(Counter(keywords))

    # If the keywords appears twice, it is considered relevant
    keywords_selected = [word for word, count in word_counts.items() if count >= 2]
    ignore_keywords = ['select', 'from']
    
    return [keyword for keyword in keywords_selected if keyword not in ignore_keywords]
    
def summarise_qa_from_result(retrieved_docs: List[Document]) -> str:
    
    
    res = [item.metadata['question'] + '\n' + item.metadata['query'] for item in retrieved_docs]
    return '\n'.join(res)
    
# def get_query_result(query):
#     return vectorstore.similarity_search(query, k=3)

# def clean_output(path, out):
#     path = 'evaluation/out/secondmodel.txt'
#     out = 'evaluation/out/secondmodel2.txt'
#     with open(path, 'r') as f, open(out, 'w') as output:
#         for l in f: 
#             if 'SELECT' in l:
#                 line = l.replace('"', '')
#                 line = re.sub(r" LIMIT \d+", "", line, flags=re.IGNORECASE)
#                 if "```sql " in l:
#                     line = line.replace('```sql ', '')
#                 if '```' in l:
#                     line = line.replace('```', '')
                
#             output.write(str(line))
        