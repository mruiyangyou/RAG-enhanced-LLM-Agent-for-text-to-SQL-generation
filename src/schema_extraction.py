import re
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Tuple

lemmatizer = WordNetLemmatizer()

def sql_parse(text: str) -> Dict[str, List[str]]:
    '''
    input: 
        text: schema of the database
    output: 
        parsed schema, a dictionary where keys are table names and values are the correponding column names
    '''
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

def col_extraction(text: str, db_schema: Dict[str, List[str]]) -> Tuple[Dict[str, str], List[str]]:
    '''
    input:
        text: natural language question
        db_schema: schema of the database that are parsed
    output:
        related_cols: predicted columns that are related
        related_tables: predicted tables that are related
    '''
    related_cols = {}
    related_tables = []
    words = text.split()
    words_lemmatize = [lemmatizer.lemmatize(word) for word in words]
    
    if type(db_schema) == dict:
        for table, cols in db_schema.items():
            cols.append(table)
            
            schema_related_toks = [word.split('_') if '_' in word else word for word in cols]
            schema_related_toks = [item for element in schema_related_toks for item in (element if isinstance(element, list) else [element])]
            # if common_schema_toks:
            #     schema_related_toks += common_schema_toks
            schema_related_toks = [lemmatizer.lemmatize(word.lower()) for word in schema_related_toks]
            
            for word in words_lemmatize:
                if word in schema_related_toks[:-1]: # exclude the last one cause it is table name
                    if table not in related_cols:
                        related_cols[table] = [word]
                    else:
                        related_cols[table].append(word)
                elif word == schema_related_toks[-1] and word not in related_tables:
                    related_tables.append(word)
    
    return related_cols, related_tables

def keyword_extraction(text:str, db_schema: str) -> List[str]:
    '''
    input:
        text: natural language question
        db_schema: schema of the databse
    output:
        tables that are predicted to be related to the natural languge question
    '''
    related_col, related_table = col_extraction(text, sql_parse(db_schema))
    return list(set(list(related_col.keys()) + related_table))

def schema_extraction(text: str, db_schema: str) -> str:
    '''
    input:
        text: natural language question
        db_schema: original schema of the database
    output:
        predicted related schema given the natural language
    '''
    schema_list = re.findall(r"(\nCREATE TABLE.*?\*/)", db_schema, re.DOTALL)
    related_schema = []
    related_table = keyword_extraction(text, db_schema)
    
    for table in related_table:
        related_schema = [schema for schema in schema_list if re.search(f"\s{table}\s", schema)]
    
    update_schema = "\n\n".join(related_schema)
    
    if len(update_schema) == 0:
        return db_schema
    else:
        return update_schema