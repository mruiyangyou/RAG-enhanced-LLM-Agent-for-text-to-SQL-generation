import numpy as np
import pandas as pd
from collections import Counter

def summarise_keywords_from_result(masked_question, index, embed_model, join_involved = None):
    
    result = get_query_result(index = index, embed_model = embed_model, masked_question = masked_question, k = 3, join_involved = join_involved, verbose = False)
    keywords = [item['sql_keywords'] for item in result]
    keywords = [item.split(',') for item in keywords]
    keywords = [words for lst in keywords for words in lst]
    word_counts = dict(Counter(keywords))

    # If the keywords appears twice, it is considered relevant
    keywords_selected = [word for word, count in word_counts.items() if count >= 2]
    
    ignore_keywords = ['select', 'from']
    
    return [keyword for keyword in keywords_selected if keyword not in ignore_keywords]
    

def get_query_result(index, embed_model, masked_question:str, k:int, join_involved:bool = None, verbose:bool = True):
    '''
    index: pinecone index to search from
    embed_model: text embedding model
    masked_question: masked natural language question
    k: top k searches to return
    join_involved: if NONE, return results without filtering; if True, return results with JOIN keywords; if False, return results without JOIN keyword
    verbose: whether or not to print results
    '''
    
    if join_involved == None:
        query_results = index.query(
            vector=embed_model.embed_documents([masked_question]), 
            top_k=k, 
            include_metadata=True)
    else:
        query_results = index.query(
            vector=embed_model.embed_documents([masked_question]), 
            top_k=k, 
            filter={"join_involved": join_involved},
            include_metadata=True)
    
    if verbose:
        for match in query_results.matches:
            print(f"Question: {match.metadata['question']}")
            print(f"Masked question: {match.metadata['masked_question']}")
            print(f"Query: {match.metadata['query']}")
            print(f"SQL keywords: {match.metadata['sql_keywords']}")
            print(f"Join involved: {match.metadata['join_involved']}")
            print(' ')
    
    return [match.metadata for match in query_results.matches]
    

# Function to search for the keyword in a specific column

def search_in_document(df, column_name, keyword):

    if column_name in df.columns:
        mask = df[column_name].apply(lambda x: keyword in str(x).lower())
        
        result_dict = {}

        for index, row in df[mask].iterrows():
            result_dict[row['title']] = {'summary': row['summary'],
                                        'content': row['content']}
            
        return result_dict

    else:
        raise ValueError(f"Column '{column_name}' is not found in DataFrame.")
    
    
    
if __name__ == '__main__':

    df = pd.read_csv('md_data.csv')
    result = search_in_document(df, 'title', 'select')

    print(result)

