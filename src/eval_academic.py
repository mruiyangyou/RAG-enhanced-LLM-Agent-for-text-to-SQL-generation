from chain import rerank_chain, rag_chain, upgrade_chain
from self_rag import model as correct_chain
from baseline import sql_chain as baseline_chain
from utils import * 
import argparse
from sqlalchemy import create_engine 
import logging
import os
import asyncio
from typing import Tuple
import tqdm
import pandas as pd

# logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def process_arguments() -> Tuple[str, int]:
    
    parser = argparse.ArgumentParser(description="Arguments for test suite evaluation")
    #parser.add_argument("database_folder", type=str, help="path of the database folder")
    parser.add_argument("model", type=str, help = 'text to sql chain type')
    parser.add_argument("question_path", type = str, help = 'path of question file')
    # parser.add_argument("db_path", type=str, help="database path")
    parser.add_argument("output_path", type=str, help="output path of txt file")
    parser.add_argument("schema_col", type = str, help="This argument is for choosing from different schema")
    args = parser.parse_args()
    return args.model, args.question_path,args.output_path, args.schema_col

# batch evaluation
async def batch_process(data, batch_size, wait_time, 
                        model, schema_col):
    res = []
    
    if model == 'baseline':
        chain = baseline_chain
    elif model == 'rag':
        chain = rag_chain
    elif model == 'rerank':
        chain = rerank_chain
    elif model == 'self-rag':
        chain = correct_chain
    elif model == 'upgrade':
        chain = upgrade_chain
    else:
        raise ValueError("Please insert argument model from [rag, rerank, baseline, self-rag]")
    
    if_baseline = True if model == 'baseline' else False
    
    for start in tqdm.tqdm(range(0, len(data), batch_size)):
        end = start + batch_size
        batch = data[start:end]
        
        if not if_baseline:
            input_chunks = [{'question': row['question'], 'schema': row[schema_col], 'masked': row['masked']} for index, row in batch.iterrows()]
        else:
            input_chunks = [{'question': row['question'], 'schema': row[schema_col]} for index, row in batch.iterrows()]
            
  
        if model != 'self-rag':
            outputs_chunk = await chain.abatch(input_chunks)
        else:
            outputs_chunk = list(map(chain, input_chunks))
        
        outputs_chunk = list(map(lambda x: x.replace('\n', ' '), outputs_chunk))
        res.extend(outputs_chunk)
        

        await asyncio.sleep(wait_time)
        
    return res

async def main() -> None:
    model, question_path, output_path, schema_col = process_arguments()
    
    # create directory for saving
    if not os.path.exists(os.path.dirname(output_path)):
            os.mkdir(os.path.dirname(output_path))
            logging.info('Output directory has been created!')
   
    df = pd.read_csv(question_path)
    
    logging.info(f'Evaluating academic database')
    # db_path = '/Users/marceloyou/Desktop/UCL-DSML/COMP0087-Boss/SQLess/data/inventory/inventory.sqlite'
    outputs = await batch_process(df, 20, 5, model, schema_col)
    
    with open(output_path, 'a') as f:
        for output in outputs:
            f.write(f"{output}\n")

if __name__ == '__main__':
    asyncio.run(main())
     
             
        
        
       
       
        
        
        
        
        

