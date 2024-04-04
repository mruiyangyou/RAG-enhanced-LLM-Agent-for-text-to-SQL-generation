from chain import rerank_chain, rag_chain, upgrade_chain
from self_rag import model as correct_chain
from baseline import sql_chain as baseline_chain
from utils import * 
import argparse
from sqlalchemy import create_engine 
import logging
import os
import asyncio
import time
from typing import Tuple
import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def process_arguments() -> Tuple[str, int]:
    
    parser = argparse.ArgumentParser(description="Arguments for test suite evaluation")
    #parser.add_argument("database_folder", type=str, help="path of the database folder")
    parser.add_argument("model", type=str, help = 'text to sql chain type')
    parser.add_argument("question_path", type = str, help = 'path of question file')
    parser.add_argument("db_path", type=str, help="database path")
    parser.add_argument("output_path", type=str, help="output path of txt file")
    #parser.add_argument("--if_baseline", action="store_true", help="Indicate whether to use the baseline method", default=False)
    args = parser.parse_args()

    # Accessing the arguments
    # print(f"Database folder: {args.database_folder}")
    # print(f"Output path of txt files: {args.output_path}")
    return args.model, args.question_path, args.db_path, args.output_path


async def batch_process(data, batch_size, wait_time, 
                        model, schema):
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
            input_chunks = [{'question': row['question'], 'schema': schema, 'masked': row['masked']} for index, row in batch.iterrows()]
        else:
            input_chunks = [{'question': row['question'], 'schema': schema} for index, row in batch.iterrows()]
            
        if model != 'self-rag':
            outputs_chunk = await chain.abatch(input_chunks)
        else:
            outputs_chunk = list(map(chain, input_chunks))
            
        outputs_chunk = list(map(lambda x: x.replace('\n', ' '), outputs_chunk))
        res.extend(outputs_chunk)

        await asyncio.sleep(wait_time)
        
    return res

async def main() -> None:
    model, question_path, db_path,output_path = process_arguments()

    # df = pd.read_csv('/Users/marceloyou/Desktop/UCL-DSML/COMP0087-Boss/SQLess/data/inventory/inventory.csv')
    df = pd.read_csv(question_path)
    print(len(df))

    
    logging.info(f'Evaluating {db_path} database')
    # db_path = '/Users/marceloyou/Desktop/UCL-DSML/COMP0087-Boss/SQLess/data/inventory/inventory.sqlite'
    engine = create_engine(f'sqlite:///{db_path}')
    db = connect_db('engine', engine = engine)
    logging.info(f'Successfully connect to {db_path}')
    schema = db.get_table_info()
    outputs = await batch_process(df, 20, 30, model, schema)
    assert len(outputs) == len(df)
    df['predicted'] = outputs

    df = await execute_all_queries(db, df, 'predicted', 'predict_ans')
    if not os.path.exists(os.path.dirname(output_path)):
        os.mkdir(os.path.dirname(output_path))
        logging.info('Output directory has been created!')
        
    if os.path.exists(output_path):
        os.remove(output_path)
        logging.info('Output files has been deleted!')
        
    df.drop(columns=['masked']).to_csv(output_path, index = None)
             

if __name__ == '__main__':
    asyncio.run(main())
     
             
        
        
       
       
        
        
        
        
        

