from chain import rerank_chain, rag_chain
from self_rag import model as correct_chain
from baseline import sql_chain as baseline_chain
from utils import * 
import argparse
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine 
import logging
import os
import asyncio
from typing import Tuple
import tqdm


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def process_arguments() -> Tuple[str, int]:
    
    parser = argparse.ArgumentParser(description="Arguments for test suite evaluation")
    parser.add_argument("model", type=str, help = 'text to sql chain type')
    parser.add_argument("database_folder", type=str, help="path of the database folder")
    parser.add_argument("output_path", type=str, help="output path of txt file")
    args = parser.parse_args()


    return args.model, args.database_folder, args.output_path


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
    model, db_folder, output_path = process_arguments()
   
    df = load_testpkl_df('./evaluation/classical_test.pkl')

    test_database = ['academic']
    for db in test_database:
        logging.info(f'Evaluating {db} database')
        df_db = df.loc[df['db_id'] == db].loc
        path = df_db['db_path'].unique()[0]
        db_path = os.path.join(db_folder, path)
        # db_path = './evaluation/database/academic/academicv1223round10group3.sqlite'
        engine = create_engine(f'sqlite:///{db_path}')
        schema = connect_db('engine', engine = engine).get_table_info()
        df_db['masked'] = df_db.apply(lambda row: mask_question(row['question'], sql_parse(schema), ['student', 'course', 'age', 'course', 'ids', 'car', 'player', 'class', 'cities', 'member', 'employee']), axis = 1)
        
        logging.info(f'Successfully connect to {db} base')
        #outputs = await batch_process(df_db, 20, 50, schema)
        outputs = await batch_process(df_db, 20, 50, model, schema)

        if not os.path.exists(os.path.dirname(output_path)):
            os.mkdir(os.path.dirname(output_path))
            logging.info('Output directory has been created!')
            
        if os.path.exists(output_path):
            os.remove(output_path)
            logging.info('Output files has been deleted!')
            
    with open(output_path, 'a') as f:
        for output in outputs:
            f.write(f"{output}\n")
             

if __name__ == '__main__':
    asyncio.run(main())

    
             
        
        
       
       
        
        
        
        
        

