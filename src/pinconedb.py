from dotenv import load_dotenv, find_dotenv
import pandas as pd
import os
import itertools
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

load_dotenv(find_dotenv())

class Pineconedb:
    def __init__(self, api_key: str):
        self.pc = Pinecone(api_key=api_key)

    def create_pinecone_index(self, index_name: str, dim_size: int, metric='euclidean'):
        self.pc.create_index(
            name=index_name,
            dimension=dim_size,
            metric=metric,
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )

    def load_pinecone_index(self, index_name: str):
        return self.pc.Index(index_name)

    @staticmethod
    def chunks(iterable, batch_size=100):
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def batch_embedding_dataframe(self, index, df: pd.DataFrame, batch_size: int, embedd):
        extract_data = lambda x: {
            "id": f'question-{x.Index + 1}',
            'values': embedd.embed_query(x.question),
            "metadata": {'type': x.query_type, 'question': x.question, 'query': x.query}
        }
        data_generator = map(extract_data, df.itertuples(index=True))

        for i in Pineconedb.chunks(data_generator, batch_size=batch_size):
            index.upsert(i)

    def create_pinecone_vectordb_from_index(self, index, embedd, text_key: str):
        db = PineconeVectorStore(index=index, embedding=embedd, text_key=text_key)
        return db
    
    def create_pincone_vectordb_from_documents(self, documents, embedd):
        db = PineconeVectorStore.from_documents(documents, embedd)

    @staticmethod
    def filter_dict(x: List[str]) -> Dict[str, str]:
        return {"$and": [{"type": type} for type in x]}

    def create_pinecone_retriever(self, pineconedb, search_type='similarity', search_kwargs={'k': 3}):
        return pineconedb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    @staticmethod
    def fit_bm25(corpus: List[str], output_path: str):
        bm25 = BM25Encoder()
        bm25.fit(corpus)
        bm25.dump(output_path)
        
    def create_pinecone_hybridsearch(self, index, embedding, bm25_path, **kwargs):
        bm25_encoder = BM25Encoder().load(bm25_path)
        hybird_search = PineconeHybridSearchRetriever(embeddings=embedding, 
                                                      sparse_encoder=bm25_encoder,
                                                      index = index,
                                                      **kwargs)
        return hybird_search
        
        
    