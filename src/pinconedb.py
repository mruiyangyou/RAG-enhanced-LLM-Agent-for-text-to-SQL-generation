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


class Pineconedb:
    """
    Provides an interface for managing and interacting with Pinecone databases.

    Methods:
        __init__(api_key: str): Initializes the Pinecone client with the given API key.
        create_pinecone_index(index_name: str, dim_size: int, metric='euclidean'): Creates a new Pinecone index.
        load_pinecone_index(index_name: str): Loads an existing Pinecone index.
        chunks(iterable, batch_size=100): Yields chunks of the iterable in the specified batch size.
        batch_embedding_dataframe(index, df: pd.DataFrame, batch_size: int, embedd): Embeds dataframe content in batches and upserts into the index.
        create_pinecone_vectordb_from_index(index, embedd, text_key: str): Creates a vector database from an existing Pinecone index.
        create_pincone_vectordb_from_documents(documents, embedd): Creates a vector database directly from documents.
        filter_dict(x: List[str]) -> Dict[str, str]: Creates a filter dict from a list of strings for database queries.
        create_pinecone_retriever(pineconedb, search_type='similarity', search_kwargs={'k': 3}): Creates a retriever for the Pinecone database.
        fit_bm25(corpus: List[str], output_path: str): Fits a BM25 model to the given corpus and saves the model.
        create_pinecone_hybridsearch(index, embedding, bm25_path, **kwargs): Creates a hybrid search retriever combining embeddings and BM25.
    """
    def __init__(self, api_key: str):
        """
        Initializes the Pinecone client with the specified API key.
        """
        self.pc = Pinecone(api_key=api_key)
        
    def create_pinecone_index(self, index_name: str, dim_size: int, metric='euclidean'):
        """
        Creates a new Pinecone index with the specified name, dimension size, and metric.
        """
        self.pc.create_index(
            name=index_name,
            dimension=dim_size,
            metric=metric,
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )

    def load_pinecone_index(self, index_name: str):
        """
        Loads an existing Pinecone index by its name.
        """
        return self.pc.Index(index_name)

    @staticmethod
    def chunks(iterable, batch_size=100):
        """
        Yields chunks of the specified iterable in batches of the given size.
        """
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def batch_embedding_dataframe(self, index, df: pd.DataFrame, batch_size: int, embedd):
        """
        Processes a DataFrame in batches, embedding its contents and upserting into the specified Pinecone index.
        """
        extract_data = lambda x: {
            "id": f'question-{x.Index + 1}',
            'values': embedd.embed_query(x.question),
            "metadata": {'type': x.query_type, 'question': x.question, 'query': x.query}
        }
        data_generator = map(extract_data, df.itertuples(index=True))

        for chunk in Pineconedb.chunks(data_generator, batch_size=batch_size):
            index.upsert(chunk)

    def create_pinecone_vectordb_from_index(self, index, embedd, text_key: str):
        """
        Creates a vector database from an existing Pinecone index, embedding, and text key.
        """
        db = PineconeVectorStore(index=index, embedding=embedd, text_key=text_key)
        return db
    
    def create_pincone_vectordb_from_documents(self, documents, embedd):
        """
        Creates a vector database directly from a list of documents and an embedding.
        """
        db = PineconeVectorStore.from_documents(documents, embedd)

    @staticmethod
    def filter_dict(x: List[str]) -> Dict[str, str]:
        """
        Creates a dictionary for filtering database queries based on a list of strings.
        """
        return {"$and": [{"type": type} for type in x]}

    def create_pinecone_retriever(self, pineconedb, search_type='similarity', search_kwargs={'k': 3}):
        """
        Creates a retriever for executing similarity or other types of searches on the Pinecone database.
        """
        return pineconedb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    @staticmethod
    def fit_bm25(corpus: List[str], output_path: str):
        """
        Fit bm25 model
        """
        bm25 = BM25Encoder()
        bm25.fit(corpus)
        bm25.dump(output_path)
        
    def create_pinecone_hybridsearch(self, index, embedding, bm25_path, **kwargs):
        """
        Construct a hybird search retriever
        """
        bm25_encoder = BM25Encoder().load(bm25_path)
        hybird_search = PineconeHybridSearchRetriever(embeddings=embedding, 
                                                      sparse_encoder=bm25_encoder,
                                                      index = index,
                                                      **kwargs)
        return hybird_search