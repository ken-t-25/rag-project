from src.data_processor.embedding_processor import get_chunk_similarities
from src.utils import get_chunk_path, get_embedding_path, get_sliding_window_embedding_path
from dotenv import load_dotenv
from openai import OpenAI
import json
import numpy as np
import os

load_dotenv()

api_key = os.getenv( "OPENAI_API_KEY" )

class QueryProcessor:
    def __init__( self, embedding_model ):
        self.embedding_model = embedding_model
        self.client = OpenAI( api_key=api_key )

    def process_query( self, query, file_name, top_k=5 ):
        retrieved_chunks = self.retrieval( query, file_name, top_k )
        prompt = self.create_prompt( query, retrieved_chunks )
        print( prompt )
        response = self.client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )
        return response.output_text

    def retrieval( self, query, file_name, top_k=5 ):
        file_embeddings = self.load_file_embeddings( file_name )
        if file_embeddings.size <= 0:
            print( f"No embeddings found for file: {file_name}" )
            return []
        file_chunks = self.load_file_chunks( file_name )
        if not file_chunks:
            print( f"No chunks found for file: {file_name}" )
            return []
        top_k = min( top_k, len( file_embeddings ) )
        query_embedding = self.embedding_model.encode( query )
        similarities = np.dot( file_embeddings, query_embedding ) # np.dot, inputs' inner dimensions should match (like for matrix multiplication)
        most_similar_idx = np.argsort( similarities )[ -top_k: ][ ::-1 ]
        print( f"Similarities: { similarities[ most_similar_idx ] }" )
        return [ file_chunks[ i ] for i in most_similar_idx ]

    # Same as retrieval, except that each chunk has several window embeddings (see
    # EmbeddingProcessor.process_chunks_sliding_window) and its similarity is the mean of its top_m
    # window similarities
    def retrieval_sliding_window( self, query, file_name, top_k=5, top_m=2 ):
        window_embeddings, window_chunk_ids = self.load_sliding_window_embeddings( file_name )
        if window_embeddings.size <= 0:
            print( f"No sliding window embeddings found for file: {file_name}" )
            return []
        file_chunks = self.load_file_chunks( file_name )
        if not file_chunks:
            print( f"No chunks found for file: {file_name}" )
            return []
        top_k = min( top_k, len( file_chunks ) )
        query_embedding = self.embedding_model.encode( query )
        window_similarities = np.dot( window_embeddings, query_embedding )
        similarities = get_chunk_similarities( window_similarities[ None, : ], window_chunk_ids, len( file_chunks ), top_m )[ 0 ]
        most_similar_idx = np.argsort( similarities )[ -top_k: ][ ::-1 ]
        print( f"Similarities: { similarities[ most_similar_idx ] }" )
        return [ file_chunks[ i ] for i in most_similar_idx ]

    def create_prompt( self, query, retrieved_chunks ):
        context = "\n\n".join(
            f"[Chunk { i + 1 }]\n{ chunk[ "text" ] }"
            for i, chunk in enumerate( retrieved_chunks )
        )
        return f"""
You are a question-answering assistant.

Answer the user's question using only the information provided in the context below.

Do not use outside knowledge or information not contained in the context.
If the context does not contain enough information to answer the question, say that you cannot determine the answer from the provided context.

Context:
---
{context}
---

Question:
{query}
        """

    def load_file_embeddings( self, file_name ):
        embedding_file_path = get_embedding_path( file_name )
        if embedding_file_path.is_file():
            return np.load( str( embedding_file_path ) )
        else:
            return np.zeros( 0 )

    def load_sliding_window_embeddings( self, file_name ):
        embedding_file_path = get_sliding_window_embedding_path( file_name )
        if embedding_file_path.is_file():
            data = np.load( str( embedding_file_path ) )
            return data[ "embeddings" ], data[ "chunk_ids" ]
        else:
            return np.zeros( 0 ), np.zeros( 0, dtype=int )

    def load_file_chunks( self, file_name ):
        chunk_file_path = get_chunk_path( file_name )
        if chunk_file_path.is_file():
            with open( str( chunk_file_path ), "r", encoding="utf-8" ) as file:
                return json.load( file )
        else:
            return []
