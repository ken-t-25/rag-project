from src.data_extractor.pdf_extractor import read_pdf
from src.data_processor.chucker import Chunker
from src.data_processor.embedding_processor import EmbeddingProcessor
from src.query_processor import QueryProcessor
from src.utils import get_chunk_path, get_embedding_path, get_test_query_path
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import sys


def evaluate_chunk( chunk, expected_pages ):
    # return True if the retrieved chunk's page number is in the expected pages
    for p in expected_pages:
        if chunk[ "start_page" ] <= p <= chunk[ "end_page" ]:
            return True
    return False

def evaluate_retrieved_chunks( retrieved_chunks, expected_pages ):
    # return the number of retrieved chunks that have page numbers in the expected pages
    relevant_chunks = 0
    for chunk in retrieved_chunks:
        print( chunk )
        if evaluate_chunk( chunk, expected_pages ):
            relevant_chunks += 1
    return relevant_chunks

def report_results( results ):
    for level, level_results in results.items():
        total_queries = sum( level_results.values() )
        print( f"Level { level }:" )
        for num_relevant_chunks, count in sorted( level_results.items() ):
            percentage = ( count / total_queries ) * 100
            print( f"  { num_relevant_chunks } relevant chunks: { count } queries ({ percentage:.2f}% )" )

def get_case_specific_file_name( file_name, max_chunk_tokens, min_chunk_tokens, overlap_tokens ):
    return f"{ file_name }_max({ max_chunk_tokens })_min({ min_chunk_tokens })_overlap({ overlap_tokens })"

def prepare_test( file_name, embedding_model, max_chunk_tokens, min_chunk_tokens, overlap_tokens ):
    case_specific_file_name = get_case_specific_file_name( file_name, max_chunk_tokens, min_chunk_tokens, overlap_tokens )
    chunk_file_path = get_chunk_path( case_specific_file_name )
    embedding_file_path = get_embedding_path( case_specific_file_name )

    if chunk_file_path.is_file():
        with open( str( chunk_file_path ), "r", encoding="utf-8" ) as file:
            chunks = json.load( file )
    else:
        pages_info = read_pdf( file_name )
        tokenizer = embedding_model.tokenizer
        chunker = Chunker( tokenizer, max_chunk_tokens, overlap_tokens, min_chunk_tokens )
        chunks = chunker.page_block_chunker( pages_info )
        with open( str( chunk_file_path ), "w", encoding="utf-8" ) as file:
            json.dump( chunks, file,indent=4 )

    if embedding_file_path.is_file():
        embeddings = np.load( str( embedding_file_path ) )
    else:
        embedding_processor = EmbeddingProcessor( embedding_model )
        embeddings = embedding_processor.process_chunks( chunks )
        np.save( str( embedding_file_path ), embeddings )

def test_retrieval( query_file_name, max_chunk_tokens, min_chunk_tokens, overlap_tokens, k=5 ):
    embedding_model = SentenceTransformer( "sentence-transformers/all-MiniLM-L6-v2" )
    query_processor = QueryProcessor( embedding_model )
    with open( str( get_test_query_path( query_file_name ) ), "r", encoding="utf-8" ) as file:
        test_queries = json.load( file )
        file_name = test_queries[ "file_name" ]
        queries = test_queries[ "queries" ]
    prepare_test( file_name, embedding_model, max_chunk_tokens, min_chunk_tokens, overlap_tokens )
    results = {}
    for q in queries:
        query = q[ "query" ]
        level = q[ "level" ]
        pages = q[ "page" ]
        if level > 0:
            print( "Query:", query )
            chunks = query_processor.retrieval( query, get_case_specific_file_name( file_name, max_chunk_tokens, min_chunk_tokens, overlap_tokens ), k )
            num_relevant_chunks = evaluate_retrieved_chunks( chunks, pages )
            print( "Number of relevant chunks retrieved:", num_relevant_chunks )
            if level not in results:
                results[ level ] = {}
            if num_relevant_chunks not in results[ level ]:
                results[ level ][ num_relevant_chunks ] = 0
            results[ level ][ num_relevant_chunks ] += 1
    report_results( results)

def test_retrieval_with_stored_output( query_file_name, max_chunk_tokens, min_chunk_tokens, overlap_tokens, k=5 ):
    with open( f"output1_max({ max_chunk_tokens })_min({ min_chunk_tokens })_overlap({ overlap_tokens })_k({ k }).txt", "w" ) as f:
        sys.stdout = f
        test_retrieval( query_file_name, max_chunk_tokens, min_chunk_tokens, overlap_tokens, k )

test_retrieval_with_stored_output( "query1", 120, 60, 10 )
test_retrieval_with_stored_output( "query1", 120, 60, 20 )
test_retrieval_with_stored_output( "query1", 120, 60, 30 )
test_retrieval_with_stored_output( "query1", 120, 60, 40 )
test_retrieval_with_stored_output( "query1", 120, 60, 50 )
test_retrieval_with_stored_output( "query1", 120, 60, 60 )

test_retrieval_with_stored_output( "query1", 240, 80, 10, 5 )
test_retrieval_with_stored_output( "query1", 240, 80, 20, 5 )
test_retrieval_with_stored_output( "query1", 240, 80, 30, 5 )
test_retrieval_with_stored_output( "query1", 240, 80, 40, 5 )
test_retrieval_with_stored_output( "query1", 240, 80, 50, 5 )
test_retrieval_with_stored_output( "query1", 240, 80, 60, 5 )