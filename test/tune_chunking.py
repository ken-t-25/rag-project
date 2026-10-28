from src.data_extractor.pdf_extractor import read_pdf
from src.data_processor.chucker import Chunker
from src.utils import get_test_query_path
from test.evaluation import score_retrieval
from sentence_transformers import SentenceTransformer
import itertools
import json
import numpy as np

# hyperparameter grid; overlap must be > 0 (the chunker's overlap loop never ends with 0) and
# < min_chunk_tokens (otherwise a chunk can be completed holding nothing but the previous overlap)
MAX_CHUNK_TOKENS = [ 80, 120, 160, 200, 240 ] # all-MiniLM-L6-v2 truncates inputs at 256 tokens
MIN_CHUNK_RATIOS = [ 0.25, 0.5 ]              # min_chunk_tokens = ratio * max_chunk_tokens
OVERLAP_TOKENS = [ 10, 20, 40 ]

# retrieved chunks are compared at an equal context budget (k = TOKEN_BUDGET / max_chunk_tokens);
# with a fixed k, larger chunks would score better simply by giving the LLM more text
TOKEN_BUDGET = 600
PRIMARY_METRIC = "passage_recall"
TIE_BREAK_METRIC = "mrr"
METRICS = [ "passage_recall", "hit", "mrr", "precision" ]


def load_queries( query_file_name ):
    with open( str( get_test_query_path( query_file_name ) ), "r", encoding="utf-8" ) as file:
        data = json.load( file )
    # unanswerable queries (level -1) have no expected passages, so they cannot be scored on retrieval
    return data[ "file_name" ], [ q for q in data[ "queries" ] if q[ "level" ] > 0 ]

def get_configs():
    configs = []
    for max_tokens, min_ratio, overlap in itertools.product( MAX_CHUNK_TOKENS, MIN_CHUNK_RATIOS, OVERLAP_TOKENS ):
        min_tokens = int( max_tokens * min_ratio )
        if overlap < min_tokens:
            configs.append( ( max_tokens, min_tokens, overlap ) )
    return configs

def get_k( max_chunk_tokens ):
    return max( 1, round( TOKEN_BUDGET / max_chunk_tokens ) )

def score_config( chunks, chunk_embeddings, query_embeddings, queries, k ):
    # return a list with the retrieval metrics of each query
    similarities = query_embeddings @ chunk_embeddings.T # embeddings are normalized, so this is cosine similarity
    return score_similarities( chunks, similarities, queries, k )

def score_similarities( chunks, similarities, queries, k ):
    # return a list with the retrieval metrics of each query, retrieving the k chunks with the highest
    # similarity (similarities: num_queries x num_chunks)
    top_k_idx = np.argsort( -similarities, axis=1 )[ :, :k ]
    return [
        score_retrieval( [ chunks[ i ] for i in top_k_idx[ qi ] ], q[ "passages" ] )
        for qi, q in enumerate( queries )
    ]

def mean_metrics( scores ):
    return { m: float( np.mean( [ s[ m ] for s in scores ] ) ) for m in METRICS }

def rank_configs( config_scores ):
    # return the configs sorted by mean primary metric (then mean tie-break metric), best first
    def key( config ):
        scores = config_scores[ config ]
        return ( np.mean( [ s[ PRIMARY_METRIC ] for s in scores ] ), np.mean( [ s[ TIE_BREAK_METRIC ] for s in scores ] ) )
    return sorted( config_scores, key=key, reverse=True )

def format_config( config ):
    max_tokens, min_tokens, overlap = config
    return f"max({ max_tokens })_min({ min_tokens })_overlap({ overlap })_k({ get_k( max_tokens ) })"

def report_metrics( name, scores ):
    print( f"  { name:<10}" + "".join( f"{ m }={ v:.3f}  " for m, v in mean_metrics( scores ).items() ) + f"(n={ len( scores ) })" )

# This function tunes the chunking hyperparameters by:
# 1) scoring every config on the train queries and selecting the best one
#    (nothing is fitted to the queries, so they only serve to compare configs)
# 2) scoring only the selected config on the test queries, which gives an unbiased estimate of its
#    performance since the test queries played no part in the selection
def tune_chunking( query_file_name, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2" ):
    file_name, train_queries = load_queries( f"{ query_file_name }_train" )
    _, test_queries = load_queries( f"{ query_file_name }_test" )

    embedding_model = SentenceTransformer( embedding_model_name )
    tokenizer = embedding_model.tokenizer
    pages_info = read_pdf( file_name )
    train_query_embeddings = embedding_model.encode( [ q[ "query" ] for q in train_queries ] )

    config_scores = {}
    config_chunks = {}
    for config in get_configs():
        max_tokens, min_tokens, overlap = config
        chunks = Chunker( tokenizer, max_tokens, overlap, min_tokens ).page_block_chunker( pages_info )
        chunk_embeddings = embedding_model.encode( [ c[ "text" ] for c in chunks ], batch_size=64 )
        config_chunks[ config ] = ( chunks, chunk_embeddings )
        config_scores[ config ] = score_config( chunks, chunk_embeddings, train_query_embeddings, train_queries, get_k( max_tokens ) )
        print( f"scored { format_config( config ) }: { len( chunks ) } chunks" )

    ranked = rank_configs( config_scores )
    print( f"\nConfigs ranked by mean { PRIMARY_METRIC } on { len( train_queries ) } train queries:" )
    for config in ranked:
        print( f"  { format_config( config ):<40} { np.mean( [ s[ PRIMARY_METRIC ] for s in config_scores[ config ] ] ):.3f}" )

    best = ranked[ 0 ]
    print( f"\nSelected config: { format_config( best ) }" )
    report_metrics( "train", config_scores[ best ] )

    # the test set is only used once, to report the performance of the selected config
    chunks, chunk_embeddings = config_chunks[ best ]
    test_query_embeddings = embedding_model.encode( [ q[ "query" ] for q in test_queries ] )
    test_scores = score_config( chunks, chunk_embeddings, test_query_embeddings, test_queries, get_k( best[ 0 ] ) )
    report_metrics( "test", test_scores )
    for level in sorted( { q[ "level" ] for q in test_queries } ):
        report_metrics( f"  level { level }", [ s for s, q in zip( test_scores, test_queries ) if q[ "level" ] == level ] )
    return best

if __name__ == "__main__":
    tune_chunking( "query1" )
