from src.data_extractor.pdf_extractor import read_pdf
from src.data_processor.chucker import Chunker
from src.data_processor.embedding_processor import EmbeddingProcessor, get_chunk_similarities, pool_windows
from test.tune_chunking import PRIMARY_METRIC, format_config, get_configs, get_k, load_queries, rank_configs, report_metrics, score_similarities
from sentence_transformers import SentenceTransformer
import itertools
import numpy as np

# sliding window hyperparameter grid (tuned together with the chunking hyperparameters of tune_chunking)
WINDOW_SIZES = [ 16, 32, 64 ]
STRIDE_RATIOS = [ 0.5, 1.0 ] # stride = ratio * window_size (0.5: windows overlap by half, 1.0: no overlap)
TOP_MS = [ 1, 2, 3 ]         # number of best window similarities averaged into a chunk similarity


def get_window_configs():
    return [
        ( window_size, max( 1, int( window_size * stride_ratio ) ), top_m )
        for window_size, stride_ratio, top_m in itertools.product( WINDOW_SIZES, STRIDE_RATIOS, TOP_MS )
    ]

def format_sliding_config( config ):
    chunk_config, window_size, stride, top_m = config
    return f"{ format_config( chunk_config ) }_window({ window_size })_stride({ stride })_m({ top_m })"

def score_sliding_window( chunks, token_embeddings, query_embeddings, queries, window_size, stride, top_m, k ):
    window_embeddings, window_chunk_ids = pool_windows( token_embeddings, window_size, stride )
    window_similarities = query_embeddings @ window_embeddings.T
    similarities = get_chunk_similarities( window_similarities, window_chunk_ids, len( chunks ), top_m )
    return score_similarities( chunks, similarities, queries, k )

def report_comparison( name, baseline_scores, sliding_scores, queries ):
    print( f"\n{ name }:" )
    report_metrics( "baseline", baseline_scores )
    report_metrics( "sliding", sliding_scores )
    for level in sorted( { q[ "level" ] for q in queries } ):
        idx = [ i for i, q in enumerate( queries ) if q[ "level" ] == level ]
        baseline_value = np.mean( [ baseline_scores[ i ][ PRIMARY_METRIC ] for i in idx ] )
        sliding_value = np.mean( [ sliding_scores[ i ][ PRIMARY_METRIC ] for i in idx ] )
        print( f"    level { level }: { PRIMARY_METRIC } baseline={ baseline_value:.3f}  sliding={ sliding_value:.3f}  (n={ len( idx ) })" )
    # per-query comparison, since both methods are scored on the same queries
    differences = [ s[ PRIMARY_METRIC ] - b[ PRIMARY_METRIC ] for b, s in zip( baseline_scores, sliding_scores ) ]
    print( f"    per query: sliding better on { sum( d > 0 for d in differences ) }, "
           f"worse on { sum( d < 0 for d in differences ) }, tied on { sum( d == 0 for d in differences ) }" )

# This function compares the sliding window embeddings with the baseline (one embedding per chunk) by:
# 1) scoring every chunking config with both methods (and every window config with the sliding window)
#    on the train queries, and selecting the best config of each method
# 2) scoring only the two selected configs on the test queries
def tune_sliding_window( query_file_name, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2" ):
    file_name, train_queries = load_queries( f"{ query_file_name }_train" )
    _, test_queries = load_queries( f"{ query_file_name }_test" )

    embedding_model = SentenceTransformer( embedding_model_name )
    embedding_processor = EmbeddingProcessor( embedding_model )
    pages_info = read_pdf( file_name )
    train_query_embeddings = embedding_model.encode( [ q[ "query" ] for q in train_queries ] )

    baseline_scores = {}
    sliding_scores = {}
    config_chunks = {}
    for chunk_config in get_configs():
        max_tokens, min_tokens, overlap = chunk_config
        k = get_k( max_tokens )
        chunks = Chunker( embedding_processor.tokenizer, max_tokens, overlap, min_tokens ).page_block_chunker( pages_info )
        chunk_embeddings = embedding_model.encode( [ c[ "text" ] for c in chunks ], batch_size=64 )
        baseline_scores[ chunk_config ] = score_similarities( chunks, train_query_embeddings @ chunk_embeddings.T, train_queries, k )

        # the transformer runs once per chunking config; every window config reuses its token embeddings
        token_embeddings = embedding_processor.get_contextualized_token_embeddings( chunks )
        config_chunks[ chunk_config ] = ( chunks, chunk_embeddings, token_embeddings )
        for window_size, stride, top_m in get_window_configs():
            sliding_scores[ ( chunk_config, window_size, stride, top_m ) ] = score_sliding_window(
                chunks, token_embeddings, train_query_embeddings, train_queries, window_size, stride, top_m, k )
        print( f"scored { format_config( chunk_config ) }: { len( chunks ) } chunks" )

    mean_primary = lambda scores: np.mean( [ s[ PRIMARY_METRIC ] for s in scores ] )
    ranked_sliding = rank_configs( sliding_scores )
    print( f"\nTop sliding window configs by mean { PRIMARY_METRIC } on { len( train_queries ) } train queries "
           f"(of { len( ranked_sliding ) }):" )
    for config in ranked_sliding[ :15 ]:
        print( f"  { format_sliding_config( config ):<75} { mean_primary( sliding_scores[ config ] ):.3f}" )

    # for each window config, the mean over chunking configs shows its effect independently of the chunking
    print( f"\nMean { PRIMARY_METRIC } over chunking configs, by window config (baseline: "
           f"{ np.mean( [ mean_primary( s ) for s in baseline_scores.values() ] ):.3f}):" )
    for window_size, stride, top_m in get_window_configs():
        value = np.mean( [ mean_primary( s ) for c, s in sliding_scores.items() if c[ 1: ] == ( window_size, stride, top_m ) ] )
        print( f"  window({ window_size })_stride({ stride })_m({ top_m }): { value:.3f}" )

    best_baseline = rank_configs( baseline_scores )[ 0 ]
    best_sliding = ranked_sliding[ 0 ]
    print( f"\nSelected baseline config: { format_config( best_baseline ) }" )
    print( f"Selected sliding window config: { format_sliding_config( best_sliding ) }" )
    report_comparison( "Train", baseline_scores[ best_baseline ], sliding_scores[ best_sliding ], train_queries )

    # the test set is only used once per method, to report the performance of its selected config
    test_query_embeddings = embedding_model.encode( [ q[ "query" ] for q in test_queries ] )
    chunks, chunk_embeddings, _ = config_chunks[ best_baseline ]
    test_baseline_scores = score_similarities( chunks, test_query_embeddings @ chunk_embeddings.T, test_queries, get_k( best_baseline[ 0 ] ) )
    chunk_config, window_size, stride, top_m = best_sliding
    chunks, _, token_embeddings = config_chunks[ chunk_config ]
    test_sliding_scores = score_sliding_window(
        chunks, token_embeddings, test_query_embeddings, test_queries, window_size, stride, top_m, get_k( chunk_config[ 0 ] ) )
    report_comparison( "Test", test_baseline_scores, test_sliding_scores, test_queries )

if __name__ == "__main__":
    tune_sliding_window( "query1" )
