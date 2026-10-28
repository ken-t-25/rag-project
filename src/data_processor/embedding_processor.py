import numpy as np
import torch

class EmbeddingProcessor:
    def __init__( self, embedding_model ):
        self.embedding_model = embedding_model
        self.transformer = self.embedding_model[ 0 ]
        self.tokenizer = embedding_model.tokenizer

    def process_chunks( self, chunks ):
        embedding_dimension = self.embedding_model.get_embedding_dimension()
        embeddings = np.zeros( ( len( chunks ), embedding_dimension ) )
        for i, chunk in enumerate( chunks ):
            embeddings[ i ] = self.embedding_model.encode( chunk[ "text" ] )
        return embeddings

    # This function embeds chunks with a sliding window by:
    # 1) contextualizing each whole chunk with the transformer
    # 2) mean-pooling the contextualized tokens of each window of window_size tokens, moving the window
    #    stride tokens at a time, and normalizing the result
    # It returns the window embeddings and, for each window, the index of the chunk it belongs to
    # (a chunk gets more windows the longer it is)
    def process_chunks_sliding_window( self, chunks, window_size=80, stride=40 ):
        token_embeddings = self.get_contextualized_token_embeddings( chunks )
        return pool_windows( token_embeddings, window_size, stride )

    # This helper function returns, for each chunk, its contextualized token embeddings
    # (num_tokens x embedding_dimension), excluding the [CLS] and [SEP] tokens
    def get_contextualized_token_embeddings( self, chunks, batch_size=32 ):
        token_embeddings = []
        for batch_start in range( 0, len( chunks ), batch_size ):
            texts = [ chunk[ "text" ] for chunk in chunks[ batch_start : batch_start + batch_size ] ]
            features = self.transformer.preprocess( texts )
            features = {
                key: value.to( self.embedding_model.device )
                for key, value in features.items()
                if hasattr( value, "to" )
            }
            with torch.no_grad():
                output = self.transformer( features )
            for embeddings, attention_mask in zip( output[ "token_embeddings" ], features[ "attention_mask" ] ):
                num_tokens = int( attention_mask.sum() )
                token_embeddings.append( embeddings[ 1 : num_tokens - 1 ].cpu().numpy() )
        return token_embeddings

# This helper function returns the start index of each window, such that:
# 1) windows start every stride tokens
# 2) a final window ending at the last token is added if the windows would otherwise not cover it
# 3) a chunk shorter than window_size gets a single window over all its tokens
def get_window_starts( num_tokens, window_size, stride ):
    starts = list( range( 0, max( num_tokens - window_size, 0 ) + 1, stride ) )
    if starts[ -1 ] + window_size < num_tokens:
        starts.append( num_tokens - window_size )
    return starts

def pool_windows( token_embeddings, window_size, stride ):
    window_embeddings = []
    window_chunk_ids = []
    for chunk_id, tokens in enumerate( token_embeddings ):
        for start in get_window_starts( len( tokens ), window_size, stride ):
            window = tokens[ start : start + window_size ].mean( axis=0 )
            window_embeddings.append( window / np.linalg.norm( window ) )
            window_chunk_ids.append( chunk_id )
    return np.array( window_embeddings ), np.array( window_chunk_ids )

# This function scores chunks from their window similarities by averaging the top_m window similarities
# of each chunk (or all of them if the chunk has at most top_m windows)
# window_similarities: num_queries x num_windows, where window_chunk_ids gives the chunk of each window
# returns: num_queries x num_chunks
def get_chunk_similarities( window_similarities, window_chunk_ids, num_chunks, top_m ):
    window_counts = np.bincount( window_chunk_ids, minlength=num_chunks )
    # place the windows of each chunk side by side (padding with -inf) so the top_m can be taken per chunk
    chunk_offsets = np.concatenate( ( [ 0 ], np.cumsum( window_counts )[ :-1 ] ) )
    window_positions = np.arange( len( window_chunk_ids ) ) - chunk_offsets[ window_chunk_ids ]
    padded = np.full( ( window_similarities.shape[ 0 ], num_chunks, window_counts.max() ), -np.inf )
    padded[ :, window_chunk_ids, window_positions ] = window_similarities
    top = -np.sort( -padded, axis=2 )[ :, :, :top_m ]
    top_counts = np.minimum( window_counts, top_m )
    return np.where( np.isinf( top ), 0, top ).sum( axis=2 ) / top_counts
