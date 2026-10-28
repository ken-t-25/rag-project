from pathlib import Path

def get_embedding_path( file_name ):
    return Path( str( Path.cwd() ) + "/saved/embeddings/" + file_name + ".npy")

def get_sliding_window_embedding_path( file_name ):
    # stores the window embeddings together with the chunk index of each window
    return Path( str( Path.cwd() ) + "/saved/embeddings/" + file_name + "_sliding_window.npz" )

def get_chunk_path( file_name ):
    return Path( str( Path.cwd() ) + "/saved/chunks/" + file_name + ".json" )

def get_test_query_path( file_name ):
    return Path( str( Path.cwd() ) + "/test/data/" + file_name + ".json" )
