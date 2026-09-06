from pathlib import Path

def get_embedding_path( file_name ):
    return Path( str( Path.cwd() ) + "/saved/embeddings/" + file_name + ".npy")

def get_chunk_path( file_name ):
    return Path( str( Path.cwd() ) + "/saved/chunks/" + file_name + ".json" )

def get_test_query_path( file_name ):
    return Path( str( Path.cwd() ) + "/test/data/" + file_name + ".json" )
