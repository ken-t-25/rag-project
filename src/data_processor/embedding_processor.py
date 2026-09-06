import numpy as np

class EmbeddingProcessor:
    def __init__( self, embedding_model ):
        self.embedding_model = embedding_model

    def process_chunks( self, chunks ):
        embedding_dimension = self.embedding_model.get_embedding_dimension()
        embeddings = np.zeros( ( len( chunks ), embedding_dimension ) )
        for i, chunk in enumerate( chunks ):
            embeddings[ i ] = self.embedding_model.encode( chunk[ "text" ] )
        return embeddings