from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.data_extractor.pdf_extractor import read_pdf
from src.data_processor.chucker import Chunker
from src.data_processor.embedding_processor import EmbeddingProcessor
from src.query_processor import QueryProcessor
from src.utils import get_embedding_path, get_chunk_path
import json
import numpy as np

file_name = "exploring_the_limits_of_transfer_learning_with_a_unified_text-to-text_transformer"

query = "Why do the authors formulate NLP problems as text-to-text tasks? What are the advantages of this approach?"

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2" # max token size: 256
)
tokenizer = embedding_model.tokenizer

chunk_file_path = get_chunk_path( file_name )
embedding_file_path = get_embedding_path( file_name )

if chunk_file_path.is_file():
    with open( str( chunk_file_path ), "r", encoding="utf-8" ) as file:
        chunks = json.load( file )
else:
    pages_info = read_pdf( file_name )
    chunker = Chunker( tokenizer )
    chunks = chunker.page_block_chunker( pages_info )
    with open( str( chunk_file_path ), "w", encoding="utf-8" ) as file:
        json.dump( chunks, file,indent=4 )

if embedding_file_path.is_file():
    embeddings = np.load( str( embedding_file_path ) )
else:
    embedding_processor = EmbeddingProcessor( embedding_model )
    embeddings = embedding_processor.process_chunks( chunks )
    np.save( str( embedding_file_path ), embeddings )

query_processor = QueryProcessor( embedding_model )
response = query_processor.process_query( query, file_name )
print( f"Response: { response }" )
