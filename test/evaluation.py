import re

NGRAM_SIZE = 4
MIN_PASSAGE_OVERLAP = 0.5


# This helper function returns the set of word n-grams in a text after:
# 1) lowercasing it (the tokenizer's decode lowercases text that was split across chunks)
# 2) keeping only alphanumeric words (so punctuation, hyphenation, and line breaks are ignored)
def get_ngrams( text, n=NGRAM_SIZE ):
    words = re.findall( r"[a-z0-9]+", text.lower() )
    return { tuple( words[ i : i + n ] ) for i in range( len( words ) - n + 1 ) }

def evaluate_chunk( chunk, expected_passages ):
    # return True if the retrieved chunk covers enough of any expected passage, where coverage is
    # the fraction of shared n-grams relative to the smaller of the passage and the chunk
    # (so a short chunk that lies entirely inside a long passage still counts)
    if not ( chunk_ngrams := get_ngrams( chunk[ "text" ] ) ):
        return False
    for passage in expected_passages:
        if not chunk[ "start_page" ] <= passage[ "page" ] <= chunk[ "end_page" ]:
            continue
        passage_ngrams = get_ngrams( passage[ "text" ] )
        overlap = len( chunk_ngrams & passage_ngrams ) / min( len( chunk_ngrams ), len( passage_ngrams ) )
        if overlap >= MIN_PASSAGE_OVERLAP:
            return True
    return False

def score_retrieval( retrieved_chunks, expected_passages ):
    # return the retrieval metrics of one query:
    # - passage_recall: fraction of expected passages found in at least one retrieved chunk
    # - hit: 1 if any retrieved chunk is relevant, else 0
    # - mrr: reciprocal rank of the first relevant chunk (0 if none)
    # - precision: fraction of retrieved chunks that are relevant
    relevant = [ evaluate_chunk( chunk, expected_passages ) for chunk in retrieved_chunks ]
    found_passages = sum(
        any( evaluate_chunk( chunk, [ passage ] ) for chunk in retrieved_chunks )
        for passage in expected_passages
    )
    first_rank = next( ( i + 1 for i, r in enumerate( relevant ) if r ), None )
    return {
        "passage_recall": found_passages / len( expected_passages ),
        "hit": float( first_rank is not None ),
        "mrr": 1 / first_rank if first_rank else 0.0,
        "precision": sum( relevant ) / len( retrieved_chunks ) if retrieved_chunks else 0.0,
    }
