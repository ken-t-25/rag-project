from src.utils import get_test_query_path
import json
import random

TEST_RATIO = 0.2
SEED = 0


# This function splits a query file into a train file and a test file by shuffling the queries of each
# difficulty level and holding out TEST_RATIO of them as the test set (stratifying by level so both sets
# have the same mix of easy, hard, and unanswerable queries)
def split_queries( query_file_name, test_ratio=TEST_RATIO, seed=SEED ):
    with open( str( get_test_query_path( query_file_name ) ), "r", encoding="utf-8" ) as file:
        data = json.load( file )
    rng = random.Random( seed )

    levels = {}
    for q in data[ "queries" ]:
        levels.setdefault( q[ "level" ], [] ).append( q )

    train_queries, test_queries = [], []
    for level in sorted( levels ):
        level_queries = levels[ level ]
        rng.shuffle( level_queries )
        num_test = round( len( level_queries ) * test_ratio )
        test_queries += level_queries[ :num_test ]
        train_queries += level_queries[ num_test: ]

    for split, queries in ( ( "train", train_queries ), ( "test", test_queries ) ):
        queries.sort( key=lambda q: q[ "id" ] )
        with open( str( get_test_query_path( f"{ query_file_name }_{ split }" ) ), "w", encoding="utf-8" ) as file:
            json.dump( { "file_name": data[ "file_name" ], "queries": queries }, file, indent=4, ensure_ascii=False )
        print( f"{ split }: { len( queries ) } queries" )

if __name__ == "__main__":
    split_queries( "query1" )
