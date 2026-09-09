from src.data_processor.chucker import clean_block, get_non_table_ratio

MAX_CHUNK_TOKENS = 240
OVERLAP_TOKENS = 40
MIN_CHUNK_TOKENS = 80
MIN_BLOCK_SIZE = 5
MIN_NON_TABLE_CONTENT_RATIO = 0.5

class Chunker():

    def __init__( self, tokenizer, max_chunk_tokens=MAX_CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS, min_chunk_tokens=MIN_CHUNK_TOKENS ):
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    # This helper function processes and evaluates a block by:
    # 1) cleaning the block
    # 2) returning the cleaned block if all following conditions are met, else return None:
    #    a) block length is greater than or equal to MIN_BLOCK_SIZE
    #    b) at least 50% of the block is not table data
    def process_and_evaluate_block( self, block, tables ):
        cleaned_block = clean_block( block[ "text" ] )
        if len( cleaned_block ) < MIN_BLOCK_SIZE:
            return None
        elif get_non_table_ratio( block, tables ) < MIN_NON_TABLE_CONTENT_RATIO:
            return None
        else:
            return cleaned_block

    def page_block_chunker( self, pages_info ):
        chunks = []
        chunk_id = 0
        curr_chunk = ""
        curr_chunk_len = 0
        start_page = 1
        end_page = 1

        def complete_chunk_handler():
            nonlocal curr_chunk, curr_chunk_len, start_page, chunk_id
            if curr_chunk_len < self.min_chunk_tokens:
                return

            chunks.append( {
                "id": chunk_id,
                "start_page": start_page,
                "end_page": end_page,
                "text": curr_chunk,
            } )
            chunk_id += 1

            overlap_text = curr_chunk
            token_ids = self.tokenizer.encode( overlap_text, add_special_tokens=False )
            while len( token_ids ) > self.overlap_tokens:
                overlap_text = self.tokenizer.decode( token_ids[ -self.overlap_tokens : ], skip_special_tokens=True )
                token_ids = self.tokenizer.encode( overlap_text, add_special_tokens=False )

            curr_chunk = overlap_text
            curr_chunk_len = len( token_ids )
            start_page = end_page

        for p in pages_info:
            for block in p[ "text" ]:
                block = self.process_and_evaluate_block( block, p[ "tables" ] )
                if not block:
                    continue
                token_ids = self.tokenizer.encode( block, add_special_tokens=False )
                if len( token_ids ) + curr_chunk_len <= self.max_chunk_tokens:
                    if curr_chunk and curr_chunk[ -1 ] != " " and curr_chunk[ -1 ] != "\n":
                        curr_chunk += "\n"
                    curr_chunk += block
                    curr_chunk_len += len( token_ids )
                    end_page = p[ "page_number" ]
                    if curr_chunk_len == self.max_chunk_tokens:
                        complete_chunk_handler()
                elif curr_chunk_len >= self.min_chunk_tokens:
                    complete_chunk_handler()
                else:
                    block_start_idx = 0
                    while block_start_idx < len( token_ids ):
                        remain_chunk_len = self.max_chunk_tokens - curr_chunk_len
                        block_end_idx = min( block_start_idx + remain_chunk_len, len( token_ids ) )
                        remain_chunk = self.tokenizer.decode( token_ids[ block_start_idx : block_end_idx ], skip_special_tokens=True )
                        if remain_chunk[ : 2 ] == "##":
                            remain_chunk = remain_chunk[ 2 : ]
                        else:
                            if curr_chunk and curr_chunk[ -1 ] != " " and curr_chunk[ -1 ] != "\n":
                                curr_chunk += "\n"
                        curr_chunk += remain_chunk
                        curr_chunk_len += ( block_end_idx - block_start_idx )
                        end_page = p[ "page_number" ]
                        if curr_chunk_len == self.max_chunk_tokens:
                            complete_chunk_handler()
                        block_start_idx = block_end_idx

        if curr_chunk_len < self.min_chunk_tokens and curr_chunk_len > self.overlap_tokens:
            prev_chunk = chunks[ -1 ]
            token_ids = self.tokenizer.encode( prev_chunk[ "text" ], add_special_tokens=False )
            parts_to_include = self.tokenizer.decode( token_ids[ -( self.min_chunk_tokens - curr_chunk_len ) : ], skip_special_tokens=True )
            if parts_to_include and parts_to_include[ -1 ] != " " and parts_to_include[ -1 ] != "\n":
                parts_to_include += "\n"
            curr_chunk = parts_to_include + curr_chunk
            curr_chunk_len = self.min_chunk_tokens
            start_page = prev_chunk[ "start_page" ]
        complete_chunk_handler()
        return chunks
