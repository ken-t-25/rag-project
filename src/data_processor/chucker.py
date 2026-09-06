MAX_CHUNK_TOKENS = 240
OVERLAP_TOKENS = 40
MIN_CHUNK_TOKENS = 80
MIN_BLOCK_SIZE = 5

class Chunker():

    def __init__( self, tokenizer, max_chunk_tokens=MAX_CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS, min_chunk_tokens=MIN_CHUNK_TOKENS ):
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    # This helper function cleans a block by:
    # 1) removing all website addresses (i.e. substrings starting with "http" until the next " ", \n, \r, or \t is seen)
    def clean_block( self, block ):
        site_idx = block.find( "http", 0 )
        while site_idx > 0:
            potential_end_idx = [ block.find( " ", site_idx  ),
                                  block.find( "\n", site_idx  ),
                                  block.find( "\r", site_idx  ),
                                  block.find( "\t", site_idx  ) ]
            potential_end_idx = [ i for i in potential_end_idx if i != -1 ]
            end_site_idx = min( potential_end_idx ) if potential_end_idx else len( block )
            block = block[ :site_idx ] + block[ end_site_idx + 1: ]
            site_idx = block.find( "http", site_idx )
        return block

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
                cleaned_block = self.clean_block( block)
                if len( cleaned_block ) < MIN_BLOCK_SIZE:
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
