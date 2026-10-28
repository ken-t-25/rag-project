# This helper function cleans a block by:
# 1) removing all website addresses (i.e. substrings starting with "http" until the next " ", \n, \r, or \t is seen)
def clean_block( block ):
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

# This helper function calculates the ratio of non-table data in a block by:
# 1) calculating the area of the block
# 2) calculating the area of the block that is covered by tables
# 3) returning the ratio of non-table area to total area
def get_non_table_ratio( block, tables ):
    x0 = block[ "x0" ]
    y0 = block[ "y0" ]
    x1 = block[ "x1" ]
    y1 = block[ "y1" ]
    block_area = ( x1 - x0 ) * ( y1 - y0 )

    def get_non_table_area( bx0, by0, bx1, by1, table_idx ):
        if table_idx >= len( tables ):
            return ( bx1 - bx0 ) * ( by1 - by0 )
        tx0, ty0, tx1, ty1 = tables[ table_idx ]
        # if the current table does not overlap with block, proceed to the next table
        # (the intervals overlap iff each one starts before the other ends, which also covers a table
        # that extends past the block on both sides)
        if not ( tx0 < bx1 and bx0 < tx1 ) or not ( ty0 < by1 and by0 < ty1 ):
            return get_non_table_area( bx0, by0, bx1, by1, table_idx + 1 )
        area = 0
        if tx0 > bx0:
            area += get_non_table_area( bx0, by0, tx0, by1, table_idx + 1 )
        if tx1 < bx1:
            area += get_non_table_area( tx1, by0, bx1, by1, table_idx + 1 )
        if ty0 > by0:
            area += get_non_table_area( max( bx0, tx0 ), by0, min( bx1, tx1 ), ty0, table_idx + 1 )
        if ty1 < by1:
            area += get_non_table_area( max( bx0, tx0 ), ty1, min( bx1, tx1 ), by1, table_idx + 1 )
        return area

    non_table_area = get_non_table_area( x0, y0, x1, y1, 0 )
    return non_table_area / block_area

# This helper function decides whether a block is tabular (table cells or figure labels) by checking that:
# 1) the block has at least MIN_TABULAR_LINES non-empty lines
# 2) at least MIN_SHORT_LINE_RATIO of those lines have at most MAX_SHORT_LINE_WORDS words
# PyMuPDF puts each cell of a table on its own line, while prose lines span the page width, so this catches
# borderless tables that page.find_tables() (which looks for ruling lines) misses
MIN_TABULAR_LINES = 3
MAX_SHORT_LINE_WORDS = 3
MIN_SHORT_LINE_RATIO = 0.75

def is_tabular_block( block ):
    lines = [ line for line in block.split( "\n" ) if line.strip() ]
    if len( lines ) < MIN_TABULAR_LINES:
        return False
    short_lines = sum( len( line.split() ) <= MAX_SHORT_LINE_WORDS for line in lines )
    return short_lines / len( lines ) >= MIN_SHORT_LINE_RATIO