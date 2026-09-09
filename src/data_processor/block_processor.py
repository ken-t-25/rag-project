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
        if not ( bx0 < tx0 < bx1 or bx0 < tx1 < bx1 ) or not ( by0 < ty0 < by1 or by0 < ty1 < by1 ):
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