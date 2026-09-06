import pymupdf
from pathlib import Path

TEXT_BLOCK_TYPE = 0

def read_pdf( file_name ):
    pdf_path = str( Path.cwd() ) + "/files/" + file_name + ".pdf"
    with pymupdf.open( pdf_path ) as pdf:
        text = []
        for page_num, page in enumerate( pdf, start=1 ):
            text.append( read_page_blocks( page, page_num ) )
    return text

def read_page( page, page_num ):
    page_info = {}
    page_info[ "page_number" ] = page_num
    page_info[ "text" ] = page.get_text()
    return page_info

def read_page_blocks( page, page_num ):
    page_info = {}
    page_info[ "page_number" ] = page_num
    page_info[ "text" ] = []
    blocks = page.get_text( "blocks" )
    for b in blocks:
        _, _, _, _, text, _, block_type = b
        if block_type == TEXT_BLOCK_TYPE:
            page_info[ "text" ].append( text )
    return page_info
