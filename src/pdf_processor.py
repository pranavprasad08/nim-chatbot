import pymupdf4llm
from create_chunks import Chunker
from vector_db import VectorDatabase
from process_images import ImageProcessor

class PDFProcessor:
    def __init__(self, image_processor, chunker, vector_db):
        self.image_processor = image_processor
        self.chunker = chunker
        self.vector_db = vector_db

    def process_pdf(self, filename):
        """Checks if the document is already indexed; if not, extracts, chunks, and vectorizes it."""
        if self.vector_db.is_document_indexed(filename):
            print(f"✅ {filename} is already indexed. Skipping processing.")
            return {"message": "Document already indexed"}

        print(f"📄 Processing {filename} ...")
        md_text = self.convert_to_markdown(filename)
        splits = self.chunker.chunk(filename, md_text)
        self.vector_db.add_texts(splits)

        return {"message": "PDF processed and indexed"}

    def convert_to_markdown(self, filename):
        """Extracts text and images, converts PDF to Markdown, and gets image summaries via NIM."""
        pages = pymupdf4llm.to_markdown(filename, page_chunks=True, write_images=True, image_format='jpg', image_path='imgs/')
        md_text = ""
        for page in pages:
            text = page['text']
            text = self.image_processor.process_images(text)
            md_text += text
        return md_text
