from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self):
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
            strip_headers=False
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

    def chunk(self, filename, md_text):
        """Splits Markdown into structured chunks."""
        md_splits = self.markdown_splitter.split_text(md_text)
        splits = self.text_splitter.split_documents(md_splits)

        for split in splits:
            split.metadata['source'] = filename

        return splits
