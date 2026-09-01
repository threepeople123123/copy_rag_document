import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


_splitter : RecursiveCharacterTextSplitter | None = None

def get_splitter()->RecursiveCharacterTextSplitter:

    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                " ",
                ""
            ]
        )
    return _splitter

async def split(content:str,filename:str)->list[Document]:
    splitter = get_splitter()
    documents = [
        Document(
            page_content=content
        )
    ]

    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] =index
        chunk.metadata["chunk_hash"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        chunk.metadata["filename"] = filename

    return chunks


