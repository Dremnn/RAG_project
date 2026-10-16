import sys
from pathlib import Path
from typing import Optional
from src.loaders.base import BaseLoader
from src.loaders.text_loader import TextLoader
from src.loaders.pdf_loader import PDFLoader
from src.loaders.docx_loader import DocxLoader
from src.loaders.excel_loader import ExcelLoader
from src.models import Document


SUPPORTED_EXTENSIONS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    ".xlsx": ExcelLoader,
    ".xlsm": ExcelLoader,
}


from src.loaders.metadata import extract_metadata, strip_metadata_header


class DocumentLoaderFactory:
    """
    Factory to auto-detect file types and load documents.
    """

    @staticmethod
    def get_loader(file_path: str | Path, **kwargs) -> BaseLoader:
        path = Path(file_path)
        ext = path.suffix.lower()

        loader_cls = SUPPORTED_EXTENSIONS.get(ext)
        if not loader_cls:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported formats: {supported}"
            )

        return loader_cls(path, **kwargs)

    @classmethod
    def load_file(cls, file_path: str | Path, **kwargs) -> list[Document]:
        loader = cls.get_loader(file_path, **kwargs)
        docs = loader.load()
        meta = extract_metadata(file_path)

        for doc in docs:
            # Cat header khoi content neu chua cat
            doc.content = strip_metadata_header(doc.content)

            # Dong bo toan bo metadata tu extract_metadata
            for k, v in meta.items():
                if k != "cleaned_content":
                    doc.metadata[k] = v

            if meta.get("doc_id"):
                if "_p" in doc.doc_id and doc.metadata.get("page_number"):
                    doc.doc_id = f"{meta['doc_id']}_p{doc.metadata['page_number']}"
                else:
                    doc.doc_id = meta["doc_id"]
            if meta.get("title"):
                doc.title = meta["title"]

        return docs

    @classmethod
    def load_directory(
        cls,
        dir_path: str | Path,
        recursive: bool = True,
        **kwargs
    ) -> list[Document]:
        """
        Load all supported documents from a directory.
        """
        folder = Path(dir_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Directory not found: {folder}")

        documents: list[Document] = []
        pattern = "**/*" if recursive else "*"

        for file_path in folder.glob(pattern):
            if (
                file_path.is_file()
                and not file_path.name.startswith("~")
                and not file_path.name.startswith(".")
                and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                try:
                    loaded = cls.load_file(file_path, **kwargs)
                    documents.extend(loaded)
                except Exception as e:
                    sys.stderr.write(f"[Warning] Failed to load {file_path.name}: {e}\n")

        return documents
