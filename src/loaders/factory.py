from pathlib import Path
from typing import Optional
from src.loaders.base import BaseLoader
from src.loaders.text_loader import TextLoader
from src.loaders.pdf_loader import PDFLoader
from src.loaders.docx_loader import DocxLoader
from src.models import Document


SUPPORTED_EXTENSIONS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
}


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
        return loader.load()

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
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    loaded = cls.load_file(file_path, **kwargs)
                    documents.extend(loaded)
                except Exception as e:
                    print(f"[Warning] Failed to load {file_path.name}: {e}")

        return documents
