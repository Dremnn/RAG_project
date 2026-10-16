from src.loaders.base import BaseLoader
from src.loaders.text_loader import TextLoader
from src.loaders.pdf_loader import PDFLoader
from src.loaders.docx_loader import DocxLoader
from src.loaders.factory import DocumentLoaderFactory, SUPPORTED_EXTENSIONS
from src.loaders.metadata import extract_metadata, strip_metadata_header

__all__ = [
    "BaseLoader",
    "TextLoader",
    "PDFLoader",
    "ExcelLoader",
    "DocxLoader",
    "DocumentLoaderFactory",
    "SUPPORTED_EXTENSIONS",
    "extract_metadata",
    "strip_metadata_header",
]
