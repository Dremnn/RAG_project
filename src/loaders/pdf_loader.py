from pathlib import Path
from pypdf import PdfReader
from src.loaders.base import BaseLoader
from src.models import Document


class PDFLoader(BaseLoader):
    """
    Loader for PDF (.pdf) documents using pypdf.
    Supports loading as a single document or split by pages.
    """

    def __init__(self, file_path: str | Path, split_pages: bool = False):
        super().__init__(file_path)
        self.split_pages = split_pages

    def load(self) -> list[Document]:
        reader = PdfReader(str(self.file_path))
        total_pages = len(reader.pages)
        docs: list[Document] = []

        if self.split_pages:
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                doc_id = f"DOC_{self.file_path.stem}_p{page_idx + 1}"
                doc = Document(
                    content=text.strip(),
                    doc_id=doc_id,
                    title=f"{self.file_path.stem} (Page {page_idx + 1})",
                    source=str(self.file_path.resolve()),
                    metadata={
                        "file_name": self.file_path.name,
                        "extension": ".pdf",
                        "page_number": page_idx + 1,
                        "total_pages": total_pages,
                    },
                )
                docs.append(doc)
        else:
            full_text_parts = []
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    full_text_parts.append(
                        f"--- [Page {page_idx + 1}] ---\n{page_text.strip()}"
                    )

            full_content = "\n\n".join(full_text_parts)
            doc_id = f"DOC_{self.file_path.stem}"
            doc = Document(
                content=full_content,
                doc_id=doc_id,
                title=self.file_path.stem,
                source=str(self.file_path.resolve()),
                metadata={
                    "file_name": self.file_path.name,
                    "extension": ".pdf",
                    "total_pages": total_pages,
                },
            )
            docs.append(doc)

        return docs
