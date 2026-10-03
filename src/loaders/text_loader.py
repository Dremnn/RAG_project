from pathlib import Path
from src.loaders.base import BaseLoader
from src.models import Document


class TextLoader(BaseLoader):
    """
    Loader for Plain Text (.txt) and Markdown (.md) files.
    """

    def __init__(self, file_path: str | Path, encoding: str = "utf-8"):
        super().__init__(file_path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        content = ""
        # Try primary encoding, fallback to latin-1 if decoding fails
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(self.file_path, "r", encoding="latin-1") as f:
                content = f.read()

        doc_id = f"DOC_{self.file_path.stem}"
        doc = Document(
            content=content.strip(),
            doc_id=doc_id,
            title=self.file_path.stem,
            source=str(self.file_path.resolve()),
            metadata={
                "file_name": self.file_path.name,
                "extension": self.file_path.suffix.lower(),
                "file_size_bytes": self.file_path.stat().st_size,
            },
        )
        return [doc]
