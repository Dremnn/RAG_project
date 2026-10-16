from pathlib import Path
from src.loaders.base import BaseLoader
from src.models import Document


from src.loaders.metadata import extract_metadata, strip_metadata_header


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

        meta = extract_metadata(self.file_path, content=content)
        cleaned_content = strip_metadata_header(content)

        metadata = {
            "file_name": self.file_path.name,
            "extension": self.file_path.suffix.lower(),
            "file_size_bytes": self.file_path.stat().st_size,
        }
        for k, v in meta.items():
            if k != "cleaned_content":
                metadata[k] = v

        doc = Document(
            content=cleaned_content,
            doc_id=meta.get("doc_id", self.file_path.stem),
            title=meta.get("title", self.file_path.stem),
            source=str(self.file_path.resolve()),
            metadata=metadata,
        )
        return [doc]
