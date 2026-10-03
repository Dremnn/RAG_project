from abc import ABC, abstractmethod
from pathlib import Path
from src.models import Document


class BaseLoader(ABC):
    """
    Abstract Base Class for all document loaders.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    @abstractmethod
    def load(self) -> list[Document]:
        """
        Load and parse the file into a list of Document objects.
        """
        pass
