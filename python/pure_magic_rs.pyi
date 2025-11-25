from typing import List, Optional, Dict, Any

class Magic:
    """Represents a detected file type's "magic" information.

    Attributes:
        source (Optional[str]): The source of the magic detection.
        message (str): A human-readable description of the detected type.
        mime_type (str): The MIME type of the detected file.
        creator_code (Optional[str]): The creator code, if available.
        strength (int): The strength of the detection.
        extensions (List[str]): Possible file extensions for the detected type.
    """

    @property
    def source(self) -> Optional[str]: ...
    @property
    def message(self) -> str: ...
    @property
    def mime_type(self) -> str: ...
    @property
    def creator_code(self) -> Optional[str]: ...
    @property
    def strength(self) -> int: ...
    @property
    def extensions(self) -> List[str]: ...
    def to_dict(self) -> Dict[str, Any]:
        """Convert this `Magic` instance into a Python dictionary.

        Returns:
            dict: A dictionary with keys `source`, `message`, `mime_type`,
                  `creator_code`, `strength`, and `extensions`.

        Example:
            >>> magic_dict = magic_instance.to_dict()
            >>> print(magic_dict["mime_type"])
        """
        ...

class MagicDb:
    """A file type detection database using magic numbers.

    This class provides methods to detect file types for both in-memory buffers
    and files on disk.

    Example:
        >>> db = MagicDb()
        >>> result = db.first_magic_file("example.txt")
        >>> print(result.mime_type)
    """

    def __init__(self) -> None: ...
    def first_magic_buffer(self, input: bytes, extension: Optional[str] = ...) -> Magic:
        """Detect the first magic match for an in-memory buffer.

        Args:
            input (bytes): The buffer to analyze.
            extension (Optional[str]): Optional file extension hint.

        Returns:
            Magic: The first detected magic result.

        Raises:
            ValueError: If magic identification failed

        Example:
            >>> with open("example.txt", "rb") as f:
            ...     buffer = f.read()
            >>> result = db.first_magic_buffer(buffer, "txt")
        """
        ...

    def first_magic_file(self, path: str) -> Magic:
        """Detect the first magic match for a file.

        Args:
            path (str): Path to the file to analyze.

        Returns:
            Magic: The first detected magic result.

        Raises:
            IOError: If the file cannot be opened.
            ValueError: If magic identification failed

        Example:
            >>> result = db.first_magic_file("example.txt")
        """
        ...

    def best_magic_buffer(self, input: bytes) -> Magic:
        """Detect the best magic match for an in-memory buffer.

        Args:
            input (bytes): The buffer to analyze.

        Returns:
            Magic: The best detected magic result.

        Raises:
            ValueError: If magic identification failed

        Example:
            >>> with open("example.txt", "rb") as f:
            ...     buffer = f.read()
            >>> result = db.best_magic_buffer(buffer)
        """
        ...

    def best_magic_file(self, path: str) -> Magic:
        """Detect the best magic match for a file.

        Args:
            path (str): Path to the file to analyze.

        Returns:
            Magic: The best detected magic result.

        Raises:
            IOError: If the file cannot be opened.
            ValueError: If magic identification failed

        Example:
            >>> result = db.best_magic_file("example.txt")
        """
        ...

    def all_magics_buffer(self, input: bytes) -> List[Magic]:
        """Detect all magic matches for an in-memory buffer.

        Args:
            input (bytes): The buffer to analyze.

        Returns:
            List[Magic]: All detected magic results.

        Raises:
            ValueError: If magic identification failed

        Example:
            >>> with open("example.txt", "rb") as f:
            ...     buffer = f.read()
            >>> results = db.all_magics_buffer(buffer)
        """
        ...

    def all_magics_file(self, path: str) -> List[Magic]:
        """Detect all magic matches for a file.

        Args:
            path (str): Path to the file to analyze.

        Returns:
            List[Magic]: All detected magic results.

        Raises:
            IOError: If the file cannot be opened.
            ValueError: If magic identification failed

        Example:
            >>> results = db.all_magics_file("example.txt")
        """
        ...
