"""
Document Parser - Text Extraction from Various File Formats

This module handles the extraction of raw text from uploaded documents.
Currently supports: PDF, TXT
Designed for easy extension to other formats (DOCX, HTML, etc.)

Why use dedicated parsers instead of raw file reading?
- PDFs have complex structure (fonts, images, tables)
- Encoding issues are common in text files
- Page boundaries need to be preserved for citations
- Error handling differs by format
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from pypdf import PdfReader
import pdfplumber

from app.utils.exceptions import DocumentProcessingError
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    """
    Represents a single page of extracted text.
    
    Keeping page-level granularity enables:
    - Accurate page number citations
    - Better context for multi-page documents
    - Debugging extraction issues
    """
    page_number: int
    content: str
    char_count: int


@dataclass
class ParsedDocument:
    """
    Complete parsed document with all pages and metadata.
    
    This is the output of the parsing stage, ready for chunking.
    """
    filename: str
    file_type: str
    total_pages: int
    pages: List[ParsedPage]
    total_chars: int
    
    @property
    def full_text(self) -> str:
        """Concatenate all pages into single text."""
        return "\n\n".join(page.content for page in self.pages)


class DocumentParser:
    """
    Unified document parser supporting multiple file formats.
    
    Design decisions:
    - Factory pattern for format-specific parsing
    - Consistent output format (ParsedDocument) regardless of input
    - Detailed error messages for debugging
    
    Usage:
        parser = DocumentParser()
        doc = parser.parse("/path/to/document.pdf")
        print(doc.total_pages, doc.full_text)
    """
    
    SUPPORTED_FORMATS = {".pdf", ".txt"}
    
    def __init__(self):
        """Initialize parser with format handlers."""
        self._handlers = {
            ".pdf": self._parse_pdf,
            ".txt": self._parse_txt,
        }
    
    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a document and extract text.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ParsedDocument with extracted text and metadata
            
        Raises:
            DocumentProcessingError: If parsing fails
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise DocumentProcessingError(
                f"File not found: {file_path}",
                details={"file_path": str(file_path)}
            )
        
        # Get file extension
        extension = file_path.suffix.lower()
        
        # Validate format is supported
        if extension not in self.SUPPORTED_FORMATS:
            raise DocumentProcessingError(
                f"Unsupported file format: {extension}",
                details={
                    "file_path": str(file_path),
                    "extension": extension,
                    "supported": list(self.SUPPORTED_FORMATS)
                }
            )
        
        logger.info(
            "Parsing document",
            file_path=str(file_path),
            format=extension
        )
        
        # Dispatch to appropriate handler
        handler = self._handlers[extension]
        return handler(file_path)
    
    def _parse_pdf(self, file_path: Path) -> ParsedDocument:
        """
        Extract text from PDF with a two-tier strategy:
        1. Primary: pypdf with per-page error resilience
        2. Fallback: pdfplumber (uses pdfminer engine) if pypdf fails entirely

        Per-page resilience ensures that a single problematic page
        (e.g. unresolved IndirectObject references) doesn't prevent
        extraction of the remaining pages.
        """
        pages = self._try_parse_pdf_pypdf(file_path)

        if pages is None:
            logger.warning(
                "pypdf failed entirely, falling back to pdfplumber",
                file_path=str(file_path)
            )
            pages = self._try_parse_pdf_pdfplumber(file_path)

        if pages is None:
            raise DocumentProcessingError(
                "Failed to parse PDF with both pypdf and pdfplumber",
                details={"file_path": str(file_path)}
            )

        total_chars = sum(p.char_count for p in pages)

        if total_chars == 0:
            logger.warning(
                "PDF appears to be empty or image-only",
                file_path=str(file_path)
            )

        logger.info(
            "PDF parsed successfully",
            file_path=str(file_path),
            pages=len(pages),
            total_chars=total_chars
        )

        return ParsedDocument(
            filename=file_path.name,
            file_type="pdf",
            total_pages=len(pages),
            pages=pages,
            total_chars=total_chars
        )

    def _try_parse_pdf_pypdf(self, file_path: Path) -> Optional[List[ParsedPage]]:
        """
        Attempt PDF text extraction with pypdf.
        Returns None if the reader cannot open the file at all.
        Individual page failures are logged and skipped so the
        remaining pages can still be extracted.
        """
        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            logger.warning(
                "pypdf could not open PDF",
                file_path=str(file_path),
                error=str(e)
            )
            return None

        pages: List[ParsedPage] = []
        failed_pages: List[int] = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                text = self._clean_text(text)
            except Exception as e:
                logger.warning(
                    "pypdf failed on page, skipping",
                    file_path=str(file_path),
                    page=page_num,
                    error=str(e)
                )
                failed_pages.append(page_num)
                text = ""

            pages.append(ParsedPage(
                page_number=page_num,
                content=text,
                char_count=len(text)
            ))

        if failed_pages:
            logger.info(
                "Retrying failed pages with pdfplumber",
                file_path=str(file_path),
                failed_pages=failed_pages
            )
            self._patch_failed_pages_with_pdfplumber(file_path, pages, failed_pages)

        return pages

    def _patch_failed_pages_with_pdfplumber(
        self,
        file_path: Path,
        pages: List[ParsedPage],
        failed_page_numbers: List[int]
    ) -> None:
        """
        Re-extract only the pages that pypdf couldn't handle
        using pdfplumber, and patch them in-place.
        """
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num in failed_page_numbers:
                    idx = page_num - 1
                    if idx < len(pdf.pages):
                        try:
                            text = pdf.pages[idx].extract_text() or ""
                            text = self._clean_text(text)
                            pages[idx] = ParsedPage(
                                page_number=page_num,
                                content=text,
                                char_count=len(text)
                            )
                            logger.info(
                                "pdfplumber recovered page",
                                file_path=str(file_path),
                                page=page_num
                            )
                        except Exception as e:
                            logger.warning(
                                "pdfplumber also failed on page",
                                file_path=str(file_path),
                                page=page_num,
                                error=str(e)
                            )
        except Exception as e:
            logger.warning(
                "pdfplumber could not open PDF for page patching",
                file_path=str(file_path),
                error=str(e)
            )

    def _try_parse_pdf_pdfplumber(self, file_path: Path) -> Optional[List[ParsedPage]]:
        """
        Full-document fallback using pdfplumber (pdfminer engine).
        Used when pypdf cannot open the file at all.
        """
        try:
            pages: List[ParsedPage] = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                        text = self._clean_text(text)
                    except Exception as e:
                        logger.warning(
                            "pdfplumber failed on page",
                            file_path=str(file_path),
                            page=page_num,
                            error=str(e)
                        )
                        text = ""

                    pages.append(ParsedPage(
                        page_number=page_num,
                        content=text,
                        char_count=len(text)
                    ))
            return pages
        except Exception as e:
            logger.error(
                "pdfplumber parsing failed",
                file_path=str(file_path),
                error=str(e)
            )
            return None
    
    def _parse_txt(self, file_path: Path) -> ParsedDocument:
        """
        Extract text from plain text file.
        
        Handles encoding detection and normalization.
        Text files are treated as single-page documents.
        """
        # Try common encodings
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        content: Optional[str] = None
        
        for encoding in encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise DocumentProcessingError(
                "Could not decode text file with any supported encoding",
                details={
                    "file_path": str(file_path),
                    "tried_encodings": encodings
                }
            )
        
        # Clean the text
        content = self._clean_text(content)
        
        # Text files are single-page
        pages = [ParsedPage(
            page_number=1,
            content=content,
            char_count=len(content)
        )]
        
        logger.info(
            "Text file parsed successfully",
            file_path=str(file_path),
            chars=len(content)
        )
        
        return ParsedDocument(
            filename=file_path.name,
            file_type="txt",
            total_pages=1,
            pages=pages,
            total_chars=len(content)
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving structure.
        
        Operations:
        - Normalize whitespace (multiple spaces -> single)
        - Preserve paragraph breaks (double newlines)
        - Remove control characters
        - Strip leading/trailing whitespace
        """
        # Replace tabs and multiple spaces with single space
        import re
        text = re.sub(r"[ \t]+", " ", text)
        
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Collapse multiple newlines to max 2 (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Remove control characters except newlines
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        
        return text.strip()
    
    def is_supported(self, file_path: str | Path) -> bool:
        """Check if a file format is supported."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_FORMATS
