import fitz  
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> list[dict]:
    """
    Read a PDF file and extract text from every page.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of dicts, one per page:
        [
            {"page": 1, "text": "Patient: John Doe..."},
            {"page": 2, "text": "Lab results..."},
        ]

    Why page by page?
        Keeping page numbers lets us tell the user exactly
        which page a piece of information came from.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    logger.info(f"Loading PDF: {path.name}")

    pages = []

    # fitz.open() opens the PDF
    with fitz.open(str(path)) as doc:
        logger.info(f"PDF has {len(doc)} page(s)")

        for page_num, page in enumerate(doc, start=1):
            # get_text() extracts all text from the page as a string
            text = page.get_text()

            # Skip pages that are empty or nearly empty
            if len(text.strip()) < 20:
                logger.warning(f"Page {page_num} appears empty — skipping")
                continue

            pages.append({
                "page": page_num,
                "text": text.strip(),
            })

            logger.info(f"Page {page_num}: extracted {len(text)} characters")

    logger.info(f"Done — extracted {len(pages)} page(s) from {path.name}")
    return pages