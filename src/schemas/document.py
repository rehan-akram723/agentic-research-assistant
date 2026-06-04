from typing import List, Optional
from pydantic import BaseModel, Field

class Document(BaseModel):
    """Internal representation of a retrieved academic document."""
    id: str = Field(description="Unique identifier for the document (e.g., arXiv ID)")
    title: str = Field(description="Title of the research paper")
    authors: List[str] = Field(description="List of author names")
    abstract: str = Field(description="Abstract or summary of the document")
    url: str = Field(description="URL to access the source document")
    published_date: Optional[str] = Field(default=None, description="Publication date of the document")
    source: str = Field(description="Origin of the document, e.g., 'arxiv' or 'semantic_scholar'")
    relevance_score: Optional[float] = Field(default=None, description="Score indicating retrieval relevance")

class Citation(BaseModel):
    """Internal representation of a verified factual claim linked to a source."""
    claim: str = Field(description="The specific factual claim extracted from a summary")
    source_doc_id: str = Field(description="ID of the Document supporting the claim")
    source_title: str = Field(description="Title of the supporting Document")
    is_verified: bool = Field(description="Boolean indicating if the source explicitly supports the claim")
    quote: Optional[str] = Field(default=None, description="Exact quote from the source supporting the claim")