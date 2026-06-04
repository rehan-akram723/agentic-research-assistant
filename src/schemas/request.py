from pydantic import BaseModel, Field

class ResearchRequest(BaseModel):
    """Validated API request payload for initiating a research task."""
    question: str = Field(
        ..., 
        min_length=10, 
        max_length=500, 
        description="The natural language research question to investigate"
    )
    max_results: int = Field(
        default=10, 
        ge=1, 
        le=30, 
        description="Maximum number of documents to retrieve across all sources"
    )
    include_citations: bool = Field(
        default=True, 
        description="Whether to route through the citation verification agent"
    )
    language: str = Field(
        default="en", 
        description="Target language for the generated final report"
    )