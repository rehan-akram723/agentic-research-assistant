from typing import List, Optional
from pydantic import BaseModel, Field
from src.schemas.document import Citation

class ResearchResponse(BaseModel):
    """Validated API response payload returning the completed research."""
    status: str = Field(description="Execution status: 'success', 'partial', or 'error'")
    question: str = Field(description="The original requested question")
    report: str = Field(description="The final Markdown-formatted synthesized report")
    citations: List[Citation] = Field(description="List of verified citations backing the report")
    queries_used: List[str] = Field(description="Search queries generated and used by the Planner Agent")
    docs_retrieved: int = Field(description="Total number of unique documents successfully retrieved")
    processing_time_seconds: float = Field(description="Total execution time in seconds")
    error_message: Optional[str] = Field(default=None, description="Error details if status is not 'success'")

class HealthResponse(BaseModel):
    """API Health Check response payload."""
    status: str = Field(description="Current health status of the application")
    version: str = Field(description="Current application version")