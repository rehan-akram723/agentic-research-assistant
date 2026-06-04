from typing import TypedDict, List, Optional
from src.schemas.document import Document, Citation

class ResearchState(TypedDict):
    """
    Single source of truth for all inter-agent shared state.
    Mutated progressively as it traverses the LangGraph state machine.
    """
    
    # 1. Initialization (Set by ResearchService API Layer)
    question: str
    max_results: int
    include_citations: bool

    # 2. Planner Agent Outputs
    search_queries: List[str]

    # 3. Retrieval Agent Outputs
    retrieved_docs: List[Document]

    # 4. Summarizer Agent Outputs
    summaries: List[str]

    # 5. Citation Verifier Agent Outputs
    verified_citations: List[Citation]

    # 6. Report Generator Agent Outputs
    final_report: str

    # 7. System & Observability Fields
    error: Optional[str]
    step_log: List[str]