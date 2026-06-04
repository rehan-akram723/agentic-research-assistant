class AgentError(Exception):
    """Raised when an LLM agent encounters an operational or parsing failure."""
    pass

class RetrievalError(Exception):
    """Raised when document retrieval fails completely or returns zero results."""
    pass

class CitationError(Exception):
    """Raised when the citation verification pipeline encounters a critical error."""
    pass