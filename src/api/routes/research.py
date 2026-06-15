from fastapi import APIRouter, HTTPException
from src.schemas.request import ResearchRequest
from src.schemas.response import ResearchResponse
from src.services.research_service import ResearchService

router = APIRouter(prefix="/research", tags=["research"])
service = ResearchService()


@router.post("/", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    try:
        return await service.run(request.question, request.max_results, request.include_citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status() -> dict:
    return {"status": "running"}
