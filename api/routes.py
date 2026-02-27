"""FastAPI route handlers."""

from __future__ import annotations

from typing import Literal, get_args

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agentic_graph_rag.service import TOOL_NAMES
from api.deps import get_service

router = APIRouter(prefix="/api/v1")

VALID_MODES = Literal[
    "vector", "cypher", "hybrid",
    "agent_pattern", "agent_llm", "agent_mangle",
]
VALID_TOOLS = Literal[
    "vector_search", "cypher_traverse",
    "hybrid_search", "temporal_query", "comprehensive_search",
    "full_document_read",
]

# Runtime guard: routes must stay in sync with the service tool list.
assert set(get_args(VALID_TOOLS)) == set(TOOL_NAMES), (
    f"VALID_TOOLS {set(get_args(VALID_TOOLS))} != TOOL_NAMES {set(TOOL_NAMES)}"
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    mode: VALID_MODES = "agent_pattern"


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    tool: VALID_TOOLS = "vector_search"


class GeologySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    language: str = ""
    software_product: str = ""
    geology_subdomain: str = ""
    source_type: str = ""


class FeatureRequest(BaseModel):
    feature_name: str = Field(..., min_length=1, max_length=1000)
    software_product: str = "Micromine"


class CompareRequest(BaseModel):
    feature: str = Field(..., min_length=1, max_length=1000)
    products: list[str] = Field(default_factory=lambda: ["Micromine", "Surpac", "ГЕОМИКС"])


class StandardRequest(BaseModel):
    standard_name: str = Field(..., min_length=1, max_length=1000)


class ListSourcesRequest(BaseModel):
    source_type: str = ""
    software_product: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    svc = get_service()
    return svc.health()


@router.post("/query")
def query(req: QueryRequest):
    svc = get_service()
    qa = svc.query(req.text, mode=req.mode)
    return qa.model_dump()


@router.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    svc = get_service()
    trace = svc.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.model_dump()


@router.post("/search")
def search(req: SearchRequest):
    svc = get_service()
    results = svc.search(req.text, tool=req.tool)
    return [r.model_dump() for r in results]


@router.get("/graph/stats")
def graph_stats():
    svc = get_service()
    return svc.graph_stats()


@router.get("/metrics")
def metrics():
    from api.middleware import get_metrics
    return get_metrics().snapshot()


# ---------------------------------------------------------------------------
# Geology domain endpoints
# ---------------------------------------------------------------------------

@router.post("/geology/search")
def geology_search(req: GeologySearchRequest):
    svc = get_service()
    results = svc.search_geology_docs(
        req.query,
        language=req.language,
        software_product=req.software_product,
        geology_subdomain=req.geology_subdomain,
        source_type=req.source_type,
    )
    return [r.model_dump() for r in results]


@router.post("/geology/feature")
def geology_feature(req: FeatureRequest):
    svc = get_service()
    results = svc.get_software_feature(req.feature_name, req.software_product)
    return [r.model_dump() for r in results]


@router.post("/geology/compare")
def geology_compare(req: CompareRequest):
    svc = get_service()
    result_map = svc.compare_implementations(req.feature, req.products)
    return {
        product: [r.model_dump() for r in results]
        for product, results in result_map.items()
    }


@router.post("/geology/standard")
def geology_standard(req: StandardRequest):
    svc = get_service()
    results = svc.get_standard(req.standard_name)
    return [r.model_dump() for r in results]


@router.get("/geology/sources")
def geology_sources(source_type: str = "", software_product: str = ""):
    svc = get_service()
    return svc.list_sources(source_type, software_product)
