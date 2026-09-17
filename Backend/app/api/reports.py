# api/reports.py
import math
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import get_db_session
from core.security import get_current_user_id
from schema.reports import ChartInfo, ChartsResponse
from api.chat import _get_owned_conversation
from utilis.sanitize import sanitize_for_json

router = APIRouter(prefix="/api/runs", tags=["Reports"])


def _sanitize_nans(obj):
    """Recursively replace NaN/inf float values with None, since Python's
    json.dumps (via Starlette's JSONResponse) rejects them outright —
    NaN/Infinity aren't valid JSON, even though Python's own float type
    allows them. Without this, any report containing a NaN (e.g. a mean
    computed over an all-missing column) crashes the endpoint with a 500."""
    obj = sanitize_for_json(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(v) for v in obj]
    return obj


def _path_to_static_url(file_path: str) -> str:
    """Convert an on-disk path like 'outputs/charts/<id>/x.png' (as saved
    by the services) into a URL the frontend can use directly, e.g.
    '/static/charts/<id>/x.png' — relies on outputs/ being mounted at
    /static in main.py."""
    normalized = file_path.replace("\\", "/")
    if normalized.startswith("outputs/"):
        normalized = normalized[len("outputs/"):]
    return f"/static/{normalized}"


def _convert_report_image_paths(report: dict) -> dict:
    """Rewrite every known raw disk-path field inside a final_report into
    a frontend-usable /static/... URL, so callers never need to know our
    internal folder layout. Mutates a shallow copy, not the original.
    """
    report = dict(report)

    if report.get("report_path"):
        report["report_path"] = _path_to_static_url(report["report_path"])

    visualization = report.get("visualization")
    if visualization and visualization.get("charts"):
        visualization = dict(visualization)
        visualization["charts"] = [
            {**chart, "path": _path_to_static_url(chart["path"])}
            if chart.get("path") else chart
            for chart in visualization["charts"]
        ]
        report["visualization"] = visualization

    evaluation = report.get("evaluation")
    if evaluation and evaluation.get("artifacts"):
        evaluation = dict(evaluation)
        artifacts = dict(evaluation["artifacts"])
        for key in (
            "confusion_matrix_path", "residual_plot_path",
            "actual_vs_predicted_path", "learning_curve_path",
            "boosting_curve_path",
        ):
            if artifacts.get(key):
                artifacts[key] = _path_to_static_url(artifacts[key])
        evaluation["artifacts"] = artifacts
        report["evaluation"] = evaluation

    # Must run LAST — after image paths are converted — since it walks
    # the whole dict recursively and doesn't care about structure, only
    # about catching any NaN/inf floats left in numeric fields (e.g.
    # EDA's numerical_summary.mean) before this ever reaches json.dumps.
    return _sanitize_nans(report)


async def _get_snapshot_for_owned_thread(
    request: Request, thread_id: str, user_id: int, session: AsyncSession,
):
    # Ownership check FIRST — confirms this thread_id belongs to the
    # requesting user (404s otherwise, same as chat/conversations).
    # Only once ownership is confirmed do we touch the checkpointer.
    await _get_owned_conversation(session, thread_id, user_id)

    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        # Shouldn't normally happen if a Conversation row exists, but
        # guards against a corrupted/missing checkpoint regardless.
        raise HTTPException(
            status_code=404, detail="No run found for this thread_id.")
    return snapshot


@router.get("/{thread_id}/report")
async def get_report(
    thread_id: str,
    request: Request,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    snapshot = await _get_snapshot_for_owned_thread(request, thread_id, user_id, session)
    report = snapshot.values.get("final_report")
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No report yet for this run — pipeline hasn't reached reporting.",
        )
    return {"thread_id": thread_id, "report": _convert_report_image_paths(report)}


@router.get("/{thread_id}/report/download")
async def download_report(
    thread_id: str,
    request: Request,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    snapshot = await _get_snapshot_for_owned_thread(request, thread_id, user_id, session)
    report = snapshot.values.get("final_report")
    if report is None or not report.get("report_path"):
        raise HTTPException(
            status_code=404, detail="No report file available yet.")

    report_path = Path(report["report_path"])
    if not report_path.is_file():
        raise HTTPException(
            status_code=404, detail="Report file missing on disk.")

    return FileResponse(
        path=report_path,
        filename=report_path.name,
        media_type="application/json",
    )


@router.get("/{thread_id}/charts", response_model=ChartsResponse)
async def get_charts(
    thread_id: str,
    request: Request,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    snapshot = await _get_snapshot_for_owned_thread(request, thread_id, user_id, session)
    results = snapshot.values.get("visualization_results") or []

    charts = [
        ChartInfo(
            chart_type=item.get("chart_type"),
            x_column=item.get("x_column"),
            y_column=item.get("y_column"),
            group_by=item.get("group_by"),
            url=_path_to_static_url(item["path"]),
        )
        for item in results
        if item.get("path")
    ]

    return ChartsResponse(thread_id=thread_id, charts=charts)


@router.get("/{thread_id}/model/download")
async def download_model(
    thread_id: str,
    request: Request,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    snapshot = await _get_snapshot_for_owned_thread(request, thread_id, user_id, session)
    model_path_str = snapshot.values.get(
        "tuned_model_path") or snapshot.values.get("trained_model_path")

    if not model_path_str:
        raise HTTPException(
            status_code=404, detail="No trained model available for this run yet.")

    model_path = Path(model_path_str)
    if not model_path.is_file():
        raise HTTPException(
            status_code=404, detail="Model file missing on disk.")

    return FileResponse(
        path=model_path,
        filename=model_path.name,
        media_type="application/octet-stream",
    )
