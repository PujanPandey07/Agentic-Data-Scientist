# api/reports.py
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from schema.reports import ChartInfo, ChartsResponse

router = APIRouter(prefix="/api/runs", tags=["Reports"])


def _path_to_static_url(file_path: str) -> str:
    """Convert an on-disk path like 'outputs/charts/<id>/x.png' (as saved
    by the services) into a URL the frontend can use directly, e.g.
    '/static/charts/<id>/x.png' — relies on outputs/ being mounted at
    /static in main.py."""
    normalized = file_path.replace("\\", "/")
    if normalized.startswith("outputs/"):
        normalized = normalized[len("outputs/"):]
    return f"/static/{normalized}"


async def _get_snapshot_or_404(request: Request, thread_id: str):
    # Same aget_state pattern used in /chat — reads the checkpoint
    # without running anything. Empty .values means this thread_id
    # was never actually started via /runs.
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(
            status_code=404, detail="No run found for this thread_id.")
    return snapshot


# api/reports.py — only get_report changes, everything else stays the same

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

    return report


@router.get("/{thread_id}/report")
async def get_report(thread_id: str, request: Request):
    snapshot = await _get_snapshot_or_404(request, thread_id)
    report = snapshot.values.get("final_report")
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No report yet for this run — pipeline hasn't reached reporting.",
        )
    return {"thread_id": thread_id, "report": _convert_report_image_paths(report)}


@router.get("/{thread_id}/report/download")
async def download_report(thread_id: str, request: Request):
    snapshot = await _get_snapshot_or_404(request, thread_id)
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
async def get_charts(thread_id: str, request: Request):
    snapshot = await _get_snapshot_or_404(request, thread_id)
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
async def download_model(thread_id: str, request: Request):
    snapshot = await _get_snapshot_or_404(request, thread_id)
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
