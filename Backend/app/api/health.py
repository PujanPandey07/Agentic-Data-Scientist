from fastapi import APIRouter

router = APIRouter(
    prefix="",
    tags=["Health"]
)


@router.get("/")
def root():
    return {
        "message": "AI Data Scientist API is running!"
    }


@router.get("/health")
def health():
    return {
        "status": "healthy"
    }
