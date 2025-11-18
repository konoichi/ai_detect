from fastapi import APIRouter, UploadFile, File
from ..services.image_detector import analyze_image

router = APIRouter()

@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    data = await file.read()
    return analyze_image(data)
