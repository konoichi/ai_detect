from pydantic import BaseModel
from typing import List, Optional

class DetectionResponse(BaseModel):
    is_ai_probability: float
    warnings: List[str]
    dimensions: dict
