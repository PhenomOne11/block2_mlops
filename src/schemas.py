from typing import List, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    image_b64: Optional[str]=Field(
        default=None,
        description="Base64 encoded image string"
    )
    tensor: Optional[List[List[List[float]]]]=Field(
        default=None,
        description="Raw normalized image array of shape 3x32x32"
    )
    
class ClassProbability(BaseModel):
    class_id:int
    class_name: str
    confidence: float
    
class PredictResponse(BaseModel):
    class_id:int
    class_name:str
    confidence: float
    top_3: List[ClassProbability]
    latency_ms: float
    batched_size: int
