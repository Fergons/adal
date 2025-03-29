# schemas.py
from pydantic import BaseModel
from typing import List, Union, Optional

class EmbedRequest(BaseModel):
    texts: Union[str, List[str]]
   

class EmbeddingOutput(BaseModel):
    embedding: List[float]
    index: int

class EmbedResponse(BaseModel):
    data: List[EmbeddingOutput]
    model: Optional[str] = None
    error: Optional[str] = None
    input_texts_count: Optional[int] = None 