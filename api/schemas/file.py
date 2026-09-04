from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class FileResponse(BaseModel):
    object_name:str
