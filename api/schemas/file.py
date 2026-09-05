from pydantic import BaseModel


class FileResponse(BaseModel):
    object_name:str
