from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ChunkResponse(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    id:UUID
    title:str
    error_message:str|None=None

