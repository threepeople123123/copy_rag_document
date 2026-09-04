from abc import abstractmethod, ABC
from typing import Any


class BaseTool(ABC):

    name:str

    description:str

    @abstractmethod
    def description(self):
        raise NotImplementedError

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict:
        """统一执行入口，统一返回结构"""
        raise NotImplementedError