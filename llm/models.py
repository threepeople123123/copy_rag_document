from typing import Type
from xml.dom import ValidationErr

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph.state import CompiledStateGraph

from core.config import settings

_model : BaseChatModel|None = None

def get_chat_model():

    global _model
    if  _model is not None:
        return _model
    _model = ChatOpenAI(
        model=settings.max_model,
        temperature=0.1,
        max_tokens=1000,
        timeout=30,
        base_url=settings.chat_base_url,
        api_key=settings.api_key
        # ...（其他参数）
    )

    return _model

_agent : None | CompiledStateGraph = None


def get_agent():
    global _agent
    if not _agent:
        _agent = create_agent(model=get_chat_model())
    return _agent


_embeddings: Embeddings | None = None

def get_embeddings()->Embeddings:
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    if not settings.embedding_api_key:
        raise ValidationErr(code="api_key is null",message="向量 api_key 没有配置")
    _embeddings =  OpenAIEmbeddings(
        model=settings.embedding_model_name,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        dimensions=settings.embedding_dimensions,
        chunk_size=settings.chunk_size,
        check_embedding_ctx_length=False
    )
    return _embeddings


# schema -> agent 缓存(进程内,每个 schema 只编译一次)
_structured_agents: dict[Type, CompiledStateGraph] = {}

def get_structured_agent(schema: Type) -> CompiledStateGraph:
    agent = _structured_agents.get(schema)
    if agent is None:
        agent = create_agent(
            model=get_chat_model(),
            response_format=ToolStrategy(schema),
        )
        _structured_agents[schema] = agent
    return agent


