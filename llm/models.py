from typing import Type
from xml.dom import ValidationErr

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
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

# schema -> agent 缓存(进程内,每个 schema 只编译一次)
_max_structured_agents: dict[Type, Runnable] = {}

def get_max_structured_agent(schema: Type) -> Runnable:
    agent = _max_structured_agents.get(schema)
    if agent is None:
        # 直接用 with_structured_output 做单次结构化调用，
        # 避免 create_agent + ToolStrategy 的 agent 重试循环。
        agent = get_chat_model_for_qwen3_8_max().with_structured_output(schema)
        _max_structured_agents[schema] = agent
    return agent

_max_model: BaseChatModel | None = None

def get_chat_model_for_qwen3_8_max():

    global _max_model
    if  _max_model is not None:
        return _max_model
    _max_model = ChatOpenAI(
        model="qwen3.7-flash",
        temperature=0.1,
        max_tokens=100000,
        # 生成完整测试用例（步骤 + 功能/边界/异常/安全四维）输出很长、很慢，
        # 30s 会频繁超时，这里放宽到 5 分钟。
        timeout=300,
        max_retries=2,
        base_url=settings.chat_base_url,
        api_key=settings.api_key
        # ...（其他参数）
    )

    return _max_model

