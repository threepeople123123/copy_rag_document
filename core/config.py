from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECTING = Path(__file__).resolve().parents[1]




class Setting(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECTING / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    base_url:str = ""
    api_key:str = ""
    max_model:str = ""
    min_model:str = ""
    embedding_model:str = ""

    date_source_url:str= ""
    log_lever:str = ""
    app_name:str = ""

    rabbit_mq_host:str = ""
    rabbit_mq_port:int = 0
    rabbit_mq_username:int = 0
    rabbit_mq_password:int = 0

    embedding_api_key:str = "sk-0363d3e4787e4ab19253e56309e0ff95"
    # aliyun最新根据业务空间id调用，猜测是为了分流和最近节点的服务器，更快
    embedding_base_url:str = "https://llm-3l2i84ztuewo30qg.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    embedding_model_name:str = "qwen3.7-text-embedding"
    embedding_dimensions:int = 1024
    chunk_size:int = 600
    chunk_overlap:int = 60



@lru_cache(maxsize=1)
def get_settings()->Setting:
    return Setting()

settings = get_settings()



