import os

from langfuse import get_client, Langfuse

from core.config import settings
from core.logging import get_logger

os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_base_url


_langfuse:  None | Langfuse = None

def _get_langfuse_client():
    global _langfuse
    if _langfuse is None:
        _langfuse = get_client()
    return _langfuse


langfuse_client = _get_langfuse_client()

logger = get_logger(__name__)

# 发送
async def send_message_to_langfuse(trace_id:str,name:str,input_message:dict|None=None,output_message:dict|None=None,as_type:str="span"):
    try:
        with langfuse_client.start_as_current_observation(
                as_type=as_type,
                name=name,
                trace_context={"trace_id": trace_id},
        ) as span:
            span.update(output=output_message,input=input_message)
            logger.info(f"langfuse发送成功，trace_id:{trace_id},name={name}")
    except Exception as one:
        logger.error(f"langfuse重试,失败原因：{one}")
        try:
            with langfuse_client.start_as_current_observation(
                    as_type=as_type,
                    name=name,
                    trace_context={"trace_id": trace_id},
            ) as span:
                span.update(output=output_message, input=input_message)
                logger.info(f"langfuse重试发送成功，trace_id:{trace_id},name={name}")
        except Exception as two:
            logger.error(f"发送langfuse重试失败，trace_id:{trace_id},name={name},报错：%s",two)
