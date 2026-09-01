from enum import Enum

import pika

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

class RabbitMQ(str,Enum):
    COPY_RAG_DOCUMENT_EXCHANGE = "copy_rag_document_exchange"
    CHUNK_QUEUE = "copy_rag_document_chunk_queue"
    CHUNK_ROUTING_KEY = "copy_rag_document_chunk_routing_key"

def create_rabbitmq_resources():
    credentials = pika.PlainCredentials(settings.rabbit_mq_username, settings.rabbit_mq_password)

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=settings.rabbit_mq_host,
            port=settings.rabbit_mq_port,
            virtual_host="/",
            credentials=credentials
        )
    )

    channel = connection.channel()

    # 1. 创建交换机
    channel.exchange_declare(
        exchange=RabbitMQ.COPY_RAG_DOCUMENT_EXCHANGE,
        exchange_type="direct",
        durable=True
    )

    # 2. 创建队列
    channel.queue_declare(
        queue=RabbitMQ.CHUNK_QUEUE,
        durable=True
    )

    # 3. 绑定队列和交换机
    channel.queue_bind(
        exchange=RabbitMQ.COPY_RAG_DOCUMENT_EXCHANGE,
        queue=RabbitMQ.CHUNK_QUEUE,
        routing_key=RabbitMQ.CHUNK_ROUTING_KEY
    )

    connection.close()

