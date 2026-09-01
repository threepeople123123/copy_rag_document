import pika

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

def send_message(message:str,exchange:str,routing_key:str,exchange_type:str="direct"):
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

    # 确保交换机存在
    channel.exchange_declare(
        exchange=exchange,
        exchange_type=exchange_type,
        durable=True
    )

    # 发送消息
    channel.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,  # 消息持久化
            content_type="application/json",
            content_encoding="utf-8"
        )
    )

    logger.info(f"发送消息成功:{message}")

    connection.close()