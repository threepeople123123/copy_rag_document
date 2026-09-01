import pika
import json
import time

from core.config import settings
from core.logging import get_logger
from rabbitmq.topology import RabbitMQ


logger = get_logger(__name__)

class RabbitMQConsumer:

    MAX_RETRY = 3

    def __init__(self):
        self.connection = None
        self.channel = None
        self.running = True

    def connect(self):

        credentials = pika.PlainCredentials(
            settings.rabbit_mq_username,
            settings.rabbit_mq_password
        )

        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=settings.rabbit_mq_host,
                port=settings.rabbit_mq_port,
                virtual_host="/",
                credentials=credentials,
                heartbeat=60
            )
        )

        self.channel = self.connection.channel()

        self.channel.basic_qos(
            prefetch_count=1
        )

        self.channel.basic_consume(
            queue=RabbitMQ.CHUNK_QUEUE,
            on_message_callback=self.callback,
            auto_ack=False
        )

    def callback(
        self,
        ch,
        method,
        properties,
        body
    ):

        try:

            data = json.loads(
                body.decode("utf-8")
            )

            logger.info("开始处理消息：%s",data)

            # ==========================
            # 业务处理
            # ==========================

            self.do_business(data)

            # ==========================
            # 成功 ACK
            # ==========================

            ch.basic_ack(
                delivery_tag=method.delivery_tag
            )

            logger.info("处理成功")

        except Exception as e:

            logger.error(f"消费失败{e}")

            # 获取重试次数
            retry_count = 0

            if properties.headers:

                retry_count = properties.headers.get(
                    "retry_count",
                    0
                )

            retry_count += 1

            logger.error(f"当前失败次数: {retry_count}")

            # ==========================
            # 小于3次：重新投递
            # ==========================

            if retry_count < self.MAX_RETRY:

                headers = {}

                if properties.headers:
                    headers.update(
                        properties.headers
                    )

                headers["retry_count"] = retry_count

                ch.basic_publish(
                    exchange=RabbitMQ.COPY_RAG_DOCUMENT_EXCHANGE,
                    routing_key=RabbitMQ.CHUNK_ROUTING_KEY,
                    body=body,
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        content_type="application/json",
                        headers=headers
                    )
                )

                # 确认原消息
                ch.basic_ack(delivery_tag=method.delivery_tag)

                logger.error(f"重新投递，第 {retry_count} 次")

            # ==========================
            # 第3次失败
            # ==========================

            else:

                print(
                    "达到最大重试次数，进入死信队列"
                )

                ch.basic_publish(
                    exchange="app.dlx",
                    routing_key="app.dlq",
                    body=body,
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        content_type="application/json",
                        headers={
                            "retry_count": retry_count,
                            "error": str(e)
                        }
                    )
                )

                # 确认原消息
                ch.basic_ack(
                    delivery_tag=method.delivery_tag
                )

    def do_business(self, data):

        print(
            "执行具体业务:",
            data
        )

        # 模拟业务失败
        if data.get("fail"):

            raise Exception(
                "业务处理失败"
            )

    def start(self):

        while self.running:

            try:

                print(
                    "Consumer 开始消费..."
                )

                self.connect()

                self.channel.start_consuming()

            except Exception as e:

                print(
                    "Consumer 发生异常:",
                    e
                )

                if self.running:

                    print(
                        "5秒后重新连接 RabbitMQ..."
                    )

                    time.sleep(5)

    def stop(self):

        print(
            "正在停止 Consumer..."
        )

        self.running = False

        if self.connection:

            try:
                self.connection.close()
            except Exception:
                pass