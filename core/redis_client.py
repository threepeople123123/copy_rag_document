import redis
from redis import Redis

from core.config import settings

# 模块级缓存:全局唯一的 Redis 客户端(进程内单例)
_redis_client: Redis | None = None


def get_redis_client() -> Redis:
    """懒加载单例:整个进程只创建一次 Redis 客户端和它的连接池。

    每次调用返回的都是同一个客户端对象;客户端内部持有同一个
    redis.ConnectionPool —— 执行每条命令时从池中借一个连接,用完归还,
    不会为每次调用新建 TCP 连接(连接数上限 = max_connections)。
    """
    global _redis_client
    # 不写 global 的话,函数内的赋值 (_redis_client = ...) 会让 Python
    # 把 _redis_client 当成函数局部变量,上面的 if 判断就会抛
    # UnboundLocalError: cannot access local variable ...
    if _redis_client is None:
        pool = redis.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            max_connections=settings.redis_max_connections,
            # decode_responses 必须放在连接池上:若只作为 Redis(..., decode_responses=True)
            # 传入而连接池是外面建好的,redis-py 8.x 会静默忽略它,返回 bytes 而非 str
            decode_responses=True,
        )
        # 注意:参数名是 connection_pool,不是 pool(写成 pool= 会抛 TypeError)
        _redis_client = redis.Redis(connection_pool=pool)
    return _redis_client


# 模块导入时即完成初始化;Python 模块有 sys.modules 缓存,
# 任何地方 `from core.redis_client import redis_client` 拿到的都是同一个实例
redis_client: Redis = get_redis_client()
