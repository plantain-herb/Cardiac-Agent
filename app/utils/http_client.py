"""HTTP 客户端 - 带连接池和重试的全局 Session"""

import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _create_session() -> requests.Session:
    """创建带有连接池和重试机制的 Session"""
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


_session = None
_session_lock = threading.Lock()


def get_http_session() -> requests.Session:
    """获取全局 Session 对象（线程安全）"""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = _create_session()
    return _session
