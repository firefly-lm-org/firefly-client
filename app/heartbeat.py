"""
firefly-client · 心跳模块
每 30 秒向调度中心上报一次心跳
"""
import asyncio
import time
from datetime import datetime

import httpx
from rich.console import Console

from app.config import ClientConfig, get_headers

console = Console()
HEARTBEAT_INTERVAL = 30  # 秒


async def send_heartbeat(cfg: ClientConfig, status: str = "online"):
    """发送一次心跳"""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(
                f"{cfg.server_url}/api/v1/node/heartbeat",
                headers=get_headers(cfg),
                json={"status": status},
            )
            if resp.status_code == 200:
                return True
            else:
                console.print(f"  ⚠️ 心跳失败: {resp.status_code}")
                return False
        except Exception as e:
            console.print(f"  ⚠️ 心跳异常: {e}")
            return False


async def heartbeat_loop(cfg: ClientConfig, status: str = "online"):
    """
    心跳循环（后台运行）
    每 30 秒发送一次，直到被取消
    """
    console.print(f"  💓 心跳启动 (间隔 {HEARTBEAT_INTERVAL}s)")
    while True:
        await send_heartbeat(cfg, status)
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def start_heartbeat_with_status(cfg: ClientConfig) -> asyncio.Task:
    """启动心跳循环，返回 Task 对象（用于后续取消）"""
    return asyncio.create_task(heartbeat_loop(cfg, "online"))
