"""Firefly API 客户端：与调度中心通信"""
import httpx
from dataclasses import dataclass
from typing import Optional
from uuid import UUID


@dataclass
class NodeCredentials:
    node_id: str
    node_key: str
    scheduler_url: str


@dataclass
class TaskPackage:
    submission_id: str
    task_id: str
    base_model: str
    config: dict
    train_data_s3_prefix: str
    submission_id_key: str


class FireflyClient:
    """调度中心 API 客户端"""

    def __init__(self, scheduler_url: str, node_id: str, node_key: str):
        self.credentials = NodeCredentials(
            node_id=node_id, node_key=node_key, scheduler_url=scheduler_url
        )
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    @classmethod
    async def register(
        cls,
        scheduler_url: str,
        username: str,
        password: str,
        hardware_info: dict,
    ) -> "FireflyClient":
        """注册新节点：先登录拿 token，再注册节点"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. 登录
            resp = await client.post(
                f"{scheduler_url}/auth/token",
                data={"username": username, "password": password},
            )
            resp.raise_for_status()
            token = resp.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            # 2. 注册节点
            resp = await client.post(
                f"{scheduler_url}/nodes/register",
                headers=headers,
                json=hardware_info,
            )
            resp.raise_for_status()
            data = resp.json()

            return cls(
                scheduler_url=scheduler_url,
                node_id=data["node_id"],
                node_key=data["node_key"],
            )

    async def heartbeat(self) -> dict:
        """发送心跳，返回调度指令"""
        resp = await self._http.post(
            f"{self.credentials.scheduler_url}/nodes/{self.credentials.node_id}/heartbeat",
            params={"node_key": self.credentials.node_key},
        )
        resp.raise_for_status()
        return resp.json()

    async def claim_task(self) -> Optional[TaskPackage]:
        """尝试领取一个训练任务"""
        resp = await self._http.post(
            f"{self.credentials.scheduler_url}/tasks/claim",
            json={
                "node_id": self.credentials.node_id,
                "node_key": self.credentials.node_key,
            },
        )
        if resp.status_code == 404:
            return None  # 暂无可用任务
        resp.raise_for_status()
        data = resp.json()
        return TaskPackage(**data)

    async def report_result(self, result: dict) -> dict:
        """上报训练结果"""
        resp = await self._http.post(
            f"{self.credentials.scheduler_url}/submissions/report",
            params={"node_key": self.credentials.node_key},
            json=result,
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._http.aclose()
