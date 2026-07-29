"""
firefly-client · 节点模块
注册节点 / 查询状态 / 设置运行模式
"""
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from app.config import ClientConfig, save_config, get_headers
from app.hardware import full_hardware_report

console = Console()


async def register_node(cfg: ClientConfig, node_name: str) -> bool:
    """向调度中心注册本机为节点"""
    # 1. 收集硬件信息
    hw = full_hardware_report()
    console.print(Panel.fit(
        f"CPU 核心: {hw['cpu_cores']}\n"
        f"内存: {hw['total_memory_gb']} GB\n"
        f"GPU: {hw['gpu_model'] or '无'}\n"
        f"显存: {hw['gpu_vram_gb'] or 'N/A'} GB\n"
        f"系统: {hw['os_type']}",
        title="🔍 硬件检测", border_style="cyan",
    ))

    # 2. 发送注册请求
    payload = {
        "node_name": node_name,
        "cpu_cores": hw["cpu_cores"],
        "total_memory_gb": hw["total_memory_gb"],
        "gpu_model": hw["gpu_model"],
        "gpu_vram_gb": hw["gpu_vram_gb"],
        "os_type": hw["os_type"],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{cfg.server_url}/api/v1/node/register",
            headers=get_headers(cfg),
            json=payload,
        )

        if resp.status_code == 200:
            data = resp.json()
            cfg.node_id = data["node_id"]
            cfg.node_name = data["node_name"]
            save_config(cfg)

            console.print(Panel.fit(
                f"[green]✅ 节点注册成功[/green]\n"
                f"节点 ID: {data['node_id'][:8]}...\n"
                f"节点名称: {data['node_name']}\n"
                f"状态: {data['status']}\n"
                f"信誉分: {data['reputation_score']}\n"
                f"最高任务等级: L{data['max_task_level']}",
                title="Firefly Node", border_style="green",
            ))
            return True
        else:
            detail = resp.json().get("detail", resp.text)
            console.print(f"[red]❌ 注册失败: {detail}[/red]")
            return False


async def query_status(cfg: ClientConfig) -> bool:
    """查询当前节点状态"""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{cfg.server_url}/api/v1/node/status",
            headers=get_headers(cfg),
        )

        if resp.status_code == 200:
            data = resp.json()

            table = Table(title="📊 节点状态", show_header=True)
            table.add_column("属性", style="cyan")
            table.add_column("值", style="white")

            table.add_row("节点 ID", data["node_id"][:12] + "...")
            table.add_row("节点名称", data["node_name"])
            table.add_row("状态", data["status"])
            table.add_row("信誉分", str(data["reputation_score"]))
            table.add_row("最高等级", f"L{data['max_task_level']}")
            table.add_row("最后心跳", str(data.get("last_heartbeat", "N/A")))

            console.print(table)
            return True
        else:
            detail = resp.json().get("detail", resp.text)
            console.print(f"[red]❌ 查询失败: {detail}[/red]")
            return False
