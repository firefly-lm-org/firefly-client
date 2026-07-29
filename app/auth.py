"""
firefly-client · 认证模块
注册 / 登录 / Token 刷新 / Token 持久化
"""
import httpx
from rich.console import Console
from rich.panel import Panel

from app.config import ClientConfig, save_config, get_headers

console = Console()


async def register(cfg: ClientConfig, username: str, password: str) -> bool:
    """注册新用户"""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{cfg.server_url}/api/v1/auth/register",
            json={"username": username, "password": password},
        )

        if resp.status_code == 200:
            data = resp.json()
            cfg.access_token = data["access_token"]
            cfg.refresh_token = data["refresh_token"]
            cfg.user_id = data["user_id"]
            cfg.username = data["username"]
            save_config(cfg)
            console.print(Panel.fit(
                f"[green]✅ 注册成功[/green]\n"
                f"用户: {cfg.username}\n"
                f"User ID: {cfg.user_id[:8]}...",
                title="Firefly", border_style="green",
            ))
            return True
        else:
            detail = resp.json().get("detail", resp.text)
            console.print(f"[red]❌ 注册失败: {detail}[/red]")
            return False


async def login(cfg: ClientConfig, username: str, password: str) -> bool:
    """用户登录"""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{cfg.server_url}/api/v1/auth/login",
            json={"username": username, "password": password},
        )

        if resp.status_code == 200:
            data = resp.json()
            cfg.access_token = data["access_token"]
            cfg.refresh_token = data["refresh_token"]
            cfg.user_id = data["user_id"]
            cfg.username = data["username"]
            save_config(cfg)
            console.print(Panel.fit(
                f"[green]✅ 登录成功[/green]\n"
                f"用户: {cfg.username}",
                title="Firefly", border_style="green",
            ))
            return True
        else:
            detail = resp.json().get("detail", resp.text)
            console.print(f"[red]❌ 登录失败: {detail}[/red]")
            return False


async def refresh_token(cfg: ClientConfig) -> bool:
    """用 refresh_token 换取新的 access_token"""
    if not cfg.refresh_token:
        console.print("[red]❌ 没有 refresh_token，请先登录[/red]")
        return False

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{cfg.server_url}/api/v1/auth/refresh",
            json={"refresh_token": cfg.refresh_token},
        )

        if resp.status_code == 200:
            data = resp.json()
            cfg.access_token = data["access_token"]
            save_config(cfg)
            console.print("[green]✅ Token 已刷新[/green]")
            return True
        else:
            console.print(f"[red]❌ 刷新失败，请重新登录[/red]")
            return False


async def ensure_authenticated(cfg: ClientConfig) -> bool:
    """
    确保有有效的 access_token
    如果 access_token 存在就直接用，否则尝试 refresh
    """
    if cfg.access_token:
        return True
    if cfg.refresh_token:
        return await refresh_token(cfg)
    console.print("[red]❌ 未登录，请先执行 `firefly login`[/red]")
    return False
