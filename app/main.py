"""
firefly-client · 主入口
命令行接口：register / login / start / stop / status / stats
"""
import asyncio
import signal
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import typer
import httpx

from app.config import ClientConfig, load_config, save_config, get_headers
from app.auth import register as auth_register, login as auth_login, refresh_token, ensure_authenticated
from app.node import register_node, query_status
from app.heartbeat import start_heartbeat_with_status
from app.task_executor import execute_task

app = typer.Typer(help="🔥 萤火虫大模型 · 火种客户端")
console = Console()

# ── 全局配置 ──────────────────────
cfg: ClientConfig = load_config()
heartbeat_task: asyncio.Task | None = None
running = True


# ─────────────────────────────────────
# 命令 1：register
# ─────────────────────────────────────
@app.command()
def register(
    username: str = typer.Option(..., "--username", prompt="用户名"),
    password: str = typer.Option(..., "--password", prompt="密码", hide_input=True),
    server: str = typer.Option(None, "--server", help="调度中心地址"),
):
    """📝 注册新用户"""
    global cfg
    if server:
        cfg.server_url = server
        save_config(cfg)

    console.print(f"🔗 服务器: {cfg.server_url}")
    success = asyncio.run(auth_register(cfg, username, password))
    if success:
        console.print("[green]💡 下一步: 运行 `firefly node-register <节点名称>` 注册本机为节点[/green]")


# ─────────────────────────────────────
# 命令 2：login
# ─────────────────────────────────────
@app.command()
def login(
    username: str = typer.Option(..., "--username", prompt="用户名"),
    password: str = typer.Option(..., "--password", prompt="密码", hide_input=True),
):
    """🔑 登录已有账户"""
    global cfg
    success = asyncio.run(auth_login(cfg, username, password))
    if success:
        console.print("[green]💡 下一步: 运行 `firefly node-register <节点名称>`[/green]")


# ─────────────────────────────────────
# 命令 3：node-register
# ─────────────────────────────────────
@app.command("node-register")
def node_register(
    node_name: str = typer.Argument(..., help="节点名称（如 alice-pc）"),
):
    """🖥️ 注册本机为算力节点"""
    global cfg
    cfg = load_config()

    if not asyncio.run(ensure_authenticated(cfg)):
        sys.exit(1)

    asyncio.run(register_node(cfg, node_name))


# ─────────────────────────────────────
# 命令 4：start（核心：开始贡献算力）
# ─────────────────────────────────────
@app.command()
def start():
    """🚀 开始贡献算力（后台持续运行）"""
    global cfg, heartbeat_task, running
    cfg = load_config()

    if not asyncio.run(ensure_authenticated(cfg)):
        sys.exit(1)

    if not cfg.node_id:
        console.print("[red]❌ 请先运行 `firefly node-register <名称>` 注册节点[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold green]🔥 萤火虫客户端已启动[/bold green]\n"
        "正在连接调度中心...\n"
        "按 Ctrl+C 停止贡献",
        title="Firefly Client", border_style="yellow",
    ))

    # 设置信号处理（Windows ProactorEventLoop 不支持 add_signal_handler，需容错）
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler():
        global running
        running = False
        console.print("\n[yellow]🛑 正在停止...[/yellow]")

    try:
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
    except NotImplementedError:
        # Windows 不支持 add_signal_handler：依赖 KeyboardInterrupt 退出
        pass

    try:
        loop.run_until_complete(_run_loop())
    finally:
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
        loop.close()
        console.print("[green]👋 萤火虫客户端已停止[/green]")


async def _run_loop():
    """主循环：心跳 + 任务执行"""
    global heartbeat_task

    # 启动心跳
    heartbeat_task = await start_heartbeat_with_status(cfg)

    # 主循环：不断领取和执行任务
    while running:
        try:
            success = await execute_task(cfg)
            if not success:
                # 没有可领取的任务，等待 10 秒
                console.print("  ⏳ 暂无可用任务，10 秒后重试...")
                for _ in range(10):
                    if not running:
                        break
                    await asyncio.sleep(1)
        except Exception as e:
            console.print(f"[red]❌ 异常: {e}[/red]")
            await asyncio.sleep(5)

    # 停止心跳
    if heartbeat_task and not heartbeat_task.done():
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    # 发送离线心跳
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(
                f"{cfg.server_url}/api/v1/node/heartbeat",
                headers=get_headers(cfg),
                json={"status": "offline"},
            )
        except Exception:
            pass


# ─────────────────────────────────────
# 命令 5：status
# ─────────────────────────────────────
@app.command()
def status():
    """📊 查看节点状态"""
    global cfg
    cfg = load_config()

    if not asyncio.run(ensure_authenticated(cfg)):
        sys.exit(1)

    asyncio.run(query_status(cfg))


# ─────────────────────────────────────
# 命令 6：stats（管理员）
# ─────────────────────────────────────
@app.command()
def stats():
    """📈 查看全局统计（需管理员权限）"""
    global cfg
    cfg = load_config()

    if not asyncio.run(ensure_authenticated(cfg)):
        sys.exit(1)

    asyncio.run(_fetch_stats())


async def _fetch_stats():
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{cfg.server_url}/api/v1/admin/stats",
            headers=get_headers(cfg),
        )
        if resp.status_code == 200:
            data = resp.json()
            from rich.table import Table
            table = Table(title="📈 全局统计", show_header=True)
            table.add_column("指标", style="cyan")
            table.add_column("数值", style="white")

            table.add_row("节点总数", str(data["nodes"]["total"]))
            table.add_row("在线节点", str(data["nodes"]["online"]))
            table.add_row("忙碌节点", str(data["nodes"]["busy"]))
            table.add_row("待处理任务", str(data["tasks"]["pending"]))
            table.add_row("运行中任务", str(data["tasks"]["running"]))
            table.add_row("已完成任务", str(data["tasks"]["completed"]))
            table.add_row("失败任务", str(data["tasks"]["failed"]))
            table.add_row("注册用户", str(data["users"]["total"]))

            console.print(table)
        else:
            console.print(f"[red]❌ 查询失败: {resp.status_code}[/red]")


# ─────────────────────────────────────
# 命令 7：config
# ─────────────────────────────────────
@app.command()
def config_show():
    """⚙️ 显示当前配置"""
    global cfg
    cfg = load_config()

    from rich.table import Table
    table = Table(title="⚙️ 客户端配置", show_header=True)
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="white")

    table.add_row("服务器", cfg.server_url)
    table.add_row("用户", cfg.username or "(未登录)")
    table.add_row("用户 ID", cfg.user_id[:12] + "..." if cfg.user_id else "(空)")
    table.add_row("节点 ID", cfg.node_id[:12] + "..." if cfg.node_id else "(未注册)")
    table.add_row("节点名称", cfg.node_name or "(空)")
    table.add_row("登录状态", "✅ 已登录" if cfg.access_token else "❌ 未登录")

    console.print(table)


# ─────────────────────────────────────
# 入口
# ─────────────────────────────────────
if __name__ == "__main__":
    app()
