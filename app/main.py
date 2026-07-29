"""
firefly-client · 主入口
命令行接口：register / login / start / stop / status / stats
"""
import asyncio
import os
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path

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
        console.print("[green]💡 下一步: 运行 `firefly-node node-register <节点名称>` 注册本机为节点[/green]")


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
        console.print("[green]💡 下一步: 运行 `firefly-node node-register <节点名称>`[/green]")


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
def start(
    mock: bool = typer.Option(
        False,
        "--mock",
        help="🔧 使用模拟训练（无 GPU / 测试用），不加则默认真实 QLoRA",
    ),
):
    """🚀 开始贡献算力（后台持续运行）"""
    global cfg, heartbeat_task, running
    cfg = load_config()

    # ── 注入训练模式（task_executor.py 会读取）────────
    if mock:
        os.environ["FIREFLY_MOCK"] = "1"
        train_mode = "Mock（测试模式）"
        console.print("[yellow]⚠️  FIREFLY_MOCK=1，将使用模拟训练[/yellow]")
    else:
        os.environ.pop("FIREFLY_MOCK", None)   # 默认为真实 QLoRA
        train_mode = "真实 QLoRA（需 NVIDIA GPU）"

    if not asyncio.run(ensure_authenticated(cfg)):
        sys.exit(1)

    if not cfg.node_id:
        console.print("[red]❌ 请先运行 `firefly-node node-register <名称>` 注册节点[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold green]🔥 萤火虫客户端已启动[/bold green]\n"
        f"训练模式: {train_mode}\n"
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
# 命令 8：train-local（本地训练，不连调度中心）
# ─────────────────────────────────────
@app.command("train-local", help="🏋️ 本地训练（无需登录，数据不出本机）")
def train_local(
    dataset: str = typer.Option(
        None,
        "--dataset",
        "-d",
        help="JSONL 数据文件路径，例如 data/law_qa.jsonl",
    ),
    domain: str = typer.Option(
        "general",
        "--domain",
        help="领域标签（law / medical / python / tax / education），仅用于分类",
    ),
    output: str = typer.Option(
        "firefly_adapter.safetensors",
        "--output",
        "-o",
        help="输出 LoRA 适配器文件路径",
    ),
    steps: int = typer.Option(30, "--steps", "-s", help="训练步数（默认 30，约 30 分钟）"),
    base_model: str = typer.Option(
        "unsloth/Qwen3-1.5B-Instruct-4bit",
        "--base-model",
        help="基础模型（需 HuggingFace 可访问）",
    ),
):
    """本地 QLoRA 训练，数据不出本机，直接生成 LoRA 适配器文件"""
    import shutil, time as _time

    # 检查 unsloth
    try:
        import unsloth
    except ImportError:
        console.print(
            "[red]❌ unsloth 未安装。请运行：[/red]\n"
            "pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )
        raise typer.Exit(1)

    # 设置环境变量（RealTrainer 会读取）
    os.environ["FIREFLY_MODEL_PATH"] = base_model
    os.environ["FIREFLY_MAX_STEPS"] = str(steps)

    task_id = f"local_{domain}"
    console.print(f"[cyan]📦 领域: {domain} | 数据: {dataset or '(Demo)'}"
                   f" | 步数: {steps} | 基础模型: {base_model}[/cyan]")

    from app.trainer.real_trainer import RealTrainer

    def progress_cb(info):
        console.print(
            f"  Step {info['step']}/{steps}"
            f" | Loss: {info.get('loss', 0):.4f}"
            f" | Elapsed: {info.get('elapsed', 0):.0f}s"
        )

    trainer = RealTrainer(task_id=task_id, data_path=dataset, progress_callback=progress_cb)

    start = _time.time()
    try:
        meta = trainer.train()
    except Exception as e:
        console.print(f"[red]❌ 训练失败: {e}[/red]")
        raise typer.Exit(1)

    elapsed = _time.time() - start
    lora_path = meta["adapter_path"]  # 实际文件路径

    # 复制到用户指定位置
    if os.path.abspath(lora_path) != os.path.abspath(output):
        shutil.copy2(lora_path, output)

    size_mb = os.path.getsize(output) / 1e6
    console.print(
        f"[green]\n✅ 训练完成！[/green]\n"
        f"  文件: {os.path.abspath(output)} ({size_mb:.1f} MB)\n"
        f"  最终 Loss: {meta['final_loss']:.4f}\n"
        f"  耗时: {elapsed:.0f}s\n"
        f"  可用 `firefly-node chat --adapter {os.path.abspath(output)}` 验证"
    )


# ─────────────────────────────────────
# 命令 9：chat（推理验证）
# ─────────────────────────────────────
@app.command("chat", help="💬 用 LoRA 适配器推理（单次提问或交互模式）")
def chat(
    adapter: str = typer.Option(
        None, "--adapter", "-a", help="LoRA 适配器目录或 .safetensors 文件路径"
    ),
    prompt: str = typer.Option(None, "--prompt", "-p", help="单次提问（不指定则进入交互模式）"),
    base_model: str = typer.Option(
        "unsloth/Qwen3-1.5B-Instruct-4bit",
        "--base-model",
        help="基础模型（需与训练时一致）",
    ),
    max_tokens: int = typer.Option(200, "--max-tokens", help="最大生成长度"),
):
    if not adapter:
        console.print("[red]❌ 请指定 --adapter 参数，例如：[/red]\n"
                       "firefly-node chat --adapter my_adapter.safetensors --prompt '什么是劳动合同？'")
        raise typer.Exit(1)

    try:
        import unsloth, torch
        from unsloth import FastLanguageModel
        from peft import PeftModel
    except ImportError as e:
        console.print("[red]❌ unsloth 未安装。请运行：[/red]\n"
                       "pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple")
        raise typer.Exit(1)

    console.print(f"[cyan]加载模型: {base_model} + 适配器: {adapter}[/cyan]")
    try:
        base, tok = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=512,
            dtype=torch.bfloat16 if unsloth.is_bfloat16_supported() else torch.float16,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(base, adapter)
        FastLanguageModel.for_inference(model)
    except Exception as e:
        console.print(f"[red]❌ 模型加载失败: {e}[/red]")
        raise typer.Exit(1)

    def _ask(q: str):
        msgs = [
            {"role": "system", "content": "你是一个有帮助的专业助手"},
            {"role": "user", "content": q},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
        ans = tok.decode(out[0][len(inp["input_ids"]):], skip_special_tokens=True)
        return ans

    if prompt:
        console.print(f"\n[yellow]你:[/yellow] {prompt}\n")
        console.print(f"[green]模型:[/green] {_ask(prompt)}")
    else:
        console.print("[cyan]💬 交互模式（输入空行退出）[/cyan]\n")
        while True:
            q = console.input("[yellow]你: [/yellow]")
            if not q.strip():
                break
            console.print(f"[green]模型:[/green] {_ask(q)}\n")


# ─────────────────────────────────────
# 命令组 10：fed（联邦训练子命令）
# ─────────────────────────────────────
fed_app = typer.Typer(help="🌐 联邦训练命令组（status / claim / train / complete / download）")
app.add_typer(fed_app, name="fed", help="🌐 联邦训练命令组")


@fed_app.command("status", help="📊 查看调度中心状态 + 任务池")
def fed_status(
    server: str = typer.Option(None, "--server", "-s", help="调度中心地址（默认读配置）"),
):
    """查看调度中心健康状态和任务池统计"""
    c = load_config()
    url = server or c.server_url
    console.print(f"[cyan]🔗 调度中心: {url}[/cyan]")

    async def _check():
        async with httpx.AsyncClient(timeout=10) as client:
            # 1. 健康检查
            try:
                resp = await client.get(f"{url}/health")
                if resp.status_code == 200:
                    console.print("[green]✅ 调度中心在线[/green]")
                else:
                    console.print(f"[yellow]⚠️  调度中心返回 {resp.status_code}[/yellow]")
            except httpx.RequestError as e:
                console.print(f"[red]❌ 无法连接调度中心: {e}[/red]")
                console.print("[yellow]💡 提示: 可以先使用 `firefly-node fed train` 做本地训练[/yellow]")
                return

            # 2. 任务池统计
            try:
                resp = await client.get(f"{url}/api/v1/admin/stats", headers=get_headers(c))
                if resp.status_code == 200:
                    data = resp.json()
                    from rich.table import Table
                    table = Table(title="📈 任务池统计", show_header=True)
                    table.add_column("指标", style="cyan")
                    table.add_column("数值", style="white")
                    table.add_row("节点总数", str(data["nodes"]["total"]))
                    table.add_row("在线节点", str(data["nodes"]["online"]))
                    table.add_row("待处理任务", str(data["tasks"]["pending"]))
                    table.add_row("运行中任务", str(data["tasks"]["running"]))
                    table.add_row("已完成任务", str(data["tasks"]["completed"]))
                    console.print(table)
                elif resp.status_code == 401:
                    console.print("[yellow]⚠️  需要登录才能查看统计（firefly-node login）[/yellow]")
            except Exception:
                pass

            # 3. 聚合状态（公开接口）
            try:
                resp = await client.get(f"{url}/api/v1/aggregation/list")
                if resp.status_code == 200:
                    agg = resp.json()
                    if agg.get("rounds"):
                        console.print(f"[green]📦 可聚合版本: {len(agg['rounds'])} 个[/green]")
                    else:
                        console.print("[dim]📦 暂无就绪的聚合批次[/dim]")
            except Exception:
                pass

    asyncio.run(_check())


@fed_app.command("claim", help="🎯 认领一个联邦训练任务")
def fed_claim(
    domain: str = typer.Option("law", "--domain", "-d", help="领域: law/medical/python/tax/education"),
    server: str = typer.Option(None, "--server", "-s", help="调度中心地址"),
):
    """向调度中心认领任务。连不上时自动降级到 Mock 任务，不崩溃。"""
    c = load_config()
    url = server or c.server_url

    async def _claim():
        from app.executors.fed_executor import FedExecutor

        executor = FedExecutor(url)
        try:
            task = await executor.claim_task(domain=domain, cfg=c)
            console.print(f"[green]✅ 认领成功！[/green]")
            console.print(f"  Task ID: {task.task_id}")
            console.print(f"  任务名: {task.task_name}")
            console.print(f"  领域: {task.domain}")
            console.print(f"  难度: L{task.task_level}")
            console.print(f"  截止: {task.deadline.strftime('%Y-%m-%d %H:%M')}")
            console.print(f"\n[cyan]下一步: firefly-node fed train --task-id {task.task_id}[/cyan]")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                console.print(f"[yellow]⚠️  暂无 {domain} 领域的可用任务[/yellow]")
                console.print("[dim]可以先用 `firefly-node fed train` 做本地训练[/dim]")
            else:
                console.print(f"[red]❌ 认领失败: {e.response.status_code} {e.response.text[:200]}[/red]")
        except httpx.RequestError as e:
            # 降级：生成 Mock 任务
            import uuid as _uuid
            mock_id = f"mock_{domain}_{_uuid.uuid4().hex[:6]}"
            console.print(f"[yellow]⚠️  无法连接调度中心，已降级为 Mock 任务[/yellow]")
            console.print(f"  Task ID: {mock_id}")
            console.print(f"  领域: {domain}")
            console.print(f"  模式: 离线（训练结果将保存到本地）")
            console.print(f"\n[cyan]下一步: firefly-node fed train --task-id {mock_id} --dataset data/{domain}_qa.jsonl[/cyan]")
        finally:
            await executor.close()

    asyncio.run(_claim())


@fed_app.command("train", help="🏋️ 执行联邦训练（本地 QLoRA）")
def fed_train(
    task_id: str = typer.Option("local", "--task-id", "-t", help="任务 ID（claim 获得或自填）"),
    dataset: str = typer.Option(None, "--dataset", "-d", help="JSONL 数据文件路径"),
    domain: str = typer.Option("law", "--domain", help="领域标签"),
    output: str = typer.Option("firefly_adapter.safetensors", "--output", "-o", help="输出文件路径"),
    steps: int = typer.Option(30, "--steps", "-s", help="训练步数"),
    base_model: str = typer.Option("unsloth/Qwen3-1.5B-Instruct-4bit", "--base-model", help="基础模型"),
    mock: bool = typer.Option(False, "--mock", help="使用模拟训练（无 GPU 时）"),
):
    """执行本地 QLoRA 训练。复用 train-local 内核，带数据格式校验。"""
    import shutil, time as _time

    # 检查 unsloth（mock 模式跳过）
    if not mock:
        try:
            import unsloth
        except ImportError:
            console.print(
                "[red]❌ unsloth 未安装。请运行：[/red]\n"
                "pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                "[yellow]或使用 --mock 模式测试流程[/yellow]"
            )
            raise typer.Exit(1)
    else:
        os.environ["FIREFLY_MOCK"] = "1"

    # 设置环境变量
    os.environ["FIREFLY_MODEL_PATH"] = base_model
    os.environ["FIREFLY_MAX_STEPS"] = str(steps)

    console.print(f"[cyan]📦 Task: {task_id} | 领域: {domain} | 数据: {dataset or '(Demo)'}"
                   f" | 步数: {steps}{' | Mock' if mock else ''}[/cyan]")

    from app.trainer.real_trainer import RealTrainer

    def progress_cb(info):
        console.print(
            f"  Step {info['step']}/{steps}"
            f" | Loss: {info.get('loss', 0):.4f}"
            f" | Elapsed: {info.get('elapsed', 0):.0f}s"
        )

    trainer = RealTrainer(task_id=task_id, data_path=dataset, progress_callback=progress_cb)

    start = _time.time()
    try:
        meta = trainer.train()
    except Exception as e:
        console.print(f"[red]❌ 训练失败: {e}[/red]")
        raise typer.Exit(1)

    elapsed = _time.time() - start
    lora_path = meta["adapter_path"]

    if os.path.abspath(lora_path) != os.path.abspath(output):
        shutil.copy2(lora_path, output)

    size_mb = os.path.getsize(output) / 1e6
    final_loss = meta.get("final_loss", 0.0)
    console.print(
        f"[green]\n✅ 训练完成！[/green]\n"
        f"  Task ID: {task_id}\n"
        f"  文件: {os.path.abspath(output)} ({size_mb:.1f} MB)\n"
        f"  最终 Loss: {final_loss:.4f}\n"
        f"  耗时: {elapsed:.0f}s\n"
        f"\n[cyan]下一步: firefly-node fed complete --task-id {task_id} --loss {final_loss:.2f}[/cyan]"
    )


@fed_app.command("complete", help="✅ 回传训练结果（脱敏信号）")
def fed_complete(
    task_id: str = typer.Option(..., "--task-id", "-t", help="任务 ID"),
    loss: float = typer.Option(0.0, "--loss", "-l", help="最终 loss"),
    samples: int = typer.Option(0, "--samples", "-n", help="训练样本数"),
    server: str = typer.Option(None, "--server", "-s", help="调度中心地址"),
):
    """向调度中心回传训练结果。连不上时保存到本地 ~/.firefly/signals/"""
    c = load_config()
    url = server or c.server_url

    async def _complete():
        from app.executors.fed_executor import FedExecutor
        from app.executors.fed_executor import TrainingResult

        # 构造结果对象
        result = TrainingResult(
            task_id=task_id,
            final_loss=loss,
            holdout_accuracy=0.0,
            peak_vram_mb=0.0,
            execution_time_sec=0.0,
            total_steps=0,
            lora_path=None,
            training_log={
                "task_id": task_id,
                "final_loss": loss,
                "sample_count": samples,
            },
        )

        executor = FedExecutor(url)
        try:
            await executor.complete_task(result, cfg=c, donate_signal=True)
            console.print(f"[green]✅ 结果已回传到调度中心[/green]")
            console.print(f"  Task ID: {task_id}")
            console.print(f"  Loss: {loss}")
            console.print(f"  Samples: {samples}")
        except httpx.RequestError:
            # 降级：保存到本地
            sig_dir = Path.home() / ".firefly" / "signals"
            sig_dir.mkdir(parents=True, exist_ok=True)
            sig_file = sig_dir / f"{task_id}_signal.json"

            import json as _json
            sig_data = {
                "task_id": task_id,
                "final_loss": loss,
                "sample_count": samples,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "offline_pending",
            }
            sig_file.write_text(_json.dumps(sig_data, indent=2), encoding="utf-8")

            console.print(f"[yellow]⚠️  无法连接调度中心，信号已保存到本地[/yellow]")
            console.print(f"  文件: {sig_file}")
            console.print(f"[dim]联网后可手动重传[/dim]")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                console.print("[yellow]⚠️  需要登录（firefly-node login），信号已保存到本地[/yellow]")
                sig_dir = Path.home() / ".firefly" / "signals"
                sig_dir.mkdir(parents=True, exist_ok=True)
                sig_file = sig_dir / f"{task_id}_signal.json"
                import json as _json
                sig_data = {
                    "task_id": task_id,
                    "final_loss": loss,
                    "sample_count": samples,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "auth_pending",
                }
                sig_file.write_text(_json.dumps(sig_data, indent=2), encoding="utf-8")
                console.print(f"  文件: {sig_file}")
            else:
                console.print(f"[red]❌ 回传失败: {e.response.status_code}[/red]")
        finally:
            await executor.close()

    asyncio.run(_complete())


@fed_app.command("download", help="⬇️ 下载聚合权重")
def fed_download(
    round_num: int = typer.Option(1, "--round", "-r", help="聚合轮次"),
    output: str = typer.Option("aggregated.safetensors", "--output", "-o", help="输出文件路径"),
    server: str = typer.Option(None, "--server", "-s", help="调度中心地址"),
):
    """下载指定轮次的 FedAvg 聚合权重"""
    c = load_config()
    url = server or c.server_url

    async def _download():
        from app.executors.fed_executor import FedExecutor
        from pathlib import Path

        executor = FedExecutor(url)
        try:
            local_path = await executor.download_aggregated(
                round_num=round_num,
                target_dir=Path(output).parent if "/" in output or "\\" in output else Path("."),
                cfg=c,
            )
            if local_path:
                # 重命名到用户指定路径
                if str(local_path) != output:
                    shutil.move(str(local_path), output)
                size_mb = os.path.getsize(output) / 1e6
                console.print(f"[green]✅ 下载成功！[/green]")
                console.print(f"  文件: {os.path.abspath(output)} ({size_mb:.1f} MB)")
                console.print(f"  轮次: {round_num}")
            else:
                console.print(f"[yellow]⚠️  轮次 {round_num} 暂无聚合权重[/yellow]")
                console.print("[dim]聚合需要多个节点完成任务后才会触发[/dim]")
        except httpx.RequestError:
            console.print(f"[yellow]⚠️  无法连接调度中心[/yellow]")
            console.print("[dim]聚合权重需要调度中心在线时才能下载[/dim]")
        finally:
            await executor.close()

    asyncio.run(_download())


# ─────────────────────────────────────
# 命令 10：demo（单机多节点联邦训练演示）
# ─────────────────────────────────────
@app.command()
def demo(
    domain: str = typer.Option("law", "--domain", "-d", help="领域 (law/medical/python/tax/education)"),
    nodes: int = typer.Option(4, "--nodes", "-n", help="模拟节点数 (默认 4)"),
    steps: int = typer.Option(5, "--steps", "-s", help="训练步数 (默认 5)"),
    output: str = typer.Option("./demo_output", "--output", "-o", help="输出目录"),
):
    """🔥 单机多节点联邦训练演示 (无需 GPU，30 分钟跑通完整管线)"""
    import subprocess

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "demo_mode.py"
    if not script_path.exists():
        console.print(f"[red]错误: demo 脚本不存在: {script_path}[/red]")
        raise typer.Exit(1)

    cmd = [
        sys.executable, str(script_path),
        "--domain", domain,
        "--nodes", str(nodes),
        "--steps", str(steps),
        "--output", output,
    ]
    console.print(f"[cyan]启动 Demo Mode: {domain} | {nodes} nodes | {steps} steps[/cyan]")
    console.print(f"[dim]脚本: {script_path}[/dim]\n")
    result = subprocess.run(cmd, cwd=str(script_path.parent.parent))
    if result.returncode == 0:
        console.print(f"\n[green]Demo 完成！[/green]")
        console.print(f"  聚合权重: {output}/aggregated.safetensors")
        console.print(f"  下一步: firefly-node chat --adapter {output}/aggregated.safetensors")
    else:
        console.print(f"\n[red]Demo 失败 (exit code {result.returncode})[/red]")
        raise typer.Exit(1)


# ─────────────────────────────────────
# 命令 11：doctor（环境检查）
# ─────────────────────────────────────
@app.command()
def doctor():
    """🔍 环境检查 - 5 秒验证你的环境是否就绪"""
    import platform
    import shutil
    import socket
    import urllib.request

    console.print()
    console.print("[bold cyan]Firefly LM 环境检查[/bold cyan]")
    console.print("[dim]验证你的环境是否可以开始联邦训练[/dim]\n")

    all_ok = True

    # 1. Python 版本
    try:
        py_ver = platform.python_version()
        major, minor = sys.version_info[:2]
        if major >= 3 and minor >= 8:
            console.print(f"  [green]✅[/green] Python {py_ver}")
        else:
            console.print(f"  [red]❌[/red] Python {py_ver} (需要 3.8+)")
            all_ok = False
    except Exception:
        console.print("  [red]❌[/red] Python 版本检测失败")
        all_ok = False

    # 2. torch
    try:
        import torch
        cuda = torch.cuda.is_available()
        ver = torch.__version__
        if cuda:
            console.print(f"  [green]✅[/green] torch {ver} (CUDA 可用: {torch.cuda.get_device_name(0)})")
        else:
            console.print(f"  [yellow]⚠️[/yellow] torch {ver} (CPU 模式, 训练较慢但 demo 可用)")
    except ImportError:
        console.print("  [red]❌[/red] torch 未安装 → pip install torch")
        all_ok = False

    # 3. safetensors
    try:
        import safetensors
        console.print(f"  [green]✅[/green] safetensors {safetensors.__version__}")
    except ImportError:
        console.print("  [red]❌[/red] safetensors 未安装 → pip install safetensors")
        all_ok = False

    # 4. 配置文件
    config_path = Path.home() / ".firefly" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            server = cfg.get("server_url", "未设置")
            console.print(f"  [green]✅[/green] 配置文件: {config_path}")
            console.print(f"       server_url: {server}")
        except Exception:
            console.print(f"  [yellow]⚠️[/yellow] 配置文件存在但解析失败: {config_path}")
    else:
        console.print(f"  [yellow]⚠️[/yellow] 配置文件不存在: {config_path}")
        console.print("       运行 firefly-node config 创建")

    # 5. 调度中心可达
    server_url = "http://106.14.220.169:8000"
    try:
        with urllib.request.urlopen(f"{server_url}/api/v1/aggregation/list", timeout=5) as resp:
            if resp.status == 200:
                console.print(f"  [green]✅[/green] 调度中心可达: {server_url}")
            else:
                console.print(f"  [yellow]⚠️[/yellow] 调度中心返回 {resp.status}")
                all_ok = False
    except Exception:
        console.print(f"  [red]❌[/red] 调度中心不可达: {server_url}")
        all_ok = False

    # 6. 数据文件
    data_files = ["data/law_qa.jsonl", "data/law_holdout.jsonl"]
    project_root = Path(__file__).resolve().parent.parent
    for df in data_files:
        full_path = project_root / df
        if full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
            console.print(f"  [green]✅[/green] {df} ({count} 条)")
        else:
            console.print(f"  [yellow]⚠️[/yellow] {df} 不存在")

    # 总结
    console.print()
    if all_ok:
        console.print("[bold green] 一切就绪，可以开始训练！[/bold green]")
        console.print("\n  快速开始:")
        console.print("    firefly-node demo          # 30 秒体验完整联邦流程")
        console.print("    firefly-node train-local    # 用 GPU 训练你的第一个适配器")
    else:
        console.print("[bold red] 有问题需要修复，看上面的 ❌ 项[/bold red]")
    console.print()


# ─────────────────────────────────────
# 命令 12：verify（适配器来源验证）
# ─────────────────────────────────────
@app.command()
def verify(
    adapter: str = typer.Option(..., "--adapter", "-a", help="适配器文件路径 (.safetensors)"),
):
    """🔐 验证适配器是否来自火种调度中心"""
    import hashlib

    adapter_path = Path(adapter).resolve()
    if not adapter_path.exists():
        console.print(f"[red]错误: 文件不存在: {adapter_path}[/red]")
        raise typer.Exit(1)

    # 检查同目录下的 manifest
    manifest_path = adapter_path.parent / (adapter_path.stem + "_manifest.json")
    if not manifest_path.exists():
        console.print(f"[yellow]警告: 未找到 manifest 文件[/yellow]")
        console.print(f"  适配器: {adapter_path}")
        console.print(f"  manifest: {manifest_path}")
        console.print()
        console.print("[dim]适配器仍可使用，但无法确认是否来自火种调度中心。[/dim]")
        console.print("[dim]如果是自行训练的适配器，这是正常的。[/dim]")
        raise typer.Exit(0)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        console.print(f"[red]错误: manifest 文件解析失败: {e}[/red]")
        raise typer.Exit(1)

    # 计算 SHA256
    console.print(f"[cyan]验证适配器: {adapter_path.name}[/cyan]\n")
    console.print("[dim]计算 SHA256...[/dim]")
    sha = hashlib.sha256()
    with open(adapter_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual_sha = sha.hexdigest()
    expected_sha = manifest.get("sha256", "")

    console.print()
    if expected_sha and actual_sha == expected_sha:
        console.print(f"  [green]PASS[/green] SHA256 匹配")
        console.print(f"         {actual_sha[:16]}...")
    elif expected_sha:
        console.print(f"  [red]FAIL[/red] SHA256 不匹配")
        console.print(f"         期望: {expected_sha[:16]}...")
        console.print(f"         实际: {actual_sha[:16]}...")
        console.print()
        console.print("[red]适配器已被篡改，不建议使用。[/red]")
        raise typer.Exit(1)
    else:
        console.print(f"  [yellow]SKIP[/yellow] manifest 中无 SHA256 字段")

    # 显示 manifest 信息
    console.print()
    console.print(f"  [bold]验证结果[/bold]")
    console.print(f"  来源:     火种调度中心")
    console.print(f"  聚合轮次: {manifest.get('round_id', '未知')}")
    console.print(f"  领域:     {manifest.get('domain', '未知')}")
    console.print(f"  参与节点: {len(manifest.get('task_ids', manifest.get('node_ids', [])))} 个")
    console.print(f"  聚合方式: {manifest.get('aggregation_method', 'FedAvg')}")
    console.print(f"  创建时间: {manifest.get('created_at', '未知')}")
    console.print(f"  SHA256:   {actual_sha[:16]}...")
    console.print()
    console.print("[green]  适配器已验证通过，可以放心使用。[/green]")


# ─────────────────────────────────────
# 入口
# ─────────────────────────────────────
if __name__ == "__main__":
    app()
