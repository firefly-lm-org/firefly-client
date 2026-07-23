"""火种客户端 CLI — 主程序"""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import click
import torch

from firefly.core.hardware import get_hardware_info
from firefly.core.client import FireflyClient
from firefly.training.qlora import QLoRATrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("firefly")


# ─── 默认配置 ────────────────────────────────────────────────────────────────

DEFAULT_SCHEDULER_URL = os.getenv("FIREFLY_SCHEDULER_URL", "http://localhost:8000")
DEFAULT_S3_ENDPOINT = os.getenv("FIREFLY_S3_ENDPOINT", "http://localhost:9000")
DEFAULT_S3_BUCKET = os.getenv("FIREFLY_S3_BUCKET", "firefly-models")


def _load_credentials(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_credentials(path: Path, creds: dict):
    path.write_text(json.dumps(creds, indent=2, ensure_ascii=False))


CREDENTIALS_FILE = Path.home() / ".firefly" / "credentials.json"


@click.group()
def cli():
    """🔥 Firefly — 萤火虫大模型火种客户端"""
    pass


@cli.command()
def setup():
    """首次设置：注册节点"""
    Path.home().joinpath(".firefly").mkdir(exist_ok=True)

    scheduler_url = click.prompt("调度中心地址", default=DEFAULT_SCHEDULER_URL)
    username = click.prompt("你的用户名")
    password = click.prompt("密码", hide_input=True)

    hw = get_hardware_info()
    hw_dict = {
        "gpu_model": hw.gpu_model,
        "gpu_vram_gb": hw.gpu_vram_gb,
        "gpu_count": hw.gpu_count,
        "cpu_cores": hw.cpu_cores,
        "ram_gb": round(hw.ram_gb or 0, 1),
        "capabilities": hw.capabilities,
        "max_batch_size": hw.max_batch_size,
        "supports_bf16": hw.supports_bf16,
    }

    logger.info("正在注册节点...")
    client = asyncio.run(FireflyClient.register(scheduler_url, username, password, hw_dict))

    _save_credentials(CREDENTIALS_FILE, {
        "scheduler_url": scheduler_url,
        "node_id": client.credentials.node_id,
        "node_key": client.credentials.node_key,
        "hardware": hw_dict,
    })
    click.echo(f"✅ 节点注册成功！")
    click.echo(f"   Node ID : {client.credentials.node_id}")
    click.echo(f"   凭证文件: {CREDENTIALS_FILE}（请勿外泄）")


@cli.command()
@click.option("--once", is_flag=True, help="只领取一个任务后退出")
def run(once: bool):
    """持续监听并执行训练任务"""
    if not CREDENTIALS_FILE.exists():
        click.echo("❌ 尚未注册节点，请先运行: firefly setup")
        sys.exit(1)

    creds = _load_credentials(CREDENTIALS_FILE)
    scheduler_url = creds["scheduler_url"]
    hw = get_hardware_info()

    client = FireflyClient(
        scheduler_url=scheduler_url,
        node_id=creds["node_id"],
        node_key=creds["node_key"],
    )

    trainer = QLoRATrainer(
        scheduler_url=scheduler_url,
        s3_endpoint=DEFAULT_S3_ENDPOINT,
        s3_access_key=os.getenv("AWS_ACCESS_KEY_ID", "firefly_access"),
        s3_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", "firefly_secret"),
        s3_bucket=DEFAULT_S3_BUCKET,
    )

    click.echo(f"🔥 Firefly 火种客户端已启动，GPU: {hw.gpu_model or '未检测到'}")

    while True:
        try:
            asyncio.run(client.heartbeat())
            logger.info("心跳已发送，等待任务...")

            task_pkg = asyncio.run(client.claim_task())
            if task_pkg:
                click.echo(f"🎯 领取到任务: {task_pkg.task_id}")
                result = asyncio.run(trainer.train(task_pkg))

                report = {
                    "submission_id": task_pkg.submission_id,
                    "status": "completed",
                    "final_loss": result.final_loss,
                    "steps_completed": result.steps_completed,
                    "epoch_completed": result.epoch_completed,
                    "lora_weights_s3_path": result.lora_weights_path,
                    "log_summary": result.log_summary,
                }
                resp = asyncio.run(client.report_result(report))
                click.echo(f"✅ 训练完成: {result.log_summary}")
                click.echo(f"   贡献分: +{resp.get('compute_score', 0)}")

                if once:
                    break
            else:
                logger.info("暂无任务，60秒后重试...")
                time.sleep(60)

        except Exception as e:
            logger.exception(f"出错: {e}")
            logger.info("10秒后重试...")
            time.sleep(10)


if __name__ == "__main__":
    cli()
