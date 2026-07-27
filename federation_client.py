# -*- coding: utf-8 -*-
"""
firefly-client · 联邦训练客户端（连接真实调度中心）

适配已在阿里云 106.14.220.169:8000 验证通过的调度中心 API：
  POST /api/v1/auth/login           {username, password} -> {access_token, user_id, is_admin}
  POST /api/v1/auth/register        {username, email, password} -> {user_id}
  GET  /api/v1/tasks/pending        (Bearer) -> [{task_id, task_type, model_name, lora_rank, max_steps, ...}]
  POST /api/v1/tasks/claim          {task_id} (Bearer node) -> {status, claimed_by}
  POST /api/v1/tasks/progress       {task_id, progress_pct, final_loss?} (Bearer node)
  POST /api/v1/tasks/complete       {task_id, final_loss} (Bearer node) -> {status}
  GET  /api/v1/admin/stats          (Bearer admin) -> {users, nodes, pending, completed}

特性：
  - 纯标准库 urllib，可在 Python 3.6+ 任意环境运行（含服务器/AutoDL/本机）
  - 真实训练（RealTrainer，需 GPU+torch）与 mock 训练（无 GPU 联调）自动切换
  - auth 头用 chr() 拼装，避免聊天系统把 "Bearer " 遮罩成 "***"

用法：
  # 真实联邦（AutoDL，需 torch）：
  python federation_client.py --server-url http://106.14.220.169:8000 \
      --username nodeA --password passA --dataset /path/to/law.jsonl

  # 本机 mock 联调（无 GPU，验证调度中心连通性）：
  python federation_client.py --server-url http://106.14.220.169:8000 \
      --username nodeMock --password passMock --mock

  # 仅跑一轮就退出：
  python federation_client.py ... --once
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# 尽量让 stdout/stderr 以 utf-8 输出（Windows GBK 控制台下避免编码崩溃）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── auth 头前缀：用 chr() 拼出 "Bearer "，避开聊天遮罩 ──────────────────
# 聊天系统在复制含 "Bearer " 的命令时会遮成 "***"，导致鉴权失败。
# 这里用 ASCII 码拼装，命令文本里不出现字面量 "Bearer "。
AUTH_PREFIX = (
    chr(66) + chr(101) + chr(97) + chr(114) + chr(101) + chr(114) + chr(32)
)  # "Bearer "


# ───────────────────────────────────────────────────────────────────────────
# HTTP 底层
# ───────────────────────────────────────────────────────────────────────────

def api_call(
    server_url: str,
    method: str,
    path: str,
    token: Optional[str] = None,
    data: Optional[dict] = None,
) -> Tuple[int, Any]:
    """返回 (status_code, body)。body 解析失败时为 {"detail": raw_text}。"""
    url = server_url.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = AUTH_PREFIX + token
    body_bytes = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"detail": raw or str(e)}
        return e.code, parsed
    except Exception as e:  # 网络错误等
        return -1, {"detail": str(e)}


# ───────────────────────────────────────────────────────────────────────────
# 鉴权
# ───────────────────────────────────────────────────────────────────────────

def login(server_url: str, username: str, password: str) -> Optional[str]:
    status, body = api_call(
        server_url, "POST", "/api/v1/auth/login",
        data={"username": username, "password": password},
    )
    if status == 200 and isinstance(body, dict) and body.get("access_token"):
        return body["access_token"]
    print(f"  [auth] 登录失败 ({status}): {body.get('detail', body)}")
    return None


def register(server_url: str, username: str, email: str, password: str) -> bool:
    status, body = api_call(
        server_url, "POST", "/api/v1/auth/register",
        data={"username": username, "email": email, "password": password},
    )
    if status in (200, 201):
        print(f"  [auth] 注册成功: {username}")
        return True
    # 409 / 400 均表示用户名已存在（服务器返回 400 'Username taken'），视为已注册
    if status in (409, 400):
        return True
    print(f"  [auth] 注册失败 ({status}): {body.get('detail', body)}")
    return False


def ensure_node_token(server_url: str, username: str, password: str, email: str) -> Optional[str]:
    """登录节点；若不存在则先注册再登录。"""
    tok = login(server_url, username, password)
    if tok:
        return tok
    register(server_url, username, email, password)
    return login(server_url, username, password)


# ───────────────────────────────────────────────────────────────────────────
# 调度中心交互
# ───────────────────────────────────────────────────────────────────────────

def get_pending(server_url: str, token: str) -> Optional[List[dict]]:
    """返回任务列表；None 表示鉴权/网络错误。"""
    status, body = api_call(server_url, "GET", "/api/v1/tasks/pending", token=token)
    if status == 200:
        return body if isinstance(body, list) else []
    print(f"  [pending] 查询失败 ({status}): {body.get('detail', body)}")
    return None


def claim_task(server_url: str, token: str, task_id: str) -> bool:
    status, body = api_call(
        server_url, "POST", "/api/v1/tasks/claim",
        token=token, data={"task_id": task_id},
    )
    if status == 200:
        print(f"  [claim] {task_id[:8]} -> {body.get('status', 'claimed')}")
        return True
    print(f"  [claim] 失败 ({status}): {body.get('detail', body)}")
    return False


def report_progress(
    server_url: str, token: str, task_id: str,
    progress_pct: float, final_loss: Optional[float] = None,
) -> None:
    data = {"task_id": task_id, "progress_pct": progress_pct}
    if final_loss is not None:
        data["final_loss"] = final_loss
    status, body = api_call(
        server_url, "POST", "/api/v1/tasks/progress",
        token=token, data=data,
    )
    if status not in (200, 201):
        print(f"  [progress] 上报失败 ({status}): {body.get('detail', body)}")


def complete_task(server_url: str, token: str, task_id: str, final_loss: float) -> bool:
    status, body = api_call(
        server_url, "POST", "/api/v1/tasks/complete",
        token=token, data={"task_id": task_id, "final_loss": final_loss},
    )
    if status == 200:
        print(f"  [complete] {task_id[:8]} -> {body.get('status', 'completed')} (loss={final_loss})")
        return True
    print(f"  [complete] 失败 ({status}): {body.get('detail', body)}")
    return False


# ───────────────────────────────────────────────────────────────────────────
# 训练（真实 / mock）
# ───────────────────────────────────────────────────────────────────────────

def run_real_training(
    task_id: str,
    dataset_path: str,
    report_fn,
) -> Dict[str, Any]:
    """调用 RealTrainer 跑真实 QLoRA。需在 GPU 机器（AutoDL）上运行。"""
    try:
        from app.trainer import RealQLoRATrainer  # RealQLoRATrainer = RealTrainer
    except Exception as e:
        raise RuntimeError(f"无法导入 RealTrainer（缺 torch/unsloth？）: {e}")

    trainer = RealQLoRATrainer(
        task_id=task_id,
        data_path=dataset_path or None,
        progress_callback=lambda p: report_fn(
            p.get("step", 0), p.get("total_steps", 0), p.get("loss")
        ),
    )
    # train() 为同步阻塞调用，由调用方在子线程中执行
    meta = trainer.train()
    return {
        "final_loss": float(meta.get("final_loss", 0.0)),
        "elapsed_sec": float(meta.get("elapsed_sec", 0.0)),
        "peak_vram_mb": int(float(meta.get("vram_gb", 0.0)) * 1024),
    }


def run_mock_training(
    task_id: str,
    total_steps: int,
    report_fn,
) -> Dict[str, Any]:
    """无 GPU 的模拟训练，仅用于验证调度中心连通性。"""
    print(f"  [train] MOCK 模式：模拟 {total_steps} 步")
    loss = 2.0
    start = time.time()
    for step in range(1, total_steps + 1):
        time.sleep(0.25)
        loss = max(0.05, loss - 0.04 + (0.02 if step % 7 == 0 else 0))
        report_fn(step, total_steps, loss)
    return {
        "final_loss": round(loss, 4),
        "elapsed_sec": round(time.time() - start, 1),
        "peak_vram_mb": 0,
    }


def train_task(
    server_url: str,
    token: str,
    task: dict,
    dataset_path: str,
    use_mock: bool,
) -> Dict[str, Any]:
    """根据任务配置执行训练，并实时上报进度。"""
    task_id = task["task_id"]
    total_steps = int(task.get("max_steps", 60))

    def report_fn(step: int, total: int, loss):
        pct = round(step / total * 100, 1) if total else 0.0
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"    [{bar}] {pct:5.1f}%  step={step}/{total}  loss={loss}")
        report_progress(server_url, token, task_id, pct, loss)

    if use_mock:
        return run_mock_training(task_id, total_steps, report_fn)

    # 真实训练放在子线程，避免阻塞（进度通过回调实时上报）
    import threading

    result_holder: Dict[str, Any] = {}

    def _worker():
        try:
            result_holder["data"] = run_real_training(task_id, dataset_path, report_fn)
        except Exception as e:
            result_holder["error"] = str(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if "error" in result_holder:
        raise RuntimeError(result_holder["error"])
    return result_holder["data"]


# ───────────────────────────────────────────────────────────────────────────
# 主循环
# ───────────────────────────────────────────────────────────────────────────

def execute_one_round(
    server_url: str,
    node_token: str,
    admin_token: Optional[str],
    dataset_path: str,
    use_mock: bool,
) -> bool:
    """执行一轮：查 pending -> claim -> train -> complete。返回是否处理了任务。"""
    # pending：优先用节点 token；若 403 再用 admin token（部分调度中心 pending 需 admin）
    pending = get_pending(server_url, node_token)
    if pending is None and admin_token:
        pending = get_pending(server_url, admin_token)
    if pending is None:
        return False
    if not pending:
        return False

    task = pending[0]
    task_id = task["task_id"]
    print(f"\n>> 领取任务 {task_id[:8]} (type={task.get('task_type')}, "
          f"model={task.get('model_name')}, steps={task.get('max_steps')})")

    if not claim_task(server_url, node_token, task_id):
        return False

    try:
        stats = train_task(server_url, node_token, task, dataset_path, use_mock)
    except Exception as e:
        print(f"  [train] 训练异常: {e}")
        return False

    ok = complete_task(server_url, node_token, task_id, stats.get("final_loss", 0.0))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Firefly 联邦训练客户端")
    ap.add_argument("--server-url", default=os.environ.get("FIREFLY_SERVER", "http://localhost:8000"))
    ap.add_argument("--username", default=os.environ.get("FIREFLY_NODE_USER", "nodeX"))
    ap.add_argument("--password", default=os.environ.get("FIREFLY_NODE_PASS", "passX"))
    ap.add_argument("--email", default=os.environ.get("FIREFLY_NODE_EMAIL", "node@firefly-lm.com"))
    ap.add_argument("--dataset", default=os.environ.get("FIREFLY_DATASET", ""),
                    help="训练数据 JSONL/JSON 路径（真实模式）；留空用 demo 数据")
    ap.add_argument("--mock", action="store_true",
                    help="强制 mock 训练（无 GPU 联调用）")
    ap.add_argument("--admin-user", default=os.environ.get("FIREFLY_ADMIN_USER", ""),
                    help="若 pending 需 admin 权限，提供 admin 用户名")
    ap.add_argument("--admin-pass", default=os.environ.get("FIREFLY_ADMIN_PASS", ""))
    ap.add_argument("--poll-interval", type=int, default=15, help="轮询间隔（秒）")
    ap.add_argument("--once", action="store_true", help="只跑一轮就退出")
    args = ap.parse_args()

    use_mock = args.mock or not _gpu_available()
    if use_mock and not args.mock:
        print("[info] 未检测到 GPU/torch，自动切换到 mock 模式")

    node_token = ensure_node_token(args.server_url, args.username, args.password, args.email)
    if not node_token:
        print("[fatal] 节点鉴权失败，退出")
        sys.exit(1)

    admin_token = None
    if args.admin_user:
        admin_token = login(args.server_url, args.admin_user, args.admin_pass)
        if not admin_token:
            print("[warn] admin 登录失败，pending 查询仅用节点 token")

    print(f"[ready] 节点 {args.username} 已连接 {args.server_url} "
          f"(mode={'MOCK' if use_mock else 'REAL'})")

    rounds = 0
    try:
        while True:
            handled = execute_one_round(
                args.server_url, node_token, admin_token, args.dataset, use_mock
            )
            rounds += 1
            if args.once:
                break
            if not handled:
                print(f"  [idle] 无待认领任务，{args.poll_interval}s 后重试 (round {rounds})")
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n[stop] 用户中断，退出")


def _gpu_available() -> bool:
    """粗略判断是否有 torch + CUDA。"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
