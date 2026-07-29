"""
firefly-client · 服务端补丁说明
=====================================
客户端 v0.1 使用 multipart/form-data 上传结果文件，
需要在 scheduler 的 task.py 中新增一个接口：

    POST /api/v1/task/submit-file

该接口接收：
    - file: 上传的结果 zip 文件
    - task_id: 任务 ID
    - result_sha256: 文件 SHA256
    - execution_time_sec: 执行时长
    - peak_vram_mb: 峰值显存
    - total_steps: 总步数

该接口负责：
    1. 将文件保存到 MinIO
    2. 验证 SHA256
    3. 更新 task 记录
    4. 触发校验流程

以下是该接口的参考实现，请复制到 scheduler 的 app/routers/task.py 中：
"""

# ── 参考实现（复制到 scheduler） ──────────────────
"""
from fastapi import UploadFile, File, Form

@router.post("/submit-file")
async def submit_task_file(
    file: UploadFile = File(...),
    task_id: str = Form(...),
    result_sha256: str = Form(...),
    execution_time_sec: float = Form(...),
    peak_vram_mb: float = Form(...),
    total_steps: int = Form(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    node = await get_active_node(user, db)

    # 查找任务
    result = await db.execute(
        select(Task).where(
            Task.id == task_id,
            Task.claimed_by == node.id,
            Task.status.in_(["claimed", "running"]),
        ).limit(1)
    )
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="No active task found")

    # 读取文件内容
    file_data = await file.read()

    # 验证 SHA256
    import hashlib
    actual_sha256 = hashlib.sha256(file_data).hexdigest()
    if actual_sha256 != result_sha256:
        # 校验失败
        node.reputation_score = max(0, node.reputation_score - 10)
        task.retry_count += 1
        if task.retry_count >= task.max_retries:
            task.status = "failed"
        else:
            task.status = "pending"
            task.claimed_by = None
        await db.flush()
        raise HTTPException(status_code=400, detail="SHA256 mismatch")

    # 上传到 MinIO
    object_name = f"results/{task_id}/{file.filename}"
    import asyncio
    from app.utils.minio_client import minio_client
    await asyncio.to_thread(
        minio_client.put_object,
        settings.minio_bucket,
        object_name,
        data=io.BytesIO(file_data),
        length=len(file_data),
        content_type=file.content_type or "application/zip",
    )

    # 更新任务
    task.status = "completed"
    task.result_object_name = object_name
    task.result_sha256 = actual_sha256
    task.completed_at = datetime.utcnow()
    node.status = "online"

    await db.flush()

    return {
        "status": "accepted",
        "task_id": task.id,
        "message": "Result uploaded and verified",
    }
"""
