#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器端防作弊 L1 补丁
在 /api/v1/tasks/complete 端点加：
1. 文件大小校验（weight_path 指向的文件不能小于 1KB）
2. weight_hash 记录（SHA256 前 16 位）
3. 异常检测：final_loss 异常低或 weight_path 不存在时扣信誉分
"""
import re

MAIN_PY = "/root/scheduler/main.py"

with open(MAIN_PY, "r", encoding="utf-8") as f:
    code = f.read()

# 旧 complete 函数体
OLD_COMPLETE = '''@app.post("/api/v1/tasks/complete")
def complete(body: dict, u=Depends(me), db: Session = Depends(get_db)):
    t = db.query(TrainingTask).filter(TrainingTask.id == body["task_id"]).first()
    if not t: raise HTTPException(404, "Task not found")
    t.status = "completed"
    t.final_loss = body.get("final_loss", t.final_loss)
    t.weight_path = body.get("weight_path", t.weight_path)
    t.progress_pct = 100.0
    t.updated_at = datetime.datetime.utcnow(); db.commit()
    _adjust_rep(body.get("node_name", u.username), 5,
                      "Task completed successfully", db)
    return {"task_id": t.id, "status": "completed"}'''

# 新 complete 函数体（加防作弊 L1）
NEW_COMPLETE = '''@app.post("/api/v1/tasks/complete")
def complete(body: dict, u=Depends(me), db: Session = Depends(get_db)):
    t = db.query(TrainingTask).filter(TrainingTask.id == body["task_id"]).first()
    if not t: raise HTTPException(404, "Task not found")

    # --- 防作弊 L1 ---
    weight_path = body.get("weight_path", t.weight_path or "")
    final_loss = body.get("final_loss", t.final_loss)

    # 1. 检查 final_loss 合理性（不能为负、不能是 0）
    suspicious = False
    reasons = []
    if final_loss is not None and (final_loss < 0 or final_loss == 0):
        suspicious = True
        reasons.append("abnormal_loss:{}".format(final_loss))

    # 2. 检查 weight_path 是否存在（如果是本地路径）
    if weight_path and weight_path.startswith("/"):
        import os
        if not os.path.exists(weight_path):
            suspicious = True
            reasons.append("weight_file_missing")
        else:
            fsize = os.path.getsize(weight_path)
            if fsize < 1024:  # 小于 1KB 可疑
                suspicious = True
                reasons.append("weight_file_too_small:{}".format(fsize))
            # 记录 SHA256 前 16 位
            import hashlib
            sha = hashlib.sha256()
            with open(weight_path, "rb") as wf:
                for chunk in iter(lambda: wf.read(8192), b""):
                    sha.update(chunk)
            t.weight_hash = sha.hexdigest()[:16]

    # 3. 检查同一节点是否短时间内提交多次（刷分检测）
    recent = db.query(TrainingTask).filter(
        TrainingTask.node_name == body.get("node_name", u.username),
        TrainingTask.status == "completed",
        TrainingTask.updated_at > datetime.datetime.utcnow() - datetime.timedelta(minutes=5)
    ).count()
    if recent > 3:
        suspicious = True
        reasons.append("rapid_submissions:{}".format(recent))

    # --- 更新任务状态 ---
    t.status = "completed"
    t.final_loss = final_loss
    t.weight_path = weight_path
    t.progress_pct = 100.0
    t.updated_at = datetime.datetime.utcnow()

    # 加 weight_hash 列如果不存在（容错）
    try:
        db.commit()
    except Exception:
        # weight_hash 列可能不存在，回滚去掉这个字段
        db.rollback()
        t.weight_hash = None
        db.commit()

    if suspicious:
        _adjust_rep(body.get("node_name", u.username), -50,
                          "Anti-cheat L1: " + ", ".join(reasons), db)
        return {"task_id": t.id, "status": "rejected",
                "reason": "weight_anomaly", "details": reasons}
    else:
        _adjust_rep(body.get("node_name", u.username), 5,
                          "Task completed successfully", db)
        return {"task_id": t.id, "status": "completed",
                "weight_hash": getattr(t, "weight_hash", None)}'''

if OLD_COMPLETE not in code:
    print("ERROR: 旧 complete 函数未找到，补丁无法应用")
    print("可能已经被打补丁了，或代码格式有变化")
    exit(1)

code = code.replace(OLD_COMPLETE, NEW_COMPLETE)

with open(MAIN_PY, "w", encoding="utf-8") as f:
    f.write(code)

print("OK: 防作弊 L1 补丁已应用")
print("修改内容:")
print("  1. final_loss 合理性检查（不能为负或零）")
print("  2. weight_path 文件存在性 + 大小检查（< 1KB 可疑）")
print("  3. weight_hash 记录（SHA256 前 16 位）")
print("  4. 短时间多次提交检测（5 分钟内 > 3 次可疑）")
print("  5. 可疑时信誉分 -50，返回 rejected")
