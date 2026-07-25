# 推送命令（覆盖更新）

## 1. 备份并清理
```bash
cd D:\firefly-client
# 备份旧版（如有）
ren firefly-client firefly-client.bak
```

## 2. 解压新文件
把本 zip 解压到 `D:\firefly-client\firefly-client\` 覆盖：
- `app/trainer/real_trainer.py` ← 覆盖
- `requirements.txt` ← 覆盖
- `_headers` ← 新增
- `README.md` ← 覆盖

## 3. 验证语法
```bash
cd D:\firefly-client\firefly-client
python -X utf8 -c "import ast; ast.parse(open('app/trainer/real_trainer.py',encoding='utf-8').read()); print('real_trainer.py 语法 OK')"
```

## 4. 推送到 GitHub
```bash
git add app/trainer/real_trainer.py requirements.txt _headers README.md
git commit -m "feat(trainer): env-var model path + accelerate/safetensors deps"
git push
```

## 5. 验证（无 GPU 机器也能跑 mock）
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
firefly start --mock
```

## 6. 有 GPU 时
```bash
set FIREFLY_MODEL_PATH=unsloth/Qwen3-1.5B-Instruct-4bit
firefly start
```
