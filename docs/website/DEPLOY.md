# Firefly LM 官网部署指南

## 文件清单

| 文件 | 用途 |
|------|------|
| `index.html` | 官网首页（Hero + 三种模式 + 工作流 + 教育案例 + 隐私 + CTA） |
| `workspace.html` | 登录后工作区（上传/训练/下载/聊天/历史/积分） |
| `style.css` | 共享样式（设计令牌、Bento、卡片、按钮、响应式） |
| `api.js` | 共享 API 封装（fetch 封装、auth、toast、格式化） |
| `DEPLOY.md` | 本文件 |

## 部署方式一：Vercel（推荐）

### 1. 推送到 GitHub

```bash
cd firefly-website
git init
git add .
git commit -m "feat: Firefly LM 官网 v0.6.3"
git remote add origin https://github.com/firefly-lm-org/firefly-website.git
git push -u origin main
```

### 2. 连接 Vercel

1. 打开 https://vercel.com/new
2. Import 你的 `firefly-lm-org/firefly-website` 仓库
3. Framework Preset 选 `Other`（纯静态）
4. Build Command 留空，Output Directory 留空
5. 点 Deploy

### 3. 自定义域名

1. Vercel 面板 → Settings → Domains
2. 添加 `firefly-lm.com`
3. 在域名服务商处添加 CNAME 记录指向 `cname.vercel-dns.com`
4. 等待 SSL 证书自动签发（Let's Encrypt）

### 4. 自动部署

每次 `git push` 到 main 分支，Vercel 自动部署，约 30 秒生效。

## 部署方式二：服务器 nginx 静态托管

### 1. 上传文件

```bash
# 本地
scp -i $env:TEMP\ff_key D:\firefly\firefly-website\* admin@106.14.220.169:/tmp/website/

# SSH
ssh -i $env:TEMP\ff_key admin@106.14.220.169
sudo mkdir -p /usr/share/nginx/firefly-web
sudo cp /tmp/website/* /usr/share/nginx/firefly-web/
sudo chown -R nginx:nginx /usr/share/nginx/firefly-web/
```

### 2. nginx 配置

```nginx
# /etc/nginx/conf.d/firefly.conf
server {
    listen 80;
    server_name firefly-lm.com www.firefly-lm.com;

    root /usr/share/nginx/firefly-web;
    index index.html;

    # SPA fallback for workspace
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(css|js|png|jpg|svg)$ {
        expires 7d;
        add_header Cache-Control "public, max-age=604800";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer" always;
}
```

### 3. 重载 nginx

```bash
sudo nginx -t && sudo systemctl reload nginx
```

### 4. HTTPS（Let's Encrypt）

```bash
sudo yum install -y certbot python3-certbot-nginx
sudo certbot --nginx -d firefly-lm.com -d www.firefly-lm.com
# 自动续期已配置
```

## API 地址配置

网页端通过 `api.js` 自动判断环境：

| 环境 | API 地址 |
|------|----------|
| `localhost` / `127.0.0.1` | `http://106.14.220.169:8000` |
| 其他（含 firefly-lm.com） | `https://api.firefly-lm.com` |

切换域名只需在 DNS 添加：
```
api.firefly-lm.com → CNAME → cname.vercel-dns.com
```
或在 nginx 加一个 server block 反代到 `127.0.0.1:8000`。

## 验证清单

部署完成后逐项验证：

- [ ] `https://firefly-lm.com/` 打开首页，Hero 区正常显示
- [ ] 三种训练模式卡片正确渲染（GPU/CPU/付费）
- [ ] 工作流五步图正常显示
- [ ] 教育案例区正常显示
- [ ] 隐私承诺区正常显示
- [ ] 点击「立即体验」弹出登录框
- [ ] 登录成功后跳转到 `workspace.html`
- [ ] 工作区侧边栏 6 个入口可切换
- [ ] 上传文件区可拖拽 + 点击选择
- [ ] 训练模式三选一正常
- [ ] 移动端（iPhone Safari）打开显示正常
- [ ] 浏览器控制台无 JS 错误

## 更新流程

```bash
# 修改后
git add .
git commit -m "fix: 修复 xxx"
git push
# Vercel 自动部署，约 30 秒后生效
```

## 回滚

Vercel 面板 → Deployments → 选上一个版本 → Promote to Production。
或：
```bash
git revert HEAD
git push
```
