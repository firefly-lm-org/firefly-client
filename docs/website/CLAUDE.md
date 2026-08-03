# Firefly LM 官网开发指南

## 项目定位

火种 Firefly LM 的官方网站，包含：
- **首页**（`index.html`）：产品介绍、三种训练模式、工作流、教育案例、隐私承诺、CTA
- **工作区**（`workspace.html`）：登录后的训练/下载/聊天/历史/积分

## 技术约束

- **零构建**：纯 HTML + CSS + Vanilla JS，不引入 npm/webpack/Vite
- **零依赖**：不引入 React/Vue/jQuery，所有交互用原生 JS
- **单文件可打开**：每个 HTML 文件可双击在浏览器打开预览（除 API 调用外）
- **Vercel 自动部署**：推送到 `firefly-lm-org/firefly-website` 的 main 分支即自动部署

## 文件结构

```
firefly-website/
├── index.html       # 官网首页
├── workspace.html   # 登录后工作区
├── style.css        # 共享样式
├── api.js           # 共享 API 封装
├── DEPLOY.md        # 部署指南
├── README.md        # 项目说明
└── CLAUDE.md       # 本文件
```

## 设计令牌（必须遵循）

所有颜色/尺寸通过 CSS 变量定义在 `style.css` 的 `:root` 中：

```css
:root {
    --bg: #0A0A0A;          /* 页面背景 */
    --surface-2: #1A1A1A;    /* 卡片背景 */
    --text: #E8E8E8;         /* 主文字 */
    --accent: #FF6B00;        /* 品牌主色（火种橙）*/
    --green: #4CAF50;         /* 成功/免费 */
    --blue: #2196F3;          /* 付费 */
    --radius: 12px;            /* 卡片圆角 */
}
```

**禁止**：在 HTML 中硬编码颜色值。一律用 `var(--xxx)`。

## API 调用规范

所有 API 调用必须通过 `api.js` 中的 `Firefly` 对象：

```javascript
// ✅ 正确
const data = await Firefly.submitTrain(file, 'law');
Firefly.showToast('完成', 'success');

// ❌ 错误：直接 fetch
fetch('http://106.14.220.169:8000/api/...')
```

API 地址自动判断：
- `localhost` / `127.0.0.1` → `http://106.14.220.169:8000`
- 其他 → `https://api.firefly-lm.com`

## 新增页面的流程

1. 在 `index.html` 或 `workspace.html` 中添加入口链接
2. 如果是独立页面，复制 `workspace.html` 的结构（nav + main + script 引入 api.js）
3. 页面内容用 `switchTab()` 模式渲染到 `#mainContent`
4. 所有 API 调用走 `Firefly` 对象

## 修改设计规范

- **按钮**：主操作唯一橙色（`--accent`），次要用 ghost（透明+边框）
- **卡片**：`background: var(--surface-2)`，圆角 `--radius`，边框 `var(--border-glass)`
- **动效**：仅 `transition: 0.15s-0.3s`，尊重 `prefers-reduced-motion`
- **移动端**：640px 断点，按钮高度 ≥44px，表格横向滚动

## 禁止事项

- ❌ 引入任何 JS 框架或 UI 库
- ❌ 硬编码 API 地址或 Token
- ❌ 在 HTML 中写内联样式（除动态生成外）
- ❌ 使用 `alert()` 或 `confirm()`（用 `Firefly.showToast()`）
- ❌ 在官网首页暴露"调度中心"或 Swagger 文档链接

## 部署检查清单

推送前确认：
- [ ] CSS 变量无硬编码颜色
- [ ] 所有 fetch 走 `Firefly` 对象
- [ ] 移动端（iPhone Safari）测试通过
- [ ] 无 console.error
- [ ] `prefers-reduced-motion` 下无动画
- [ ] 登录态跳转逻辑正确

## 测试账号

| 环境 | 地址 | 说明 |
|------|------|------|
| 本地预览 | `file://` 或 `localhost:8000` | 双击 HTML 或起静态服务器 |
| Vercel 预览 | `https://firefly-lm.com` | `git push` 后 30 秒生效 |
| API 测试 | `http://106.14.220.169:8000/docs` | Swagger UI（仅 localhost 可见） |
