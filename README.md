# AI 访谈原型

基于 FastAPI 的 AI 访谈应用，支持多阶段对话访谈、访问口令保护和图片生成功能。


> 🌐 **[在线宣传页](https://samge0.github.io/ai-interview/)** — 可视化了解功能特性与工作流程

## 功能特点

- **多阶段访谈**：支持 5 个阶段的访谈流程，每个阶段有独立的系统提示词
- **访问口令**：可选的访问口令保护，避免公网暴露后被恶意访问
- **图片生成**：集成图片生成 API（DALL·E/Imagen/Gemini 等）
- **灵活配置**：支持 OpenAI 兼容的各种 API 提供商
- **持久化存储**：访谈记录自动保存到本地

## 环境要求

- Docker 部署：Docker 和 Docker Compose
- 源码部署：Python 3.12+ 和 uv

## 部署方式

### 方式一：Docker 部署（推荐）

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd AI-Interview
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   ```

   编辑 `.env` 文件，填入你的 API 配置：
   ```env
   OPENAI_API_KEY=your_api_key_here
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_MODEL=gpt-4

   # 图片生成配置（可选）
   IMAGE_API_KEY=your_image_api_key
   IMAGE_API_BASE_URL=https://api.example.com
   IMAGE_API_MODEL=dall-e-3

   # 访问口令（可选）
   ACCESS_CODE=your_secret_code
   ```

3. **启动服务**
   ```bash
   docker-compose up -d
   ```

4. **访问应用**
   - 应用地址：http://localhost:8000
   - 健康检查：http://localhost:8000/api/health

5. **常用命令**
   ```bash
   # 查看日志
   docker-compose logs -f

   # 停止服务
   docker-compose down

   # 重新构建并启动
   docker-compose up -d --build
   ```

### 方式二：源码部署（使用 uv）

1. **安装 uv**
   ```bash
   # Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex

   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **克隆项目**
   ```bash
   git clone <repository-url>
   cd AI-Interview
   ```

3. **创建虚拟环境并安装依赖**
   ```bash
   # 创建虚拟环境
   uv venv

   # 激活虚拟环境
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate

   # 安装依赖
   uv pip install -r requirements.txt
   ```

4. **配置环境变量**
   ```bash
   cp .env.example .env
   ```

   编辑 `.env` 文件，填入你的 API 配置（参考 Docker 部署方式）

5. **启动应用**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **访问应用**
   - 应用地址：http://localhost:8000
   - API 文档：http://localhost:8000/docs

## 环境变量说明

| 变量名 | 必填 | 说明 | 默认值 |
|--------|------|------|--------|
| `OPENAI_API_KEY` | 是 | LLM API 密钥 | - |
| `OPENAI_BASE_URL` | 否 | LLM API 基础地址 | `https://api.newcoin.tech` |
| `OPENAI_MODEL` | 否 | 使用的模型名称 | `doubao-seed-2-0-pro-260215` |
| `IMAGE_API_KEY` | 否 | 图片生成 API 密钥 | - |
| `IMAGE_API_BASE_URL` | 否 | 图片生成 API 地址 | - |
| `IMAGE_API_MODEL` | 否 | 图片生成模型 | `gemini-2.5-flash-image` |
| `ACCESS_CODE` | 否 | 访问口令，留空则不需要验证 | - |

## 项目结构

```
AI-Interview/
├── main.py              # FastAPI 应用主文件
├── templates/           # HTML 模板
│   └── index.html       # 前端页面
├── output/              # 访谈记录输出目录
├── Dockerfile           # Docker 镜像构建文件
├── docker-compose.yml   # Docker Compose 配置
├── requirements.txt     # Python 依赖列表
├── .env.example         # 环境变量示例
└── README.md            # 项目说明文档
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/health` | GET | 健康检查 |
| `/api/chat` | POST | 聊天接口 |
| `/api/advance-stage` | POST | 推进访谈阶段 |
| `/api/verify-code` | POST | 验证访问口令 |

## License

MIT
