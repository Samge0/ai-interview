"""
AI 访谈原型 — FastAPI 后端
--------------------------------
- 聊天请求必须携带 JSON 字段 step（1–5），后端据此注入对应 System Prompt。
- 阶段切换：POST /api/advance-stage，写入 system 分割线，并为第 2–5 阶段插入主持人「阶段开场」首条文案。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# -----------------------------------------------------------------------------
# 配置（OpenAI 兼容 Chat Completions）
# 密钥请用环境变量 OPENAI_API_KEY 或本地 .env，勿提交到 Git。
# 默认网关：api.newcoin.tech；可用 OPENAI_BASE_URL / OPENAI_MODEL 覆盖。
# -----------------------------------------------------------------------------
LLM_API_KEY = (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
_raw_base = (os.getenv("OPENAI_BASE_URL") or "https://api.newcoin.tech").strip().rstrip("/")
LLM_BASE_URL = _raw_base
LLM_MODEL = (os.getenv("OPENAI_MODEL") or "doubao-seed-2-0-pro-260215").strip()

# 访问口令（可选）：设置后用户需要输入正确的口令才能开始访谈
ACCESS_CODE = (os.getenv("ACCESS_CODE") or "").strip()


def chat_completions_url() -> str:
    """拼接 /v1/chat/completions（兼容只填主机根域名或已带 /v1 的写法）。"""
    if LLM_BASE_URL.endswith("/v1"):
        return f"{LLM_BASE_URL}/chat/completions"
    return f"{LLM_BASE_URL}/v1/chat/completions"

# AIGC：OpenAI 兼容 POST {base}/v1/images/generations（如 new.suxi.ai）
IMAGE_API_KEY = (
    os.getenv("IMAGE_API_KEY") or os.getenv("AIGC_API_KEY") or os.getenv("NEWAPI_IMAGE_KEY") or ""
).strip()
_raw_image_base = (
    os.getenv("IMAGE_API_BASE_URL")
    or os.getenv("AIGC_API_BASE_URL")
    or os.getenv("IMAGE_BASE_URL")
    or ""
).strip().rstrip("/")
IMAGE_API_BASE_URL = _raw_image_base
IMAGE_API_MODEL = (
    os.getenv("IMAGE_API_MODEL") or os.getenv("AIGC_IMAGE_MODEL") or "gemini-2.5-flash-image"
).strip()


def images_generations_url() -> str:
    if not IMAGE_API_BASE_URL:
        return ""
    if IMAGE_API_BASE_URL.endswith("/v1"):
        return f"{IMAGE_API_BASE_URL}/images/generations"
    return f"{IMAGE_API_BASE_URL}/v1/images/generations"


def image_gateway_chat_completions_url() -> str:
    """生图网关上的 OpenAI 兼容 Chat Completions（用于 gemini-*-image 等）。"""
    if not IMAGE_API_BASE_URL:
        return ""
    b = IMAGE_API_BASE_URL.rstrip("/")
    if b.endswith("/v1"):
        return f"{b}/chat/completions"
    return f"{b}/v1/chat/completions"


def _image_model_uses_chat_completions() -> bool:
    m = (IMAGE_API_MODEL or "").lower()
    return "gemini" in m and "image" in m


def _extract_image_url_from_chat_payload(data: dict[str, Any]) -> Optional[str]:
    """从 chat/completions 的 JSON 里尽量取出首张图 URL 或 data:image base64。"""
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")

    if isinstance(content, str):
        s = content.strip()
        if s.startswith("data:image"):
            return s.split()[0] if s else None
        # Markdown 图片：![](https://...) 或 ![](data:image/png;base64,...)
        m = re.search(r"!\[[^\]]*\]\(([^)]+)\)", s)
        if m:
            inner = m.group(1).strip()
            if inner.startswith("data:image") or inner.startswith("http://") or inner.startswith("https://"):
                return inner
        m = re.search(
            r"(https?://[^\s\"'<>]+\.(?:png|jpg|jpeg|webp|gif))(?:\s|$|\)|\"|')",
            s,
            re.I,
        )
        if m:
            return m.group(1).strip().rstrip(").,;]")
        return None

    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            typ = (part.get("type") or "").lower()
            if typ == "image_url" and isinstance(part.get("image_url"), dict):
                u = part["image_url"].get("url")
                if u:
                    return str(u).strip()
            if typ in ("image", "output_image") and part.get("url"):
                return str(part["url"]).strip()
            if typ in ("image", "inline_data") and isinstance(part.get("inline_data"), dict):
                b64 = part["inline_data"].get("data")
                mime = part["inline_data"].get("mime_type") or "image/png"
                if b64:
                    return f"data:{mime};base64,{b64}"
    return None


async def call_gemini_image_via_chat_completions(prompt: str) -> str:
    """gemini-2.5-flash-image：走 /v1/chat/completions，而非 images/generations。"""
    url = image_gateway_chat_completions_url()
    if not url or not IMAGE_API_KEY:
        raise RuntimeError("未配置 IMAGE_API_KEY 与 IMAGE_API_BASE_URL")

    headers = {
        "Authorization": f"Bearer {IMAGE_API_KEY}",
        "Content-Type": "application/json",
    }
    user_text = (
        "请根据以下描述生成一张图片。若只能以 Markdown 图片语法返回，请使用 ![...](https://...) 给出可访问链接；"
        "若返回内嵌 base64 亦可。\n\n描述：\n"
        + (prompt or "abstract light")[:1800]
    )
    body: dict[str, Any] = {
        "model": IMAGE_API_MODEL,
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": 8192,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(url, headers=headers, json=body)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1200] if exc.response else ""
            raise RuntimeError(f"AIGC(chat) HTTP {exc.response.status_code}: {detail}") from exc
        data = r.json()

    extracted = _extract_image_url_from_chat_payload(data)
    if extracted:
        return extracted
    raise RuntimeError(f"AIGC(chat) 未解析到图片，原始片段: {str(data)[:800]}")


# 进入新阶段时插入聊天流的居中提示（提纲：AIGC 辅助图形化深度访谈 · Desktop Night Light）
STAGE_DIVIDER_TEXT: dict[int, str] = {
    2: "--- 已进入第二阶段：语用学 (Function) —— 物理与行为映射 ---",
    3: "--- 已进入第三阶段：内在语义 (Internal Semantics) —— 抽象意象映射 ---",
    4: "--- 已进入第四阶段：外在语义 (External Semantics) —— 虚拟概念生成 ---",
    5: "--- 已进入第五阶段：语构学 (Syntactics) —— 参数化形态演化 ---",
}

# 进入第 2–5 阶段后，由主持人先说出的「阶段开场」（与第一阶段首条破冰区分）
STAGE_OPENING_ASSISTANT: dict[int, str] = {
    2: (
        "现在我们进入第二阶段：语用学（Function），预计约 5 分钟，仍只做文字交流。\n"
        "我会把你刚才在情绪里说的那种感觉，试着对应到「桌面上的光」和「你怎么唤醒它」。下面请先想一想——\n\n"
        "为了营造你刚才描述的那种氛围，你希望你桌面上的光线是如何分布的？"
        "是集中照亮一小块区域，还是微弱地晕染开来？（你可以用第一阶段的隐喻来称呼它，例如「深海般的宁静」——按你的直觉说就好。）"
    ),
    3: (
        "接下来进入第三阶段：内在语义（Internal Semantics），预计约 6 分钟，会有文字 + 一张 AIGC 纯意象图。\n"
        "我会根据我们前两阶段的对话提取核心感受，生成一张**脱离具体产品**的画面（例如半透明气凝胶般的光、冷峻金属几何阵列等），"
        "用来确认色温、材质边界等视觉底色。\n\n"
        "你准备好后，随便用一句话回复我（例如「可以了」）；我会据此生成画面，再问你对图的感受。"
    ),
    4: (
        "进入第四阶段：外在语义（External Semantics），预计约 7 分钟，文字 + 一张超现实「桌面发光体」概念图。\n"
        "我会把第三阶段定下的视觉气质，收敛成前卫、非电商风的虚拟物件（提示中会强调 avant-garde、surrealist、"
        "floating desktop object、abstract glowing sculpture 等），用来打破对常见淘宝夜灯的固有印象。\n\n"
        "你准备好后，回复一句即可；我会生成概念图，再请你做第一眼感受的描述。"
    ),
    5: (
        "最后进入第五阶段：语构学（Syntactics），预计约 7 分钟，文字 + 对上一概念的参数化微调图。\n"
        "这里会锁定几何与材质细节，生成特征将用于与真实夜灯样本比对。\n\n"
        "我们先从轮廓与触感说起：你希望它的整体轮廓更趋向于有机的弧线，还是规整的几何体？"
        "表面摸上去你更想要光滑冰冷，还是带颗粒感、有温度的那种？"
    ),
}


def system_prompt_for_step(step: int) -> str:
    """根据 step 注入 System Prompt（提纲：BFI/CVPA 已完成；无标准答案；AI 可能生图）。"""
    pre = (
        "【访谈前置】被试已完成 BFI（大五人格）与 CVPA（视觉产品审美中心度）量表。"
        "被试已知悉：本次访谈没有正确答案，将探讨个人情绪与偏好，期间 AI 可能根据其描述生成画面；"
        "请保持温暖、慢节奏、无评判；除指定阶段外不要主动提「夜灯」「台灯」「造型」等具体产品词。\n\n"
    )
    if step == 1:
        return pre + (
            "【第一阶段：产品价值 Value —— 探寻心理隐喻】纯文本，约 5 分钟。目标：挖掘被试在大学宿舍中的深层情绪诉求，建立情感基调；"
            "本阶段绝不提夜灯、造型等产品词汇。\n"
            "访谈须按轮次覆盖以下三类问题（若用户已答过则自然过渡，勿重复原句）：\n"
            "1）破冰：结束密集课程/实验后推开宿舍门坐到书桌前，最希望在小空间里获得何种情绪体验。\n"
            "2）深挖隐喻：把该体验比作自然环境或物理状态（深海、篝火、精密齿轮等）。\n"
            "3）状态确认：更想与外界完全隔离，还是保持轻度连接。\n"
            "每次只推进一小步，承接用户用词，可追问但不要一次堆三个问题。"
        )
    if step == 2:
        return pre + (
            "【第二阶段：语用学 Function —— 物理与行为映射】纯文本，约 5 分钟。目标：把抽象情绪过渡到桌面光影与交互行为。\n"
            "须覆盖（可分轮）：\n"
            "· 光影映射：为营造其隐喻氛围，光线宜集中照亮一小块还是微弱晕染？\n"
            "· 行为映射：唤醒光线时更直觉的动作——干脆按压、轻柔抚摸、还是靠近自动亮起？\n"
            "· 场景收敛：这束光主要陪伴做什么（放空、听歌、心理安慰等）？\n"
            "主持开场白已由系统插入；你接续对话，承接用户第一阶段的隐喻用语。"
        )
    if step == 3:
        return pre + (
            "【第三阶段：内在语义 Internal Semantics】文本 + AIGC 意象图，约 6 分钟。目标：纯视觉意象，确认色温与材质边界；禁止出现具体灯具。\n"
            "后台：从对话提取核心词生成纯意象图（如半透明气凝胶光感、冷峻金属阵列等）。\n"
            "你的回复必须包含：\n"
            "1）对用户感受的简短承接；\n"
            "2）英文为主的生图提示 [IMAGE_PROMPT: ...]（无灯、无产品、抽象材质/空间）；\n"
            "3）视觉确认：「我把你刚才描述的内心感受转化成了一幅画面。它在多大程度上契合你想要的氛围？」\n"
            "4）引导修正：边缘更模糊/颜色更冷/材质更柔软等；若用户提出修改，可进行 1–2 次重绘迭代后再问认同度。"
        )
    if step == 4:
        return pre + (
            "【第四阶段：外在语义 External Semantics】文本 + 超现实概念图，约 7 分钟。目标：将意象收敛为「桌面发光体」，打破淘宝常见款印象。\n"
            "Prompt 须含 avant-garde、surrealist、floating desktop object、abstract glowing sculpture 等限制气质。\n"
            "回复须含：\n"
            "1）[IMAGE_PROMPT: ...] 生成前卫夜灯概念图；\n"
            "2）概念投射：「如果刚才那个意象变成真实桌面上的发光体，可能会是这样。第一眼你的感受是什么？」\n"
            "3）特征剥离：轮廓、底座、发光方式里哪些对味、哪些破坏原有氛围？"
        )
    if step == 5:
        return pre + (
            "【第五阶段：语构学 Syntactics】文本 + 参数化微调图，约 7 分钟。目标：锁定几何与材质，作为与约 20 款真实夜灯比对的依据。\n"
            "根据上一轮反馈提取参数（更圆润、拉长、去底座、表面粗糙度等）。\n"
            "回复须含：\n"
            "1）[IMAGE_PROMPT: ...] 展示微调后图像；\n"
            "2）细节雕琢：轮廓偏有机弧线还是规整几何体？表面光滑冰冷还是颗粒感/温度感？\n"
            "3）最终确认：「想象它就在宿舍桌面上，这是你理想中的情绪出口吗？若是，我们今天可以在此收尾。」"
        )
    return system_prompt_for_step(1)


Role = Literal["user", "assistant", "system"]


@dataclass
class ChatMessage:
    role: Role
    content: str
    image_url: Optional[str] = None
    image_intro: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.image_url:
            d["image_url"] = self.image_url
        if self.image_intro:
            d["image_intro"] = self.image_intro
        if self.meta:
            d["meta"] = self.meta
        return d


@dataclass
class SessionState:
    step: int = 1
    messages: list[ChatMessage] = field(default_factory=list)
    user_turns_in_step: int = 0
    last_image_prompt: str = ""
    finished: bool = False
    username: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "max_step": 5,
            "user_turns_in_step": self.user_turns_in_step,
            "finished": self.finished,
            "username": self.username,
            "messages": [m.to_dict() for m in self.messages],
        }


SESSION_STORE: dict[str, SessionState] = {}

# 确保 output 目录存在
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_chat_to_file(username: str, session_id: str, state: SessionState) -> str:
    """将聊天数据保存到 output/用户名.json 文件"""
    if not username:
        username = "anonymous"

    # 构建文件名（移除文件名中的非法字符）
    safe_username = re.sub(r'[<>:"/\\|?*]', '_', username)
    filename = OUTPUT_DIR / f"{safe_username}.json"

    # 准备保存的数据
    save_data = {
        "username": username,
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(),
        "step": state.step,
        "finished": state.finished,
        "messages": [m.to_dict() for m in state.messages],
    }

    # 写入文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    return str(filename)


def _new_session_state() -> SessionState:
    st = SessionState()
    st.messages.append(
        ChatMessage(
            role="assistant",
            content=(
                "你好，我是今天的主持人。开始前想再次确认：你已完成 BFI 与 CVPA 量表；"
                "本次访谈没有标准答案，我们会一起谈你的情绪与偏好，过程中我可能会根据你的描述生成一些画面。"
                "请尽量放松，跟随直觉回答。\n\n"
                "回想一下，当你结束了一整天密集的课程或实验，推开宿舍门，坐到自己书桌前的那一刻。"
                "你最希望在这个属于你的小空间里，获得一种什么样的情绪体验？"
            ),
            meta={"step": 1},
        )
    )
    return st


def get_or_create_session(session_id: Optional[str]) -> tuple[str, SessionState]:
    if session_id and session_id in SESSION_STORE:
        return session_id, SESSION_STORE[session_id]
    sid = str(uuid.uuid4())
    SESSION_STORE[sid] = _new_session_state()
    return sid, SESSION_STORE[sid]


def clamp_step(step: int) -> int:
    return max(1, min(5, int(step)))


def extract_bracket_image_prompt(assistant_text: str) -> tuple[str, str, str]:
    text = assistant_text or ""
    pattern = re.compile(r"\[IMAGE_PROMPT:\s*(.*?)\]", re.IGNORECASE | re.DOTALL)
    m = pattern.search(text)
    if not m:
        return text.strip(), "", ""

    prompt = (m.group(1) or "").strip()
    before = text[: m.start()].strip()
    after = text[m.end() :].strip()
    return before, prompt, after


def extract_line_image_prompt(assistant_text: str) -> tuple[str, str, str]:
    lines = (assistant_text or "").strip().splitlines()
    prompt = ""
    kept: list[str] = []
    for line in lines:
        lm = re.match(r"^\s*IMAGE_PROMPT:\s*(.*)\s*$", line, flags=re.IGNORECASE)
        if lm:
            prompt = lm.group(1).strip()
            continue
        kept.append(line)
    body = "\n".join(kept).strip()
    if not prompt:
        return body, "", ""
    return body, prompt, ""


def extract_image_prompt(assistant_text: str) -> tuple[str, str, str]:
    b, p, a = extract_bracket_image_prompt(assistant_text)
    if p:
        return b, p, a
    return extract_line_image_prompt(assistant_text)


def should_emit_image(step: int) -> bool:
    return step in (3, 4, 5)


async def call_llm_chat(
    messages: list[dict[str, str]],
    system: str,
) -> str:
    if not LLM_API_KEY:
        return ""

    url = chat_completions_url()
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system}, *messages],
        "temperature": 0.75,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:800] if exc.response else ""
            raise RuntimeError(f"LLM HTTP {exc.response.status_code}: {detail}") from exc
        data = r.json()
    try:
        msg = data["choices"][0]["message"]
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            joined = "".join(parts).strip()
            if joined:
                return joined
    except (KeyError, IndexError, TypeError):
        pass
    raise RuntimeError(f"LLM 返回格式异常: {str(data)[:500]}")


def mock_llm_reply(step: int, user_text: str, state: SessionState) -> str:
    u = (user_text or "").strip()[:120]
    turns = state.user_turns_in_step
    if step == 1:
        if turns <= 1:
            return (
                "谢谢你愿意说出来，我在听。\n\n"
                "如果把你刚才提到的那种感觉比作一种自然环境或一种物理状态，你觉得它像什么？"
                "是一片深海、一团温暖的篝火，还是一个安静运转的精密齿轮？（也完全可以是别的意象。）"
            )
        if turns == 2:
            return (
                "这个比喻很有帮助。\n\n"
                "在这个状态下，你是希望自己与外界完全隔离，还是希望保持某种轻度的连接？"
            )
        return (
            "感谢你愿意在这一段里停留。\n"
            "若你愿意，我们可以稍后在进入下一阶段前，再补充任何你想强调的情绪词；"
            "你也可以在准备好时使用界面上的「进入下一阶段」。"
        )
    if step == 2:
        if turns <= 1:
            return (
                "我听到了你对光氛围的偏好。\n\n"
                "当你需要打破黑暗、唤醒这束光时，你本能地更希望用什么动作？"
                "是需要一点力度的干脆按压、毫无阻力的轻轻抚摸，还是它能感知到你的靠近而自动亮起？"
            )
        if turns == 2:
            return (
                "我理解了你的动作直觉。\n\n"
                "这束光在你的桌面上，主要是用来陪伴你做些什么？例如放空、听歌，还是作为一种纯粹的心理安慰？（按你的真实习惯说即可。）"
            )
        return (
            "谢谢，这样第二阶段里「光—动作—场景」的线索就清楚多了。\n"
            "准备好后，你可以进入第三阶段，我们会把文字感受转译成一张纯意象图。"
        )
    if step == 3:
        if turns <= 1:
            return (
                f"我会把你前面谈到的「{u or '那种感受'}」尽量压进色温与材质里，做成一张不含灯具的纯意象。\n\n"
                "[IMAGE_PROMPT: abstract translucent aerogel-like warm glow, soft gradient, "
                "cool metal geometry hints in background, no lamps, no products, cinematic still]\n\n"
                "我把你刚才描述的内心感受转化成了一幅画面。它在多大程度上契合你想要的那个氛围？\n"
                "如果希望更完美，你想先改哪一点：边缘更模糊、颜色更冷，还是材质看起来更柔软？"
            )
        return (
            "收到，我按你的直觉做了一版微调。\n\n"
            "[IMAGE_PROMPT: softer edges, cooler color temperature, matte velvety material, "
            "abstract luminous field, no lamps, no products]\n\n"
            "现在这版更接近你心里那团氛围了吗？若还差一点点，用最口语的一个词形容你想改的感觉即可。"
        )
    if step == 4:
        if turns <= 1:
            return (
                "下面把第三阶段定下的气质，强行收敛成一个「桌面上的发光体」——会偏前卫、超现实，不是常见电商白底商品图。\n\n"
                "[IMAGE_PROMPT: avant-garde surrealist floating desktop object, abstract glowing sculpture, "
                "dramatic rim light, impossible silhouette, not ecommerce catalog]\n\n"
                "如果刚才那个完美的意象，变成了一个真实存在于你桌面上的「发光体」，它可能会是这个样子。"
                "第一眼看到它，你的感受是什么？\n"
                "这个造型里，有哪些部分（轮廓、底座、发光方式）特别对味？又有哪些让你觉得破坏了原有氛围？"
            )
        return (
            "我根据你刚才的取舍，压了一版更贴你描述的形态。\n\n"
            "[IMAGE_PROMPT: surreal desktop luminaire iteration, adjusted proportions, "
            "avant-garde sculptural light, floating object]\n\n"
            "这一版里，最「对味」和最想再削掉的一点分别是什么？"
        )
    if step == 5:
        if turns <= 1:
            return (
                "我根据你刚才说的轮廓与触感偏好，做了一版参数化微调（仍是概念图，不是商品实拍）。\n\n"
                "[IMAGE_PROMPT: parametric desktop light study, refined silhouette, "
                "controlled surface roughness, soft internal glow, minimal studio]\n\n"
                "想象这个物件现在就放在你的宿舍桌面上，这就是你理想中的那个情绪出口吗？"
                "如果是，我们今天的探索就到这里；若还想改，用一个短语说出最想动的一处。"
            )
        return (
            f"我根据你说的「{u}」又推了一版细节。\n\n"
            "[IMAGE_PROMPT: glowing primitive cylinder primitive, matte rough concrete texture, soft internal glow, "
            "minimal studio backdrop, parametric design iteration]\n\n"
            "现在它离你心里的「刚刚好」更近了吗？"
        )
    return "（系统）访谈阶段异常，请刷新页面后重试。"


async def _pollinations_placeholder(prompt: str) -> str:
    safe = re.sub(r"\s+", " ", prompt).strip()[:280]
    from urllib.parse import quote

    q = quote(safe or "abstract light study")
    return f"https://image.pollinations.ai/prompt/{q}?width=640&height=640&nologo=true"


async def call_openai_compatible_image_generation(prompt: str) -> str:
    """生图：Gemini 画图模型走 chat/completions；其余走 images/generations。"""
    if not IMAGE_API_KEY or not IMAGE_API_BASE_URL:
        raise RuntimeError("未配置 IMAGE_API_KEY 与 IMAGE_API_BASE_URL")

    if _image_model_uses_chat_completions():
        return await call_gemini_image_via_chat_completions(prompt)

    url = images_generations_url()
    if not url:
        raise RuntimeError("IMAGE_API_BASE_URL 无效")

    headers = {
        "Authorization": f"Bearer {IMAGE_API_KEY}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "prompt": (prompt or "abstract")[:2000],
        "n": 1,
        "size": "1024x1024",
    }
    if IMAGE_API_MODEL:
        body["model"] = IMAGE_API_MODEL
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(url, headers=headers, json=body)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1200] if exc.response else ""
            raise RuntimeError(f"AIGC HTTP {exc.response.status_code}: {detail}") from exc
        data = r.json()

    items = data.get("data")
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"AIGC 返回无 data 列表: {str(data)[:500]}")

    first = items[0]
    if not isinstance(first, dict):
        raise RuntimeError("AIGC data[0] 格式异常")

    if first.get("url"):
        return str(first["url"]).strip()
    b64 = first.get("b64_json")
    if b64:
        return f"data:image/png;base64,{b64}"

    raise RuntimeError(f"AIGC 未返回 url/b64_json: {str(first)[:300]}")


async def call_text_to_image(prompt: str) -> str:
    if IMAGE_API_KEY and IMAGE_API_BASE_URL:
        return await call_openai_compatible_image_generation(prompt)
    return await _pollinations_placeholder(prompt)


async def run_turn(state: SessionState, user_text: str) -> ChatMessage:
    state.messages.append(ChatMessage(role="user", content=user_text))
    state.user_turns_in_step += 1

    conv: list[dict[str, str]] = []
    for m in state.messages:
        if m.role in ("user", "assistant"):
            conv.append({"role": m.role, "content": m.content})

    system = system_prompt_for_step(state.step)
    raw = ""
    if LLM_API_KEY:
        try:
            raw = await call_llm_chat(conv, system)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"大模型调用失败: {exc}") from exc
    if not raw:
        raw = mock_llm_reply(state.step, user_text, state)

    before, img_prompt, after = extract_image_prompt(raw)
    image_url: Optional[str] = None
    image_intro: Optional[str] = None
    footer = after
    stripped_full = (before + ("\n\n" + after if after else "")).strip()

    if should_emit_image(state.step) and img_prompt:
        try:
            image_url = await call_text_to_image(img_prompt)
        except Exception as exc:  # noqa: BLE001
            footer = (footer + f"\n\n（生图暂时失败：{exc}）").strip()
        state.last_image_prompt = img_prompt
        image_intro = before if before else None
        content = footer if footer else "请结合画面说说你的第一感受。"
    else:
        content = stripped_full if stripped_full else raw.strip()

    assistant_msg = ChatMessage(
        role="assistant",
        content=content,
        image_url=image_url,
        image_intro=image_intro,
        meta={"step": state.step, "image_prompt": img_prompt},
    )
    state.messages.append(assistant_msg)
    return assistant_msg


app = FastAPI(title="AI Interview Prototype", version="0.3.0")

# 跨域：避免前端与 API 不同端口 / 临时静态页访问 API 时出现 Failed to fetch（浏览器 CORS 拦截）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 访问口令中间件（如果配置了 ACCESS_CODE，则校验）
@app.middleware("http")
async def access_code_middleware(request: Request, call_next):
    # 如果未配置访问口令，直接放行
    if not ACCESS_CODE:
        return await call_next(request)

    path = request.url.path

    # 不需要验证的路径（首页允许加载以便显示弹窗）
    excluded_paths = {"/", "/api/set-username", "/api/health", "/api/health/llm", "/api/health/image"}
    if path in excluded_paths:
        return await call_next(request)

    # 检查 cookie 中的访问口令
    client_access_code = (request.cookies.get("access_code") or "").strip()
    if client_access_code == ACCESS_CODE:
        return await call_next(request)

    # 访问口令验证失败，返回 403
    return JSONResponse(
        status_code=403,
        content={"detail": "请输入正确的访问口令"}
    )

templates = Jinja2Templates(directory="templates")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def session_id_from_request(request: Request) -> Optional[str]:
    """优先 Cookie；跨域或第三方页面可回传 X-Session-Id 与后端对齐同一会话。"""
    c = (request.cookies.get("session_id") or "").strip()
    if c:
        return c
    h = request.headers.get("x-session-id") or request.headers.get("X-Session-Id")
    if h and isinstance(h, str):
        return h.strip()
    return None


def cookie_response(sid: str, body: dict[str, Any], access_code: Optional[str] = None) -> JSONResponse:
    """JSON 内始终带 session_id，便于前端在 Cookie 不可写时仍用请求头续聊。"""
    out = {**body, "session_id": sid}
    resp = JSONResponse(out)
    resp.set_cookie(
        key="session_id",
        value=sid,
        httponly=True,
        samesite="lax",
    )
    # 如果提供了访问口令且配置了 ACCESS_CODE，设置访问口令 cookie（session 级别）
    if access_code and ACCESS_CODE and access_code == ACCESS_CODE:
        resp.set_cookie(
            key="access_code",
            value=access_code,
            httponly=True,
            samesite="lax",
        )
    return resp


def _aigc_page_context() -> dict[str, Any]:
    """首页展示用：不含密钥，仅网关与模型等。"""
    route = (
        "chat_completions"
        if _image_model_uses_chat_completions()
        else "images_generations"
    )
    meta = {
        "keyConfigured": bool(IMAGE_API_KEY),
        "baseUrl": IMAGE_API_BASE_URL or "",
        "model": IMAGE_API_MODEL,
        "route": route,
        "endpoints": {
            "chatCompletions": image_gateway_chat_completions_url() or None,
            "imagesGenerations": images_generations_url() or None,
        },
    }
    raw = json.dumps(meta, ensure_ascii=False)
    # 嵌入 <script> 时避免字面量 </script> 截断 HTML 解析
    safe_json = re.sub(r"(?i)</script", r"<\\/script", raw)
    return {
        "aigc_key_configured": meta["keyConfigured"],
        "aigc_base_url": meta["baseUrl"] or None,
        "aigc_model": meta["model"],
        "aigc_route": route,
        "aigc_route_label": (
            "Chat Completions（多模态/生图）"
            if route == "chat_completions"
            else "Images Generations（/v1/images/generations）"
        ),
        "aigc_chat_completions_url": meta["endpoints"]["chatCompletions"],
        "aigc_images_generations_url": meta["endpoints"]["imagesGenerations"],
        "aigc_client_json": safe_json,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, response: Response):
    # 问卷模式：每次访问首页都创建新会话
    old_sid = session_id_from_request(request)
    new_sid = str(uuid.uuid4())
    SESSION_STORE[new_sid] = _new_session_state()
    # 清除旧会话（如果存在）
    if old_sid and old_sid in SESSION_STORE:
        del SESSION_STORE[old_sid]

    # Starlette 约定：TemplateResponse(request, 模板名, context)；勿把 (name, dict) 旧顺序混用，否则会 500
    ctx: dict[str, Any] = {"app_title": "AI 深访实验"}
    ctx.update(_aigc_page_context())
    resp = templates.TemplateResponse(
        request,
        "index.html",
        ctx,
    )
    # 设置会话 cookie（session 级别，浏览器关闭后失效）
    resp.set_cookie(
        key="session_id",
        value=new_sid,
        httponly=True,
        samesite="lax",
    )
    return resp


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    """浏览器或 curl 访问，用于确认本服务已启动及 LLM 环境变量是否已配置。"""
    return {
        "ok": True,
        "service": "ai-interview",
        "llm_api_key_configured": bool(LLM_API_KEY),
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
        "chat_completions_url": chat_completions_url(),
        "aigc_api_key_configured": bool(IMAGE_API_KEY),
        "aigc_api_base_url": IMAGE_API_BASE_URL or None,
        "aigc_image_model": IMAGE_API_MODEL,
        "aigc_route": (
            "chat_completions"
            if _image_model_uses_chat_completions()
            else "images_generations"
        ),
        "images_generations_url": images_generations_url() or None,
        "image_chat_completions_url": image_gateway_chat_completions_url() or None,
    }


@app.get("/api/health/llm")
async def api_health_llm() -> dict[str, Any]:
    """
    向网关发起一次最小 Chat Completions 请求，验证 Key / URL / 模型是否可用。
    浏览器打开：http://127.0.0.1:8000/api/health/llm（端口以你启动 uvicorn 为准）
    """
    if not LLM_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="未配置 OPENAI_API_KEY：请在项目根目录 .env 中设置，或导出环境变量后重启 uvicorn。",
        )
    try:
        reply = await call_llm_chat(
            [{"role": "user", "content": "只回复一个字：好"}],
            "你是助手，只输出用户要求的单个字，不要标点或解释。",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "ok": True,
        "model": LLM_MODEL,
        "endpoint": chat_completions_url(),
        "reply_preview": (reply or "")[:500],
    }


@app.get("/api/health/image")
async def api_health_image() -> dict[str, Any]:
    """
    测试 AIGC：向网关发起一次 OpenAI 兼容的 /v1/images/generations。
    需在 .env 中配置 IMAGE_API_KEY、IMAGE_API_BASE_URL（例如 https://new.suxi.ai）。
    """
    if not IMAGE_API_KEY or not IMAGE_API_BASE_URL:
        raise HTTPException(
            status_code=400,
            detail=(
                "未配置生图环境变量：请设置 IMAGE_API_KEY 与 IMAGE_API_BASE_URL（网关根地址，勿带路径末尾斜杠多余段）。"
            ),
        )
    try:
        img_url = await call_openai_compatible_image_generation(
            "极简测试图：纯白背景上一个红色实心圆，扁平插画，无文字无水印"
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if img_url.startswith("data:"):
        return {
            "ok": True,
            "endpoint": images_generations_url(),
            "model": IMAGE_API_MODEL,
            "format": "base64_data_url",
            "length": len(img_url),
            "hint": "返回为内联 base64，前端可用 img src 直接绑定该字符串（较长，此处不展开）。",
        }
    return {
        "ok": True,
        "endpoint": images_generations_url(),
        "model": IMAGE_API_MODEL,
        "image_url": img_url,
    }


@app.get("/api/state")
async def api_state(request: Request):
    sid, state = get_or_create_session(session_id_from_request(request))
    return cookie_response(sid, {**state.to_public_dict()})


@app.post("/api/chat")
async def api_chat(request: Request, payload: dict[str, Any]):
    if payload.get("step") is None:
        raise HTTPException(status_code=400, detail="缺少必填字段 step（1-5）")

    text = (payload.get("message") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message 不能为空")

    sid, state = get_or_create_session(session_id_from_request(request))
    if state.finished:
        raise HTTPException(status_code=400, detail="访谈已结束，无法继续发送")

    request_step = clamp_step(payload["step"])
    state.step = request_step

    await run_turn(state, text)
    return cookie_response(sid, {**state.to_public_dict()})


@app.post("/api/advance-stage")
async def api_advance_stage(request: Request, payload: dict[str, Any]):
    """
    同步前端进入的新阶段：body 需包含 step（1-5）。
    当 step 大于会话原 step 时，插入居中的 system 分割线文案。
    """
    if payload.get("step") is None:
        raise HTTPException(status_code=400, detail="缺少必填字段 step（1-5）")

    new_step = clamp_step(payload["step"])
    sid, state = get_or_create_session(session_id_from_request(request))
    if state.finished:
        raise HTTPException(status_code=400, detail="访谈已结束")

    prev = state.step
    if new_step > prev:
        for s in range(prev + 1, new_step + 1):
            line = STAGE_DIVIDER_TEXT.get(s)
            if line:
                state.messages.append(
                    ChatMessage(
                        role="system",
                        content=line,
                        meta={"stage_marker": True, "step": s},
                    )
                )
        state.step = new_step
        state.user_turns_in_step = 0
        opening = STAGE_OPENING_ASSISTANT.get(new_step)
        if opening:
            state.messages.append(
                ChatMessage(
                    role="assistant",
                    content=opening,
                    meta={"step": new_step, "stage_opening": True},
                )
            )
    elif new_step < prev:
        state.step = new_step
        state.user_turns_in_step = 0
    else:
        state.step = new_step

    return cookie_response(sid, {**state.to_public_dict()})


@app.post("/api/finish-interview")
async def api_finish_interview(request: Request):
    sid, state = get_or_create_session(session_id_from_request(request))
    if state.finished:
        return cookie_response(sid, {**state.to_public_dict()})

    state.finished = True
    state.messages.append(
        ChatMessage(
            role="system",
            content="--- 访谈已结束，感谢你的参与与时间 ---",
            meta={"stage_marker": True, "finished": True},
        )
    )
    return cookie_response(sid, {**state.to_public_dict()})


@app.post("/api/set-username")
async def api_set_username(request: Request, payload: dict[str, Any]):
    """设置用户名（可选访问口令验证）"""
    username = (payload.get("username") or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="用户名不能为空")

    # 验证访问口令（如果配置了）
    verified_access_code = None
    if ACCESS_CODE:
        access_code = (payload.get("access_code") or "").strip()
        if access_code != ACCESS_CODE:
            raise HTTPException(status_code=403, detail="访问口令错误，请检查后重试")
        verified_access_code = access_code

    sid, state = get_or_create_session(session_id_from_request(request))
    state.username = username

    return cookie_response(sid, {**state.to_public_dict()}, access_code=verified_access_code)


@app.post("/api/save-chat")
async def api_save_chat(request: Request):
    """保存当前聊天数据到文件"""
    sid, state = get_or_create_session(session_id_from_request(request))

    if not state.username:
        raise HTTPException(status_code=400, detail="请先设置用户名")

    try:
        filepath = save_chat_to_file(state.username, sid, state)
        return cookie_response(sid, {
            "ok": True,
            "filepath": filepath,
            "username": state.username,
            "message_count": len(state.messages),
            "step": state.step,
            "finished": state.finished,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存失败: {exc}") from exc


@app.post("/api/reset")
async def api_reset(request: Request, response: Response):
    sid = session_id_from_request(request)
    new_sid = str(uuid.uuid4())
    SESSION_STORE[new_sid] = _new_session_state()
    if sid and sid in SESSION_STORE:
        del SESSION_STORE[sid]
    return cookie_response(new_sid, {"ok": True, **SESSION_STORE[new_sid].to_public_dict()})
