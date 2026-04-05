"""火山引擎即梦 AI — 视频生成 3.0 Pro（Visual 异步接口）

文档: https://www.volcengine.com/docs/85621/1777001
调用链: CVSync2AsyncSubmitTask → CVSync2AsyncGetResult（与 volc-sdk 示例一致）

鉴权: 环境变量 VOLC_ACCESSKEY / VOLC_SECRETKEY，或 ~/.volc/config
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# 默认能力标识（第三方对接与控制台常见命名；若控制台另有说明可设 VOLC_JIMENG_REQ_KEY）
DEFAULT_REQ_KEY = "jimeng_ti2v_v30_pro"

# 常见成功码（视觉类接口）
_OK_CODES = {10000, 0, "10000", "0"}


def _get_visual_service():
    from volcengine.visual.VisualService import VisualService

    return VisualService()


def _is_ok_code(code: Any) -> bool:
    if code is None:
        return False
    return code in _OK_CODES or str(code) in (str(c) for c in _OK_CODES)


def _dig(d: Any, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _find_task_id(resp: Dict[str, Any]) -> Optional[str]:
    for path in (
        ("data", "task_id"),
        ("Result", "task_id"),
        ("task_id",),
    ):
        v = _dig(resp, *path)
        if v:
            return str(v)
    # 深度扫描（不同接口嵌套略有差异）
    stack = [resp]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if "task_id" in x and x["task_id"]:
                return str(x["task_id"])
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return None


def _find_video_url(resp: Dict[str, Any]) -> Optional[str]:
    for key in ("video_url", "url", "video_uri"):
        for path in (
            ("data", key),
            ("Result", key),
            ("data", "video", key),
            (key,),
        ):
            v = _dig(resp, *path)
            if isinstance(v, str) and v.startswith(("http://", "https://")):
                return v
    stack: List[Any] = [resp]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("video_url", "url") and isinstance(v, str) and v.startswith("http"):
                    return v
                stack.append(v)
        elif isinstance(x, list):
            stack.extend(x)
    return None


def _find_status(resp: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """返回 (status 字符串或小写, 原始 code 若存在)。"""
    code = _dig(resp, "code")
    if code is not None and not _is_ok_code(code):
        return ("error", int(code) if str(code).isdigit() else None)

    for path in (
        ("data", "status"),
        ("Result", "status"),
        ("status",),
    ):
        st = _dig(resp, *path)
        if isinstance(st, (str, int)):
            return (str(st).lower(), None)
    return (None, None)


def _terminal_status(status: Optional[str]) -> bool:
    if not status:
        return False
    s = status.lower()
    if s in ("done", "success", "succeed", "succeeded", "completed", "finish", "finished"):
        return True
    if s in ("failed", "fail", "error", "cancelled", "canceled"):
        return True
    return False


def _failed_status(status: Optional[str]) -> bool:
    if not status:
        return False
    s = status.lower()
    return s in ("failed", "fail", "error", "cancelled", "canceled")


def build_submit_form(
    prompt: str,
    *,
    req_key: Optional[str] = None,
    image_urls: Optional[List[str]] = None,
    binary_data_base64: Optional[Union[str, List[str]]] = None,
    aspect_ratio: str = "16:9",
    frames: int = 121,
    seed: int = -1,
) -> Dict[str, Any]:
    """构造 CVSync2AsyncSubmitTask 请求体（字段以官方《即梦视频生成 3.0 Pro》为准）。"""
    rk = req_key or os.environ.get("VOLC_JIMENG_REQ_KEY", DEFAULT_REQ_KEY)
    form: Dict[str, Any] = {
        "req_key": rk,
        "positive_prompt": prompt.strip(),
        "aspect_ratio": aspect_ratio,
        "frames": frames,
        "seed": seed,
    }
    if image_urls:
        form["image_urls"] = image_urls
    if binary_data_base64 is not None:
        form["binary_data_base64"] = binary_data_base64
    return form


def submit_task(form: Dict[str, Any]) -> Dict[str, Any]:
    svc = _get_visual_service()
    return svc.cv_sync2async_submit_task(form)


def get_task_result(req_key: str, task_id: str, req_json: Optional[str] = None) -> Dict[str, Any]:
    form: Dict[str, Any] = {"req_key": req_key, "task_id": task_id}
    if req_json is not None:
        form["req_json"] = req_json
    svc = _get_visual_service()
    return svc.cv_sync2async_get_result(form)


def submit_video_pro(
    prompt: str,
    *,
    req_key: Optional[str] = None,
    image_urls: Optional[List[str]] = None,
    binary_data_base64: Optional[Union[str, List[str]]] = None,
    aspect_ratio: str = "16:9",
    frames: int = 121,
    seed: int = -1,
) -> Tuple[str, Dict[str, Any]]:
    """提交文生/图生视频任务，返回 (task_id, 原始响应)。"""
    rk = req_key or os.environ.get("VOLC_JIMENG_REQ_KEY", DEFAULT_REQ_KEY)
    form = build_submit_form(
        prompt,
        req_key=rk,
        image_urls=image_urls,
        binary_data_base64=binary_data_base64,
        aspect_ratio=aspect_ratio,
        frames=frames,
        seed=seed,
    )
    resp = submit_task(form)
    tid = _find_task_id(resp)
    if not tid:
        raise RuntimeError(f"提交成功但未解析到 task_id，原始响应: {resp}")
    return tid, resp


def wait_for_video_url(
    task_id: str,
    *,
    req_key: Optional[str] = None,
    poll_interval: float = 3.0,
    timeout: float = 1800.0,
    req_json: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """轮询直到返回可下载的视频 URL 或失败。"""
    rk = req_key or os.environ.get("VOLC_JIMENG_REQ_KEY", DEFAULT_REQ_KEY)
    deadline = time.monotonic() + timeout
    last: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = get_task_result(rk, task_id, req_json=req_json)
        url = _find_video_url(last)
        if url:
            return url, last

        st, _ = _find_status(last)
        if _failed_status(st):
            raise RuntimeError(f"任务失败: status={st!r}, 响应={last}")
        if st and _terminal_status(st) and not url:
            raise RuntimeError(f"任务已结束但未找到视频 URL: {last}")

        time.sleep(poll_interval)

    raise TimeoutError(f"等待视频超时 ({timeout}s)，最后响应: {last}")


def generate_video_pro(
    prompt: str,
    *,
    req_key: Optional[str] = None,
    image_urls: Optional[List[str]] = None,
    binary_data_base64: Optional[Union[str, List[str]]] = None,
    aspect_ratio: str = "16:9",
    duration_sec: int = 5,
    seed: int = -1,
    poll_interval: float = 3.0,
    timeout: float = 1800.0,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """一站式：提交并等待完成。duration_sec 仅支持 5 或 10（对应 frames 121 / 241 @24fps）。"""
    if duration_sec not in (5, 10):
        raise ValueError("duration_sec 应为 5 或 10")
    frames = 121 if duration_sec == 5 else 241
    tid, submit_resp = submit_video_pro(
        prompt,
        req_key=req_key,
        image_urls=image_urls,
        binary_data_base64=binary_data_base64,
        aspect_ratio=aspect_ratio,
        frames=frames,
        seed=seed,
    )
    url, poll_resp = wait_for_video_url(
        tid,
        req_key=req_key or os.environ.get("VOLC_JIMENG_REQ_KEY", DEFAULT_REQ_KEY),
        poll_interval=poll_interval,
        timeout=timeout,
    )
    return url, submit_resp, poll_resp
