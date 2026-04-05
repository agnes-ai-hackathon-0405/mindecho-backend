"""Volcengine IAM via Python SDK (set_ak / set_sk on IamService).

Per https://github.com/volcengine/volc-sdk-python — credentials can be set on the client
instead of (or in addition to) VOLC_ACCESSKEY / VOLC_SECRETKEY env vars.
"""

from __future__ import annotations

from typing import Any

from volcengine.iam.IamService import IamService


def iam_service(ak: str, sk: str) -> IamService:
    svc = IamService()
    svc.set_ak(ak)
    svc.set_sk(sk)
    return svc


def iam_probe(ak: str, sk: str) -> dict[str, Any]:
    """
    Run a few read-only IAM calls. Success on any call means the AK/SK is valid for IAM.
    (Visual / 即梦 APIs may still return Access Denied if IAM policies omit those products.)
    """
    svc = iam_service(ak, sk)
    calls: list[dict[str, Any]] = []
    for action, fn in (
        ("ListAccessKeys", lambda: svc.list_access_keys({})),
        ("GetUser", lambda: svc.get_user({})),
    ):
        try:
            data = fn()
            calls.append({"action": action, "success": True, "data": data})
        except Exception as e:
            calls.append({"action": action, "success": False, "error": str(e)})
    ok = any(c.get("success") for c in calls)
    return {
        "ok": ok,
        "sdk": "volcengine.IamService — set_ak / set_sk (see volc-sdk-python)",
        "calls": calls,
    }
