# backend_gifts.py — プレゼント系APIだけを集約（他機能は触らない）
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from random import choice

import user_manager  # 既存の保存/読込ロジックを利用（互換性維持）

router = APIRouter(prefix="/backend")


# ============ 共通ユーティリティ ============
def _today_iso() -> str:
    """ローカル日付のYYYY-MM-DD（使用上限のリセット判定用）"""
    return datetime.now().date().isoformat()


def _load_user_data(username: str):
    """ユーザーの永続データを取得（デフォルト値を補完）"""
    mem = user_manager.load_user_memory(username)
    data = (mem or {}).get(username, {}) if isinstance(mem, dict) else {}
    # デフォルト値
    data.setdefault("points", 100)
    data.setdefault(
        "meters",
        {"like": 50, "amae": 50, "tereru": 40, "yasuragi": 50, "jealousy": 20},
    )
    data.setdefault("gift_usage", {})  # {gift_id: {"count": int, "date": "YYYY-MM-DD"}}
    return mem or {username: {}}, data


def _save_user_data(username: str, mem: dict, data: dict) -> None:
    mem[username] = data
    user_manager.save_user_memory(username, mem)


def _apply_meters_delta(data: dict, delta: dict | None) -> None:
    """メーター加算＆0..100クランプ。親密度はポジティブ平均。"""
    meters = data.setdefault("meters", {})
    for k, v in (delta or {}).items():
        base = int(meters.get(k, 50))
        meters[k] = max(0, min(100, base + int(v)))

    like = meters.get("like", 50)
    amae = meters.get("amae", 50)
    tereru = meters.get("tereru", 40)
    yasuragi = meters.get("yasuragi", 50)
    meters["intimacy"] = int(round((like + amae + tereru + yasuragi) / 4))


def _remaining(per_day: int, used_count: int) -> int | None:
    """残回数を計算。上限なし(<=0)なら None。"""
    if per_day <= 0:
        return None
    return max(0, per_day - used_count)


# ============ ギフト定義（ここを増やせば拡張） ============
GIFTS_CONFIG = {
    "bouquet": {  # 花束
        "label": "花束",
        "cost": 120,
        "stamp": "/presents/present1.png",  # Next の public/presents に配置
        "replies": [
            "わぁ…お花？ すっごく綺麗。ありがとう！",
            "いい香り…！こういうの、実は嬉しいんだよ？",
            "照れるけど…大切に飾っておくね。",
            "今日はちょっと特別な日みたい。ありがと！",
            "たまに、こういうサプライズ…ずるいなぁ。",
        ],
        # メーター増減：like +10, yasuragi -3（要件）
        "meters": {"like": +10, "yasuragi": -3},
        # 使用制限：1日3回
        "limit": {"per_day": 3},
    },
    # 例）追加したい場合は同じ形式で増やすだけ
    # "chocolate": {...},
}


# ============ API: ギフト一覧（残回数つき） ============
@router.get("/gifts")
def list_gifts(username: str | None = None):
    # 定義を返却用に整形
    gifts = [
        {
            "id": gid,
            "label": cfg.get("label", gid),
            "cost": int(cfg.get("cost", 0)),
            "stamp": cfg.get("stamp"),
            "limit": cfg.get("limit", {}),
        }
        for gid, cfg in GIFTS_CONFIG.items()
    ]

    # 残回数（読み取り専用。カウントは増やさない）
    remaining = {}
    if username:
        _, data = _load_user_data(username)
        today = _today_iso()
        usage_all = data.get("gift_usage", {})
        for gid, cfg in GIFTS_CONFIG.items():
            per_day = int(cfg.get("limit", {}).get("per_day", 0) or 0)
            usage = usage_all.get(gid, {"count": 0, "date": today})
            count_today = usage["count"] if usage.get("date") == today else 0
            remaining[gid] = _remaining(per_day, count_today)

    return {"gifts": gifts, "remaining": remaining}


# ============ API: ポイントの更新（GET/POST両対応） ============
def _update_points_impl(username: str, delta: int | None, new_points: int | None):
    if not username or (delta is None and new_points is None):
        return JSONResponse(status_code=400, content={"error": "BAD_REQUEST"})

    mem, data = _load_user_data(username)
    cur = int(data.get("points", 100))
    nxt = int(new_points) if new_points is not None else max(0, cur + int(delta))
    data["points"] = nxt
    _save_user_data(username, mem, data)
    return {"username": username, "points": nxt}


@router.api_route("/update_points", methods=["GET", "POST"])
async def update_points(
    request: Request,
    username: str | None = None,
    delta: int | None = None,
    new_points: int | None = None,
):
    # POST(JSON or form) はここで取り出す
    if request.method == "POST" and (username is None and delta is None and new_points is None):
        ct = (request.headers.get("content-type") or "").lower()
        payload = await (request.json() if "application/json" in ct else request.form())
        username = payload.get("username") or username
        if "delta" in payload:
            delta = int(payload.get("delta")) if payload.get("delta") not in (None, "") else None
        if "new_points" in payload:
            new_points = int(payload.get("new_points")) if payload.get("new_points") not in (None, "") else None

    return _update_points_impl(username, delta, new_points)


# ============ API: ギフトを贈る（残回数/ポイント/メーター更新） ============
def _give_gift_impl(username: str, gift_id: str):
    if not username or not gift_id:
        return JSONResponse(status_code=400, content={"error": "BAD_REQUEST"})

    cfg = GIFTS_CONFIG.get(gift_id)
    if not cfg:
        return JSONResponse(status_code=404, content={"error": "GIFT_NOT_FOUND"})

    per_day = int(cfg.get("limit", {}).get("per_day", 0) or 0)

    mem, data = _load_user_data(username)
    today = _today_iso()
    usage_all = data.get("gift_usage", {})
    usage = usage_all.get(gift_id, {"count": 0, "date": today})

    # 日付が変わっていればリセット
    if usage.get("date") != today:
        usage = {"count": 0, "date": today}

    # 1) 使用上限チェック（まだカウントは増やさない）
    if per_day > 0 and usage["count"] >= per_day:
        return JSONResponse(status_code=400, content={"error": "USAGE_LIMIT", "remaining": 0})

    # 2) ポイントチェック
    cost = int(cfg.get("cost", 0))
    points = int(data.get("points", 100))
    if points < cost:
        need = cost - points
        return JSONResponse(
            status_code=400,
            content={"error": "INSUFFICIENT_POINTS", "need": need, "points": points},
        )

    # 3) ここで初めて使用回数を +1（成功時のみ消費する）
    usage["count"] += 1
    usage["date"] = today
    usage_all[gift_id] = usage
    data["gift_usage"] = usage_all

    # 4) ポイント減算 & メーター更新
    data["points"] = points - cost
    _apply_meters_delta(data, cfg.get("meters"))

    # 5) 保存
    _save_user_data(username, mem, data)

    # 6) レスポンス
    reply = choice(cfg.get("replies", ["ありがとう！"]))
    remaining = _remaining(per_day, usage["count"])
    return {
        "username": username,
        "gift_id": gift_id,
        "label": cfg.get("label", gift_id),
        "stamp": cfg.get("stamp"),
        "reply": reply,
        "points": data["points"],
        "meters": data["meters"],
        "remaining": remaining,  # ← 今回消費後の“残り”
    }


@router.api_route("/give_gift", methods=["GET", "POST"])
async def give_gift(request: Request, username: str | None = None, gift_id: str | None = None):
    # POST(JSON or form) はここで取り出す
    if request.method == "POST" and (username is None or gift_id is None):
        ct = (request.headers.get("content-type") or "").lower()
        payload = await (request.json() if "application/json" in ct else request.form())
        username = username or payload.get("username")
        gift_id = gift_id or payload.get("gift_id")

    return _give_gift_impl(username, gift_id)
