from typing import List, Dict

class EchoBackend:
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        # 最後のユーザー発話を短くオウム返し（最低限のデモ用）
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
        content = last_user.get("content", "")
        return f"うん、つまり『{content[:80]}』ってことだよね。"

class RuleBackend:
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        # 超簡易なルール応答（本番は差し替え前提）
        txt = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if any(k in txt for k in ["虫", "セミ", "昆虫"]):
            return "虫はちょっと苦手だけど、君が気になるなら一緒に調べよっか。"
        if any(k in txt for k in ["ラーメン", "麺", "昼ごはん"]):
            return "ラーメン行こっか。こってりとあっさり、どっち気分？"
        return "うん、話の続き聞かせて？"

class CustomBackend:
    def __init__(self, fn=None):
        """fn(messages)->str を渡すと任意のLLM接続に差し替え可能。"""
        self.fn = fn

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        if self.fn is None:
            # 未設定ならエコーで代替
            return EchoBackend()(messages)
        return self.fn(messages)

def generate_response(messages: List[Dict[str, str]], backend: str = "echo", custom_fn=None) -> str:
    if backend == "echo":
        return EchoBackend()(messages)
    if backend == "rule":
        return RuleBackend()(messages)
    if backend == "custom":
        return CustomBackend(custom_fn)(messages)
    return EchoBackend()(messages)
