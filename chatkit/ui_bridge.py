import json
from typing import List, Dict, Any, Tuple
from .messages import build_messages
from .generator import generate_response

def some_function_name(state: List[Dict[str, str]] | None, user_text: str, bot: str, user_data: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str, str]:
    """UIから呼ばれる統一関数。
    - state: これまでの履歴（[{role, content}, ...]）
    - user_text: 直近ユーザー入力
    - bot: 直近アシスタント応答（UI側が保持している場合）
    - user_data: プロフィール辞書
    戻り値: (history, bot, profile_json_str)
    """
    history: List[Dict[str, str]] = []
    if state:
        for m in state:
            if isinstance(m, dict) and "role" in m and "content" in m:
                history.append(m)

    # メッセージを構築し、必要なら応答を生成
    messages = build_messages(user_text, user_data, history)
    if not bot:
        bot = generate_response(messages, backend="rule")

    # 履歴更新（修正版）
    history += [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": bot},
    ]

    profile = json.dumps(user_data, ensure_ascii=False, indent=2)
    return history, bot, profile
