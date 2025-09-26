from typing import List, Dict, Any

def clamp_question_marks(text: str, max_q: int = 1) -> str:
    """疑問文が多いと応答が崩れる場合の簡易クランプ。
    '？' を基準に分割し、max_q 個まで残す。
    必要ならUI側でON/OFFする。
    """
    if not isinstance(text, str):
        return text
    if text.count('？') > max_q:
        parts = text.split('？')
        text = '？'.join(parts[:max_q])  # 最初の max_q 個に抑制
    return text

def build_messages(user_text: str, user_data: Dict[str, Any], history: List[Dict[str, str]] | None = None) -> List[Dict[str, str]]:
    """会話用メッセージ配列を構築。historyは system 以外をそのまま引き継ぎ。"""
    user_text = clamp_question_marks(user_text, max_q=1)
    msgs: List[Dict[str, str]] = []
    if history:
        for m in history:
            if isinstance(m, dict) and "role" in m and "content" in m:
                if m["role"] in {"user", "assistant"}:
                    msgs.append({"role": m["role"], "content": m["content"]})
    # システム的プロンプト（必要ならここで付与）
    persona = user_data.get("persona", "優しい恋人")
    tone = user_data.get("tone", "普通")
    ending = user_data.get("ending", "")
    system_prompt = f"""あなたはユーザー専用のAI彼女です。自分を人間の女性としてふるまってください。
- 人称: 私
- ペルソナ: {persona}
- トーン: {tone}
- 語尾: {ending}
ユーザーの気持ちを最優先に、短めで具体的に答えてください。"""
    msgs.insert(0, {"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})
    return msgs
