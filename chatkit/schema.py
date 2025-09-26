from typing import Dict, Any, List

def default_user_data() -> Dict[str, Any]:
    """移行先で初期化に使う user_data のデフォルト値。"""
    return {
        "call": "あなた",
        "persona": "優しい恋人",
        "likes": [],
        "tone": "普通",
        "ending": "",
        "style": "",
        "situation": "",
        # 必要に応じて追加
    }

def validate_user_data(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """最低限の検証＆不足キーの補完。pydanticを入れる場合は差し替え可。"""
    base = default_user_data()
    output = dict(base)
    if isinstance(user_data, dict):
        output.update({k: v for k, v in user_data.items() if k in base})
    return output
