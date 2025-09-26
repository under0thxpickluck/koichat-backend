import os
import json
from pathlib import Path

# 全ユーザーのデータを保存するベースディレクトリ
DATA_ROOT_STR = os.getenv("DATA_ROOT", str(Path(__file__).parent / "data"))
DATA_ROOT = Path(DATA_ROOT_STR)

def get_user_list():
    """データディレクトリ内に存在するユーザー名のリストを取得する"""
    if not DATA_ROOT.exists():
        return []
    return sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])

def get_user_dir(username: str) -> Path:
    """指定されたユーザーのデータディレクトリのパスを返す"""
    return DATA_ROOT / username

def create_new_user(username: str):
    """新しいユーザーのデータディレクトリと初期設定ファイルを作成する"""
    if not username or " " in username or Path(username).is_absolute():
        raise ValueError("無効なユーザー名です。")

    user_dir = get_user_dir(username)
    if user_dir.exists():
        raise ValueError(f"ユーザー '{username}' は既に存在します。")

    # ユーザー用のディレクトリとfeedbackサブディレクトリを作成
    (user_dir / "feedback").mkdir(parents=True, exist_ok=True)

    # 初期設定 (user_memory.json) を作成
    initial_memory = {
        username: {
            "call": "キミ",
            "persona": "世話焼きで甘えん坊な彼女",
            "tone": "くだけたやさしい話し方",
            "ending": "だよ♡",
            "likes": [],
            "dislikes": [],
            "meters": {
                "like": 55, "intimacy": 50, "amae": 45,
                "tereru": 40, "yasuragi": 50, "jealousy": 20
            },
            # ここから追加（ポイント＆ギフト＆アルバムの既定値）
            "points": 100,
            "gift_usage": {},
            "album": []
        }
    }
    save_user_memory(username, initial_memory)

    # 空のintents.jsonも作成
    save_json(username, "intents.json", {"by_intent": {}, "stats": {}})

    print(f"ユーザー '{username}' を作成しました。")
    return True

def load_json(username: str, filename: str):
    """ユーザーの指定したJSONファイルを読み込む"""
    path = get_user_dir(username) / filename
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(username: str, filename: str, data: dict):
    """ユーザーの指定したJSONファイルに保存する"""
    user_dir = get_user_dir(username)
    user_dir.mkdir(exist_ok=True)
    with (user_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_user_memory(username: str):
    """ユーザーの記憶ファイルを読み込む"""
    data = load_json(username, "user_memory.json")
    if data is None:
        # フォールバックとして空のデータ構造を返す
        return {username: {}}
    return data

def save_user_memory(username: str, data: dict):
    """ユーザーの記憶ファイルを保存する"""
    save_json(username, "user_memory.json", data)

def append_jsonl(username: str, file_path_in_feedback: str, obj: dict | str):
    """ユーザーのfeedbackディレクトリ内のjsonlファイルに追記する"""
    feedback_dir = get_user_dir(username) / "feedback"
    feedback_dir.mkdir(exist_ok=True)
    path = feedback_dir / file_path_in_feedback
    with path.open("a", encoding="utf-8") as f:
        if isinstance(obj, str):
            f.write(obj.strip() + "\n")
        else:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_text_file_lines(username: str, file_path_in_feedback: str) -> list[str]:
    """ユーザーのfeedbackディレクトリ内のテキストファイルを読み、行のリストを返す"""
    path = get_user_dir(username) / "feedback" / file_path_in_feedback
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()

def write_text_file_lines(username: str, file_path_in_feedback: str, lines: list[str]):
    """ユーザーのfeedbackディレクトリ内のテキストファイルに、行のリストを書き込む"""
    path = get_user_dir(username) / "feedback" / file_path_in_feedback
    path.write_text("\n".join(lines), encoding="utf-8")
# user_manager.py に追加する関数のイメージ
def consume_points(username, amount):
    mem = load_user_memory(username)
    user_data = mem.get(username, {})
    
    current_points = user_data.get("points", 0)
    if current_points < amount:
        return False # ポイント不足
    
    user_data["points"] = current_points - amount
    save_user_memory(username, mem)
    return True # 消費成功