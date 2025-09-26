# ingest_csv.py
# ユーザー↔AIの会話CSVを読み込み、意図→フレーズ辞書(intents.json)へ取り込む

import os, json, re, time
from pathlib import Path
import pandas as pd

# === パス設定（Windowsパスは raw 文字列で） ===
CSV_PATH = Path(r"C:\Users\unitx\AIproject\WebUI\u1_1000 1(Sheet1).csv")
INTENTS_JSON = "intents.json"
PHRASE_CAP = 400  # 各ラベルの最大保持数（古いものから間引き）

# === intent 別名正規化 ===
INTENT_ALIAS = {
    "POSITIVE_AFFECTION": "affection",
    "AFFECTION_POS": "affection",
    "LIKE_POSITIVE": "affection",
    "DATE_PLAN": "date",
    "FOOD_TALK": "food",
    "SMALL_TALK": "neutral",
}

def normalize_intent(label: str) -> str:
    if not label:
        return "neutral"
    return INTENT_ALIAS.get(label, label).lower()

# === ヒントの種語彙（SEED） ===
SEED_HINTS = {
    "greeting":  ["おはよう","やっほー","こんちゃ","会いたかった"],
    "affection": ["大好きだよ","ぎゅってしたい","独り占めしたい","一緒に過ごしたい","キミが一番"],
    "worry":     ["大丈夫？","無理しないで","そばにいるよ","休もう","抱きしめたい"],
    "jealousy":  ["ちょっとやきもち","私だけ見てて","他の子と話してた？","独占したい"],
    "date":      ["一緒に行こう","今夜どう？","次いつ会える？","手つなご？","映画どう？","どの席が好き？"],
    "food":      ["一緒に食べよう","何が食べたい？","お腹すいたね","夜食つくる？"],
    "banter":    ["冗談でしょ？","もう〜","かわいいこと言うね","もーっ"],
    "neutral":   ["今日はどう？","どうした？","今はどうしたい？","それ良さそう","おすすめ3つ出そうか？"],
}

# === 簡易・意図分類（CSVの取り込み時用 / コスト0） ===
def fallback_intent(t: str) -> str:
    tl = (t or "").lower()
    if any(k in tl for k in ["すき","好き","愛してる","らぶ","chu","ぎゅ","ハグ"]):
        return "affection"
    if any(k in t for k in ["デート","遊び","会お","行かない","行こう","出かけ"]):
        return "date"
    if any(k in t for k in ["ラーメン","ご飯","オムライス","食べ","夜食"]):
        return "food"
    if any(k in t for k in ["ばか","ばーか","あほ","死ね","嫌い"]):
        return "banter"
    if any(k in t for k in ["何する","なにする","何したい","なにしたい","今から"]):
        return "neutral"
    if any(k in t for k in ["怖い","つらい","疲れた","痛い","不安"]):
        return "worry"
    return "neutral"

# === intents.json 読み書き ===
def load_intents():
    if os.path.exists(INTENTS_JSON):
        with open(INTENTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"by_intent": {}, "stats": {}}

def save_intents(db):
    with open(INTENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def merge_seed(db):
    by = db.setdefault("by_intent", {})
    for k, arr in SEED_HINTS.items():
        cur = by.setdefault(k, [])
        merged = list(dict.fromkeys([*cur, *arr]))         # 重複除去・順序維持
        by[k] = merged[-PHRASE_CAP:]
        st = db.setdefault("stats", {}).setdefault(k, {"n":0,"last_ts":0})
        st["n"] = len(by[k]); st["last_ts"] = int(time.time())

# === フレーズ追加（軽いノイズ対策含む） ===
def add_phrase(db, intent, phrase):
    if not phrase:
        return
    phrase = re.sub(r"\s+", " ", str(phrase).strip())
    if len(phrase) < 2:
        return
    NG = ["空巣","饿満","ラーヘイズ"]  # 既知ノイズ
    if any(bad in phrase for bad in NG):
        return
    by = db.setdefault("by_intent", {})
    arr = by.setdefault(intent, [])
    if phrase not in arr:
        arr.append(phrase)
        if len(arr) > PHRASE_CAP:
            del arr[:-PHRASE_CAP]
    st = db.setdefault("stats", {}).setdefault(intent, {"n":0,"last_ts":0})
    st["n"] = len(arr); st["last_ts"] = int(time.time())

def main():
    # === CSV 読込（エンコーディング自動フォールバック） ===
    encs = ["utf-8", "utf-8-sig", "cp932"]
    last_err = None
    df = None
    for enc in encs:
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc)
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise last_err

    # すべて文字列化＆前後空白トリム
    df = df.applymap(lambda x: str(x).strip() if pd.notnull(x) else "")
    # 完全空行は除去
    df = df[df.apply(lambda r: any(bool(str(v)) for v in r), axis=1)]

    cols = list(df.columns)

    # === u* の直後列を返答列とみなす（a* でなくてもOK） ===
    pairs = []
    for idx, col in enumerate(cols):
        if col.startswith("u"):
            uc = col
            ac = None
            if idx + 1 < len(cols):
                nxt = cols[idx + 1]
                # 直後が a* ならそれ。a* でなくても「u 以外」なら返答候補とする
                if nxt.startswith("a") or not nxt.startswith("u"):
                    ac = nxt
            if ac:
                pairs.append((uc, ac))
    if not pairs:
        raise RuntimeError("u*/(a* または 非u列) のペアが見つかりません。列名を確認してください。")

    # === intents.json に SEED マージ + 追記 ===
    db = load_intents()
    merge_seed(db)

    added = 0
    for _, row in df.iterrows():
        for uc, ac in pairs:
            u = row.get(uc, "")
            a = row.get(ac, "")
            if not u or not a:
                continue
            intent = normalize_intent(fallback_intent(u))
            add_phrase(db, intent, a)
            added += 1

    save_intents(db)
    print(f"done: added {added} phrases to {INTENTS_JSON}")
    # 確認用サマリ
    by = db.get("by_intent", {})
    summary = {k: len(v) for k, v in by.items()}
    print("summary:", summary)

if __name__ == "__main__":
    main()
