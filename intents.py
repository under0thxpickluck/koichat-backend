# intents.py
from rapidfuzz import fuzz, process
import json, os, re
import openai

INTENT_SEEDS = {
    "POSITIVE_AFFECTION": ["好き", "会いたい", "ぎゅ", "ハグ", "キス", "大好き", "抱きしめて"],
    "INVITE": ["会う", "会お", "デート", "行こ", "行こう", "電話", "通話", "遊ぼ"],
    "JEALOUSY_TRIGGER": ["元カノ", "元カレ", "同僚", "先輩", "後輩", "他の子", "浮気", "二人で飲み", "連絡先"],
    "INSULT": ["黙れ", "死ね", "バカ", "ブス", "きも", "最低", "うざい"],
    "SADNESS": ["不安", "寂し", "さみし", "つらい", "疲れた", "しんどい", "泣きそう"],
    "NEUTRAL": ["おはよ", "なにしてる", "どうしてる", "元気"]
}
FUZZ_THRES = 85

CACHE_PATH = "intent_cache.json"
def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(d):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _lexicon():
    d = _load_cache()
    lex = {k:set(v) for k,v in d.items()}
    for k, seeds in INTENT_SEEDS.items():
        lex.setdefault(k, set()).update(seeds)
    return {k:list(v) for k,v in lex.items()}

def _fuzzy_hit(text: str, vocab: list[str], th=FUZZ_THRES) -> bool:
    if not vocab: return False
    m = process.extractOne(text, vocab, scorer=fuzz.WRatio)
    return bool(m and m[1] >= th)

def detect_intents(user_text: str) -> list[str]:
    text = (user_text or "").strip()
    lex = _lexicon()
    hits = []
    # まず fuzzy で拾う（優先度順だが"追加"していく）
    priority = ["JEALOUSY_TRIGGER", "INSULT", "POSITIVE_AFFECTION", "INVITE", "SADNESS", "NEUTRAL"]
    for intent in priority:
        if _fuzzy_hit(text, lex.get(intent, [])):
            hits.append(intent)

    # 補強（正規表現の共起）：女の子/彼女 + かわいい/好き で嫉妬も点火
    if re.search(r"(女の子|彼女|あの子|同僚|先輩|後輩).*(かわいい|好き)", text):
        if "JEALOUSY_TRIGGER" not in hits:
            hits.append("JEALOUSY_TRIGGER")

    # ← 絞り込みはしない：複数インテントをそのまま返す
    #   if "JEALOUSY_TRIGGER" in hits:  ...  ← これを削除

    # 重複排除して返す
    return list(dict.fromkeys(hits))


# ここが外部公開API：インテント→メーター適用
def apply_intents_to_meters(user_data: dict, user_text: str):
    def _clip(x): return int(max(0, min(100, x)))
    m = user_data.setdefault("meters", {"like":55,"intimacy":50,"amae":45,"tereru":40,"yasuragi":50,"jealousy":20})

    intents = detect_intents(user_text)
    # print("[INTENTS]", intents, "text=", user_text)

    # --- 同時ヒットバランス係数（お好みで調整） ---
    both_factor_pos   = 0.5   # 嫉妬と同時のとき、ポジ効果は50%に弱める
    both_bonus_jeal   = 2     # 同時のとき、嫉妬に+2のボーナス
    both_penalty_like = 1     # 同時のとき、like を-1（軽くブレーキ）
    # ---------------------------------------------

    pos_hit   = "POSITIVE_AFFECTION" in intents
    invit_hit = "INVITE" in intents
    jeal_hit  = "JEALOUSY_TRIGGER" in intents
    insult_hit= "INSULT" in intents
    sad_hit   = "SADNESS" in intents

    # ベース効果（まずは普通に適用）
    if pos_hit:
        m["like"]      = _clip(m["like"] + 3)
        m["intimacy"]  = _clip(m["intimacy"] + 2)
        m["amae"]      = _clip(m["amae"] + 1)
    if invit_hit:
        m["intimacy"]  = _clip(m["intimacy"] + 2)
        m["amae"]      = _clip(m["amae"] + 1)
    if jeal_hit:
        m["jealousy"]  = _clip(m["jealousy"] + 8)
        m["yasuragi"]  = _clip(m["yasuragi"] - 3)
        m["like"]      = _clip(m["like"] - 2)
    if insult_hit:
        m["like"]      = _clip(m["like"] - 8)
        m["yasuragi"]  = _clip(m["yasuragi"] - 8)
    if sad_hit:
        m["yasuragi"]  = _clip(m["yasuragi"] + 2)
        m["amae"]      = _clip(m["amae"] + 1)

    # --- 同時ヒットのバランス補正 ---
    if jeal_hit and (pos_hit or invit_hit):
        # ポジ側の効きを弱める（直近加点を半分に圧縮するイメージ）
        # ここでは単純に「追加で逆向き補正」をかける
        # like/intimacy/amae の直近上昇見込みをざっくり半減する補正
        m["like"]      = _clip(m["like"] - int(3 * (1 - both_factor_pos)))      # 3の半分→差し引き1〜2
        m["intimacy"]  = _clip(m["intimacy"] - int(2 * (1 - both_factor_pos)))
        m["amae"]      = _clip(m["amae"] - int(1 * (1 - both_factor_pos)))

        # 嫉妬側を少しだけ上乗せ
        m["jealousy"]  = _clip(m["jealousy"] + both_bonus_jeal)

        # like に軽いブレーキ
        m["like"]      = _clip(m["like"] - both_penalty_like)

    # （任意）INSULT と POSITIVE が同時に来たときの特別処理なども足せる
    print("[INTENTS]", intents, "text=", user_text)
    print("[METERS]", m)

    # 自動語彙学習のフック（既存のまま）
    try:
        suggest_intent_phrases(user_text, intents)
    except Exception:
        pass

# ==== suggestions / auto-learn ====
import re, json, os
from collections import Counter

SUGG_PATH = "intent_suggestions.json"

_JCHUNK = re.compile(r"[ぁ-んァ-ン一-龥ー]{2,6}")  # 2-6文字の日本語塊
STOP = {"それ","これ","あれ","ここ","そこ","あの","この","その","かな","だよ","です","ます","する","した","して"}

def _load_json(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_json(path, data):
    try:
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        pass

def _extract_candidates(text: str) -> list[str]:
    # 日本語っぽい連続文字から候補を拾って、短すぎ/機能語を除外
    cands = [m.group(0) for m in _JCHUNK.finditer(text or "")]
    cands = [c for c in cands if c not in STOP]
    # 連続同一文字の圧縮（「すごーーい」→「すごーい」程度は許容、ここではそのままでもOK）
    return list(dict.fromkeys(cands))  # 重複除去（順序保持）

def suggest_intent_phrases(user_text: str, intents: list[str]):
    """
    1) 抽出した語が既知（種語/キャッシュ）ならスキップ
    2) 未知語は intent_suggestions.json にカウント追加
    3) カウント>=3で intent_cache.json に昇格（自動採用）
    """
    if not intents:
        return []

    lex = _lexicon()   # 既存のシード＋キャッシュ
    known = set().union(*[set(lex.get(k, [])) for k in lex.keys()])
    cands = [c for c in _extract_candidates(user_text) if c not in known]

    if not cands:
        return []

    sugg = _load_json(SUGG_PATH)  # {intent: {phrase: count}}
    adopted = []
    for intent in intents:
        bucket = sugg.setdefault(intent, {})
        for c in cands:
            bucket[c] = int(bucket.get(c, 0)) + 1
            if bucket[c] >= 3:  # 3回以上で自動採用
                learn_intent_phrase(intent, c)   # intent_cache.json に保存
                adopted.append((intent, c))
                # 採用済みは候補から除去しておく
                bucket.pop(c, None)
    _save_json(SUGG_PATH, sugg)
    return adopted

def approve_intent_phrase(command_text: str) -> bool:
    """
    承認: フレーズ => INTENT
    の形式で即時採用するための開発用ショートカット
    """
    m = re.search(r"承認\s*:\s*(.+?)\s*=>\s*([A-Z_]+)", command_text)
    if not m:
        return False
    phrase = m.group(1).strip()
    intent = m.group(2).strip()
    learn_intent_phrase(intent, phrase)  # そのまま採用
    # suggestions からも削除
    sugg = _load_json(SUGG_PATH)
    if intent in sugg and phrase in sugg[intent]:
        sugg[intent].pop(phrase, None)
        _save_json(SUGG_PATH, sugg)
    return True
# ==== /suggestions ====

# ==== persistent learning: add phrases to intent_cache.json ====
import json, os

_INTENT_CACHE_PATH = "intent_cache.json"

def _load_intent_cache():
    if os.path.exists(_INTENT_CACHE_PATH):
        try:
            with open(_INTENT_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_intent_cache(d):
    try:
        with open(_INTENT_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def learn_intent_phrase(intent: str, phrase: str) -> bool:
    """
    Teacherが返したインテントに対応する“言い回し”を永続保存する。
    次回以降 detect_intents() の語彙として使われる。
    """
    if not intent or not phrase:
        return False
    # 正規化（短すぎる/長すぎるのガード）
    phrase = phrase.strip()
    if not (2 <= len(phrase) <= 24):
        return False

    cache = _load_intent_cache()  # 形式: { "JEALOUSY_TRIGGER": ["前の彼女", ...], ... }
    lst = cache.get(intent, [])
    if phrase not in lst:
        lst.append(phrase)
        cache[intent] = lst
        _save_intent_cache(cache)
        return True
    return False
# ==== /persistent learning ====

