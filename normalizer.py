# normalizer.py
import re
from functools import lru_cache
from rapidfuzz import fuzz, process
from wordfreq import zipf_frequency, top_n_list
from sudachipy import tokenizer, dictionary

# Sudachi 初期化
_tokenizer = dictionary.Dictionary().create()
MODE = tokenizer.Tokenizer.SplitMode.C  # 最長一致で安定

# 日本語の高頻度語彙（上位5万語程度）
@lru_cache(maxsize=1)

def vocab():
    # wordfreq から日本語上位語を取得
    v = top_n_list("ja", 50000)  # ← n_top= ではなく、引数だけ
    # よく使う口語も少しだけ足す
    extra = ["おはよ", "まじ", "やばい", "ありがと", "ごめん", "ねむい", "つかれた", "ラーメン", "カフェ"]
    return list(set(v) | set(extra))


VOCAB = vocab()

STOP_SINGLE = {"ねえ","ねぇ","え","あ","うん","ううん","はぁ","ふーん"}
PUNCT_END = re.compile(r"[。！？!?？]$")

def _freq(word: str) -> float:
    # Zipf頻度（~1~8、5以上ならかなり一般語）
    return zipf_frequency(word, "ja")

def _is_common(word: str, thres=3.0) -> bool:
    return _freq(word) >= thres

def _tok(text: str):
    return _tokenizer.tokenize(text, MODE)

def _best_cand(word: str):
    # 近い語を上位から探索（距離スコア > 90 くらいで置換）
    match = process.extractOne(word, VOCAB, scorer=fuzz.WRatio, score_cutoff=90)
    return match[0] if match else None

def normalize_user_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t

    # 空白・改行の整理
    t = re.sub(r"\s+", " ", t)

    # 軽い表記ゆれ（よくある手入力ミス）
    t = t.replace("なにすれう", "なにする").replace("なにすれ", "なにする")

    # 形態素で分割し、未知語/低頻度語だけを候補補正
    out = []
    for m in _tok(t):
        surf = m.surface()
        base = m.dictionary_form() or surf
        # Sudachiの未知語は base==surf になりがち。頻度も併用して判定
        if len(surf) >= 2 and not _is_common(surf):
            cand = _best_cand(surf)
            out.append(cand or surf)
        else:
            out.append(surf)

    s = "".join(out)

    # 「なに/どこ/どう/いつ/なんで」で終わってるのに疑問符がない場合だけ付与
    if re.search(r"(なに|どこ|どう|いつ|なんで)$", s) and not PUNCT_END.search(s):
        s += "？"

    # 句読点の暴発を軽く抑制
    s = re.sub(r"[。\.]{2,}", "。", s)
    s = re.sub(r"[？\?]{2,}", "？", s)
    s = re.sub(r"[！!]{2,}", "！", s)

    return s
