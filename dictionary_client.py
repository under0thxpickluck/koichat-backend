# dictionary_client.py
import re, time, requests
from functools import lru_cache

HEADERS = {"User-Agent": "Aichan-Dict/1.0 (+local-use)"}
TIMEOUT = 4.0

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

@lru_cache(maxsize=512)
def jisho_lookup(term: str) -> str | None:
    try:
        url = "https://jisho.org/api/v1/search/words"
        r = requests.get(url, params={"keyword": term}, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if not data.get("data"):
            return None
        d0 = data["data"][0]
        senses = d0.get("senses", [])
        # 先頭の英語定義を短くまとめる
        glosses = senses[0].get("english_definitions", []) if senses else []
        pos = (senses[0].get("parts_of_speech") or [""])[0]
        jp_word = (d0.get("japanese") or [{"word": term}])[0]
        head = jp_word.get("word") or jp_word.get("reading") or term
        meaning = ", ".join(glosses[:3])
        out = f"{head}（{pos}）= {meaning}" if meaning else None
        return _clean(out) if out else None
    except Exception:
        return None

@lru_cache(maxsize=512)
def wikipedia_summary_ja(title: str) -> str | None:
    try:
        url = f"https://ja.wikipedia.org/api/rest_v1/page/summary/{title}"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        summary = data.get("extract")
        return _clean(summary.split("。")[0] + "。") if summary else None
    except Exception:
        return None

@lru_cache(maxsize=512)
def wiktionary_extract_ja(title: str) -> str | None:
    try:
        url = "https://ja.wiktionary.org/w/api.php"
        r = requests.get(url, params={
            "action": "query", "prop": "extracts", "explaintext": 1,
            "format": "json", "titles": title
        }, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        ext = page.get("extract")
        if not ext:
            return None
        # 先頭2〜3行を要約
        lines = [ln.strip() for ln in ext.splitlines() if ln.strip()]
        return _clean(" ".join(lines[:2]))
    except Exception:
        return None

def lookup_all(term: str) -> str | None:
    """複数プロバイダを順に当てて最初にヒットした定義を返す。"""
    for fn in (jisho_lookup, wikipedia_summary_ja, wiktionary_extract_ja):
        ans = fn(term)
        if ans:
            return ans
    return None

def extract_candidates(user_text: str) -> list[str]:
    """超簡易：ひらがな/カタカナ/漢字を含む2〜8文字程度を候補に"""
    # 実運用はMeCab推奨だが、依存を増やさない版
    toks = re.findall(r"[ぁ-んァ-ン一-龥]{2,8}", user_text)
    # ノイズ除去：よくある助詞/あいづちは除外
    stop = {"それは","これで","あれは","だけど","だけどね","なんか","やっぱ","ほんと","まじで","だから"}
    toks = [t for t in toks if t not in stop]
    # 重複除去
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:3]  # 引きすぎ防止
