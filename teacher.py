# teacher.py
import os, json, time
from rapidfuzz import fuzz, process
import openai
import json

MAX_CALLS_PER_DAY = 20
CACHE_PATH = "teacher_cache.json"
DATASET_PATH = "teacher_dataset.jsonl"
BUDGET_PATH = "teacher_budget.json"
PROVENANCE_LOG = "teacher_provenance.jsonl"

def _load(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def _save(path, obj, jsonl=False):
    try:
        if jsonl:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _today(): return time.strftime("%Y-%m-%d")
def _can_call():
    b = _load(BUDGET_PATH, {})
    return int(b.get(_today(), 0)) < MAX_CALLS_PER_DAY
def _inc_call():
    b = _load(BUDGET_PATH, {})
    b[_today()] = int(b.get(_today(), 0)) + 1
    _save(BUDGET_PATH, b)

def _normalize(s: str) -> str:
    return (s or "").strip().replace("\u3000", " ").lower()

def _cache_get(q: str):
    cache = _load(CACHE_PATH, {})
    if q in cache: return cache[q]
    if cache:
        cand, score, _ = process.extractOne(q, list(cache.keys()), scorer=fuzz.WRatio)
        if score >= 92:
            return cache[cand]
    return None

def _cache_put(q: str, ans: dict):
    cache = _load(CACHE_PATH, {})
    cache[q] = ans
    _save(CACHE_PATH, cache)

def _log_pair(task: str, user_text: str, teacher_out: dict):
    rec = {"ts": int(time.time()), "task": task, "input": user_text, "label": teacher_out}
    _save(DATASET_PATH, rec, jsonl=True)

def _prov_log(event: dict):
    try:
        with open(PROVENANCE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass

def ask_teacher(tag: str, text: str):
    import os, json
    try:
        from openai import OpenAI
    except ImportError:
        print("openai SDK が見つかりません")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY が未設定です")
        return None

    project = os.getenv("OPENAI_PROJECT")
    client = OpenAI(api_key=api_key, project=project if project else None)

    system = "You are a Japanese writing coach. Return compact JSON: {\"meters\":{}, \"intents\":[], \"phrases\":[]}"
    user = f"[{tag}] {text}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=200,
            temperature=0.3,
        )
        content = (resp.choices[0].message.content or "").strip()
        # JSON想定。失敗時は raw を返す
        try:
            return json.loads(content)
        except Exception:
            return {"raw": content}
    except Exception as e:
        print(f"[ask_teacher] error: {e}")
        return None




    _inc_call()
    _cache_put(q, ans)
    _log_pair(task, user_text, ans)
    _prov_log({"ts": int(time.time()), "source": "openai", "model": "gpt-4o-mini",
               "task": task, "query": qnorm, "result_preview": str(ans)[:160]})
    return ans
