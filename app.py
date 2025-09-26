# app.py (ユーザーのコードを元にAPI部分のみを修正した完全版)

from pathlib import Path
import os, json, re, unicodedata # ★os を修正
import gradio as gr # ★gr を修正
from datetime import datetime
from normalizer import normalize_user_text
import random
from intents import apply_intents_to_meters, approve_intent_phrase
from intents import detect_intents
from teacher import ask_teacher
import time
import uuid
from openai import OpenAI
from fastapi.responses import JSONResponse
from backend_gifts import router as gifts_router
from backend_images import router as images_router

import user_manager
from collections import deque
import traceback

IMG_PATH = (Path(__file__).parent / "backimage.png").resolve()
import base64
BG_B64 = base64.b64encode(open(IMG_PATH, "rb").read()).decode("ascii") if IMG_PATH.exists() else ""
import re, json
from pathlib import Path
from random import sample

ENABLE_MODEL = os.getenv("ENABLE_MODEL", "0") == "1"

if ENABLE_MODEL:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
else:
    torch = None  # 型だけ用意しておく
    AutoTokenizer = AutoModelForCausalLM = BitsAndBytesConfig = PeftModel = None
_DEFAULT_USER = (user_manager.get_user_list() or [None])[0]
FEEDBACK_DIR = (user_manager.get_user_dir(_DEFAULT_USER)/"feedback") if _DEFAULT_USER else (Path(__file__).parent/"data")

def _load_jsonl(path: Path):
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

GOLD = {}            # intent -> [reply例文...]
AVOID_PATTERNS = []  # [compiled_regex...]
FIX_RULES = []       # [(compiled_regex, repl), ...]

def load_feedback_assets():
    global GOLD, AVOID_PATTERNS, FIX_RULES
    gold = _load_jsonl(FEEDBACK_DIR / "gold_replies.jsonl")
    tmp = {}
    for row in gold:
        intent = (row.get("intent") or "default").strip()
        tmp.setdefault(intent, []).append(row.get("reply") or row.get("text") or "")
    GOLD = tmp

    avoid = _load_jsonl(FEEDBACK_DIR / "avoid_phrases.jsonl")
    AVOID_PATTERNS = [re.compile(r.get("pattern") or r.get("text"), re.I) for r in avoid if (r.get("pattern") or r.get("text"))]

    fix = _load_jsonl(FEEDBACK_DIR / "fix.jsonl")
    FIX_RULES = [(re.compile(r.get("pattern"), re.I), r.get("replace", "")) for r in fix if r.get("pattern")]

# アプリ起動時のどこか一度だけ呼ぶ
load_feedback_assets()
if ENABLE_MODEL:
    MODEL = "Qwen/Qwen2-7B-Instruct"
    LORA_PATH = "./lora-Achan"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
        device_map="auto"
    )
    try:
        model = PeftModel.from_pretrained(model, LORA_PATH)
        print(f"✅ LoRA adapter loaded: {LORA_PATH}")
    except Exception as e:
        print(f"⚠️ LoRA adapter not loaded: {e}\n→ ベースモデルで起動します")

    try:
        gc = model.generation_config
        for k in ("do_sample", "temperature", "top_p", "top_k", "typical_p"):
            if hasattr(gc, k): setattr(gc, k, None)
        print(">>> generation_config sanitized (sampling params removed)")
    except Exception as e:
        print(">>> generation_config patch skipped:", e)

    try:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
        print(f">>> DEVICE: {device}  torch={torch.__version__}  cuda_runtime={torch.version.cuda}  gpu={name}  dtype={model.dtype}")
    except Exception as _e:
        print(">>> DEVICE LOG ERROR:", _e)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
    except Exception as _e:
        print("token embeddings resize skipped:", _e)

else:
    # モデルを無効化する軽量モード（Render Free での起動用）
    MODEL = None
    tokenizer = None
    model = None
    device = "cpu"

ALLOWED_AUTO_UPDATE = {"persona": True, "likes": True, "tone": True, "ending": True}
_AUTO_UPDATE_LAST_TS = {}
_AUTO_UPDATE_COOLDOWN_SEC = 10 * 60

def ask_openai(system_prompt: str, user_prompt: str, max_tokens: int = 128):
    import os
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
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": user_prompt or "ping"},
            ],
            max_tokens=max_tokens, temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[ask_openai] error: {e}")
        return None

def teacher_refine_tone(draft: str, style: str):
    sys ="""あなたは日本語の会話文校正の専門家。
ため口・優しい・恋人調で、不自然な接続や敬語・説教・過剰な謝罪を除去し、1文に整える。
自分をAI/機械/アシストタントと絶対に言わない。
一人称は『あたし』または『わたし』。『僕』『俺』は禁止。"""
    usr = f"""以下の文をAちゃんの口調に整えてください。
- 基本ため口（〜だよ／〜だね／〜しよ）
- 敬語・説教・上から目線・過度な謝罪はNG
- 文は1文、句点は1つまで、改行なし
- 自分をAI/機械/アシスタントと絶対に言わない。
- 一人称は『あたし』または『わたし』。『僕』『俺』は禁止
- スタイルヒント: {style}
---
{draft}"""
    try:
        out = ask_openai(sys, usr)
        if not out: return draft
        return out
    except Exception: return draft

import re as _re2
def _extract_prefs_from_user_text(text: str) -> dict:
    res = {}
    t = text or ""
    m = _re2.search(r"(?:私を|ぼくを|俺を)?(?P<val>[^「」『』\s]{1,12})と(?:呼んで|呼ばせて)", t)
    if m: res["call"] = {"value": m.group("val").strip(), "confidence": 0.9}
    if "call" not in res:
        m = _re2.search(r"(?P<val>[^「」『』\s]{1,12})って呼びたい", t)
        if m: res["call"] = {"value": m.group("val").strip(), "confidence": 0.8}
    m = _re2.search(r"(?:私は|俺は|ぼくは)?(?P<val>[^、。！.!?]{1,20})が好き", t)
    if m: res["likes"] = {"value": m.group("val").strip(), "confidence": 0.8}
    return res

def _allow_auto_update(key: str) -> bool:
    now = time.time()
    last = _AUTO_UPDATE_LAST_TS.get(key, 0)
    if now - last < _AUTO_UPDATE_COOLDOWN_SEC: return False
    _AUTO_UPDATE_LAST_TS[key] = now
    return True

def _load_avoid_phrases(username: str):
    lines = user_manager.read_text_file_lines(username, "avoid_phrases.jsonl")
    return sorted(set(line.strip() for line in lines if line.strip()), key=lambda x: (-len(x), x))

def _filter_avoid_phrases(text: str, username: str) -> str:
    t = text or ""
    avoid = _load_avoid_phrases(username)
    if not avoid or not t: return t
    for bad in avoid:
        if bad and bad in t: t = t.replace(bad, "")
    t = t.replace("  ", " ").strip()
    if not t: t = "ごめん、別の言い方にするね。"
    return t

# --- ADD: fix.jsonl を適用する軽量ポストプロセス -----------------
def _apply_fix_rules(username: str, text: str) -> str:
    """
    data/<username>/feedback/fix.jsonl を1行ずつ {pattern, replace} で解釈して置換。
    失敗は握りつぶして安全側に。
    """
    try:
        p = user_manager.get_user_dir(username) / "feedback" / "fix.jsonl"
        if not p.exists():
            return text
        for line in p.read_text(encoding="utf-8").splitlines():
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pat = obj.get("pattern")
                rep = obj.get("replace", "")
                if pat:
                    text = re.sub(pat, rep, text, flags=re.I)
            except Exception:
                # 1行の不正はスキップ
                continue
        return text
    except Exception:
        return text
# -------------------------------------------------------------------


def _pop_gold_reply(username: str) -> str | None:
    lines = user_manager.read_text_file_lines(username, "gold_replies.jsonl")
    if not lines: return None
    gold = lines[0].strip()
    user_manager.write_text_file_lines(username, "gold_replies.jsonl", lines[1:])
    return gold or None

def log_feedback(username: str, kind: str, user_text: str, bot_text: str, main_intent: str | None, correction: str | None = None):
    rec = { "ts": datetime.now().isoformat(timespec="seconds"), "kind": kind, "intent": (main_intent or ""), "user": (user_text or ""), "bot": (bot_text or ""), "correction": (correction or "") }
    user_manager.append_jsonl(username, "feedback_log.jsonl", rec)
    if kind == "good":
        user_manager.append_jsonl(username, "good.jsonl", rec)
    elif kind == "bad":
        if bot_text: user_manager.append_jsonl(username, "avoid_phrases.jsonl", bot_text[:80])
        user_manager.append_jsonl(username, "bad.jsonl", rec)
    elif kind == "fix":
        if correction: user_manager.append_jsonl(username, "gold_replies.jsonl", correction.strip()[:120])
        user_manager.append_jsonl(username, "fix.jsonl", rec)

INTENT_ALIAS = {"POSITIVE_AFFECTION": "affection", "AFFECTION_POS": "affection", "LIKE_POSITIVE": "affection", "DATE_PLAN": "date", "FOOD_TALK": "food", "SMALL_TALK": "neutral"}
def normalize_intent(label: str) -> str:
    if not label: return "neutral"
    return INTENT_ALIAS.get(label, label).lower()

def recall_intent_phrases(username: str, intent: str, k: int = 3) -> list[str]:
    if not intent: return []
    db = user_manager.load_json(username, "intents.json") or {"by_intent": {}}
    learned = (db.get("by_intent") or {}).get(intent) or []
    seeds = SEED_HINTS.get(intent, SEED_HINTS.get("neutral", []))
    merged = list(dict.fromkeys([*learned, *seeds]))
    if not merged: return []
    random.shuffle(merged)
    return merged[:k]

def remember_intent_phrases(username: str, intent: str, phrases: list[str], cap: int = 200):
    if not phrases: return
    db = user_manager.load_json(username, "intents.json") or {"by_intent": {}, "stats": {}}
    by = db.setdefault("by_intent", {}).setdefault(intent, [])
    norm = [re.sub(r'\s+', ' ', (p or '').strip()) for p in phrases if p and p.strip()]
    for p in norm:
        if p and p not in by: by.append(p)
    if len(by) > cap: by[:] = by[-cap:]
    st = db.setdefault("stats", {}).setdefault(intent, {"n":0,"last_ts":0})
    st["n"] = len(by); st["last_ts"] = int(time.time())
    user_manager.save_json(username, "intents.json", db)

SEED_HINTS = {
    "greeting":  ["おはよう","やっほー","こんちゃ","会いたかった"],
    "affection": [ "大好きだよ","ぎゅってしたい","独り占めしたい","一緒に過ごしたい","キミが一番", "そばにいたい","ぎゅーしたい","会いたいよ","大事にしたい","ずっと一緒にいたい","なでたい","キスしたい"],
    "worry":     ["大丈夫？","無理しないで","そばにいるよ","休もう","抱きしめたい"],
    "jealousy":  ["ちょっとやきもち","私だけ見てて","他の子と話してた？","独占したい"],
    "date":      ["一緒に行こう","今夜どう？","次いつ会える？","手つなご？", "今夜会える？","どこ行く？","夜景見よ","映画いこ","カフェいこ","花火見に行こ","週末空いてる？"],
    "food":      ["一緒に食べよう","何が食べたい？","お腹すいたね","夜食つくる？", "ラーメン食べよ","甘いもの食べたい","寿司いこ","焼肉いこ","ピザ食べよ","おやつタイム","お腹減った？"],
    "banter":    ["冗談でしょ？","もう〜","かわいいこと言うね","もーっ"],
    "neutral":   ["今日はどう？","どうした？","今はどうしたい？","それ良さそう"],
}

def seeds_merge_into_intents(username: str, cap_per_label: int = 400):
    db = user_manager.load_json(username, "intents.json") or {"by_intent": {}, "stats": {}}
    by = db.setdefault("by_intent", {})
    for lab, arr in SEED_HINTS.items():
        cur = by.setdefault(lab, [])
        merged = list(dict.fromkeys([*(cur or []), *arr]))
        by[lab] = merged[-cap_per_label:]
    st = db.setdefault("stats", {})
    for lab in SEED_HINTS.keys():
        rec = st.setdefault(lab, {"n":0,"last_ts":0})
        rec["n"] = len(by.get(lab, []))
        rec["last_ts"] = int(time.time())
    user_manager.save_json(username, "intents.json", db)

import re as _re_helpers
def _enforce_first_person(text: str, prefer: str = "私") -> str:
    return _re_helpers.sub(r'(僕|ぼく|俺|おれ)(?=(は|が|も|を|に|で|と|の|へ|や|より|だ|です|でした|、|。|!|！|\?|？|$))', prefer, text)
_ADDR_Q = _re_helpers.compile(r'(どこ住み|何県|どこの県|住所|最寄り|最寄駅|何区|何市|家どこ|住まい|住んでる|どこに住|駅名)')
def _is_address_question(user_text: str) -> bool:
    return bool(_ADDR_Q.search(user_text or ""))
def _soft_fix_geo(text: str) -> str:
    text = _re_helpers.sub(r'[一-龥]{1,4}(都|道|府|県)', '東京近辺', text)
    text = _re_helpers.sub(r'[一-龥]{1,4}(市|区)', '東京近辺', text)
    return text
_META_Q = _re_helpers.compile(r'(AI|エーアイ|人工知能|アシスタント|LLM|モデル|プロンプト|出力|生成|メタ|システム|設定|chain of thought)', _re_helpers.IGNORECASE)
def _is_meta_question(user_text: str) -> bool:
    return bool(_META_Q.search(user_text or ""))
_CONTACT_Q = _re_helpers.compile(r'(line|ライン|電話|通話|番号|連絡先|id|ディスコード|discord|インスタ|instagram|位置|ロケーション|位置情報|共有|住所|家|地図|送って|教えて)', _re_helpers.IGNORECASE)
def _is_contact_request(user_text: str) -> bool:
    return bool(_CONTACT_Q.search(user_text or ""))
def _avoid_question_spam(text: str, last: str) -> str:
    t = (text or "").strip(); l = (last or "").strip()
    if not l: return t
    if (t == l) or (t.endswith(('どう思う？','どう思う?')) and l.endswith(('どう思う？','どう思う?'))) or (t.endswith(('？','?')) and l.endswith(('？','?'))):
        t = _re_helpers.sub(r'[？?]+$', '。', t)
    return t
def _typo_fix(text: str) -> str:
    typo_map = {"大失聴": "大失敗"};
    for k, v in typo_map.items(): text = text.replace(k, v)
    return text

def build_messages(user_message: str, user_data: dict, state=None, history_turns=12, hints: str = ""):
    call = user_data.get("call", "キミ"); ending = user_data.get("ending", "だね")
    system = f"""あなたは恋人AI「Aちゃん」。
- 直近のやり取りを踏まえ、最新の{call}の発言に応答。予定の捏造や会話の打ち切りは禁止。
- 返答はほとんどの場合は1文（句点1つ、改行なし）。
- 口調は砕けたため口：「〜だよ／〜だね／〜しよ」を基本。敬語や「〜しなさい」は禁止。
- 絶対に上から目線・命令形は使わない。優しく、甘め寄りを基本とする。"""
    m = user_data.get("meters", {});
    def V(k, d=50):
        try: return int(max(0, min(100, m.get(k, d))))
        except: return d
    style = "gentle"
    if V('jealousy') >= 60: style = "jealous"
    elif V('amae') >= 60 or V('intimacy') >= 65: style = "flirty"
    elif V('yasuragi') >= 65: style = "comfort"
    system += f"\n[現在の関係メータ 0-100]\n- 好感度: {V('like')} / 親密度: {V('intimacy')} / 甘え: {V('amae')} / 照れ: {V('tereru')} / 安心度: {V('yasuragi')} / 嫉妬: {V('jealousy')}\n[style] = {style}"
    if hints: system += f"\n[意図ヒント]\n- 似たニュアンスの例: {hints}\n- これをコピペせず、自然に“それっぽい”語彙やトーンに寄せること。"
    likes = ", ".join(user_data.get("likes", [])[:5]); dislikes = ", ".join(user_data.get("dislikes", [])[:5])
    system += f"\n[記憶ヒント]\n- 好き: {likes or '（未登録）'}\n- 苦手: {dislikes or '（未登録）'}"
    examples = [{"role":"user", "content": f"{call}：会いたいなぁ"}, {"role":"assistant", "content": f"私も会いたい…その気持ちだけで今日は頑張れそう{ending}"}]
    recent = []
    if state:
        for m in state[-history_turns:]:
            role = 'assistant' if m.get('role') == 'assistant' else 'user'
            content = m.get('content','')
            if role == 'user': content = f"{call}：{content}"
            recent.append({'role': role, 'content': content})
    msgs = [{'role':'system','content':system}]
    msgs += recent if recent else examples
    msgs.append({'role':'user','content': f"{call}：{user_message}"})
    return msgs

def build_model_inputs(messages):
    # 軽量モードや初期化失敗時は None を返す
    if (not ENABLE_MODEL) or (tokenizer is None) or (device is None):
        return None
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt_text, return_tensors="pt").to(device)


def generate_response(messages, max_new_tokens=110):
    """
    モデル有効時: これまで通り torch+HF で生成
    軽量モード: OpenAI にフォールバック（失敗時は簡易メッセージ）
    """
    # 軽量モード or 未初期化 → OpenAI にフォールバック
    if (not ENABLE_MODEL) or (tokenizer is None) or (model is None):
        sys_msg = ""
        for m in messages:
            if m.get("role") == "system":
                sys_msg = m.get("content") or ""
                break
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content") or ""
                break
        out = ask_openai(sys_msg, user_msg, max_tokens=max_new_tokens)
        return out or "うん、そうしよ。"

    # ここから従来のローカルモデル経路
    inputs = build_model_inputs(messages)
    if inputs is None:
        sys_msg = ""
        for m in messages:
            if m.get("role") == "system":
                sys_msg = m.get("content") or ""
                break
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content") or ""
                break
        out = ask_openai(sys_msg, user_msg, max_tokens=max_new_tokens)
        return out or "うん、そうしよ。"

    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                trust_remote_code=True,
                penalty_alpha=0.6,
                top_k=4,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        response_ids = out[0][input_len:]
        text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return text
    except Exception as e:
        print(f"[local-generate error] {e}")
        sys_msg = ""
        for m in messages:
            if m.get("role") == "system":
                sys_msg = m.get("content") or ""
                break
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content") or ""
                break
        out = ask_openai(sys_msg, user_msg, max_tokens=max_new_tokens)
        return out or "うん、そうしよ。"

def postprocess(text: str, username: str, user_text: str = "", user_data: dict = None) -> str:
    user_data = user_data or {}
    text = _typo_fix(text)
    if _is_contact_request(user_text): return "ここでは連絡先やIDの交換はできないよ、今この場所で話そ。"
    if _is_meta_question(user_text): return "そんなことは聞かないの。今の時間を一緒に楽しもう？"
    if _is_address_question(user_text): return "東京近辺だよ。女の子に細かい住所を聞くのはちょっと…内緒ね。"
    else: text = _soft_fix_geo(text)
    text = _enforce_first_person(text, prefer="私")
    text = _avoid_question_spam(text, user_data.get("__last_bot", ""))
    text = _filter_avoid_phrases(text, username)
    text = _apply_fix_rules(username, text)
    text = text.strip()
    if not text: text = "うん、そうしよ。"
    return text

def openai_healthcheck():
    results = {"ask_openai": {"ok": False, "detail": ""}, "ask_teacher": {"ok": False, "detail": ""}}
    try:
        pong = ask_openai("You are a healthcheck.", "ping", max_tokens=4)
        results["ask_openai"]["ok"] = bool(pong); results["ask_openai"]["detail"] = (pong or "")[:60]
    except Exception as e:
        results["ask_openai"]["ok"] = False; results["ask_openai"]["detail"] = f"err:{e}"
    try:
        from teacher import ask_teacher
        ans = ask_teacher("INTENT_HELP", "テスト")
        results["ask_teacher"]["ok"] = bool(ans); results["ask_teacher"]["detail"] = (json.dumps(ans, ensure_ascii=False) if ans else "")[:60]
    except Exception as e:
        results["ask_teacher"]["ok"] = False; results["ask_teacher"]["detail"] = f"err:{e}"
    okA = "✅" if results["ask_openai"]["ok"] else "❌"; okB = "✅" if results["ask_teacher"]["ok"] else "❌"
    return (f"**OpenAI接続テスト**\n- app.py 経路（ask_openai）: {okA} {results['ask_openai']['detail']}\n- teacher.py 経路（ask_teacher）: {okB} {results['ask_teacher']['detail']}\n\n環境変数 OPENAI_API_KEY を設定されているか確認してください。")

def _clip(x): return int(max(0, min(100, x)))
def update_meters_after_bot(user_data, bot_text: str):
    m=user_data.get("meters", {}); t=bot_text or ""
    if re.search(r"(ぎゅ|くっつ|甘え|なで|一緒に|そばに)", t):
        m["amae"]=_clip(m["amae"]+1); m["intimacy"]=_clip(m["intimacy"]+1)
def decay_meters(user_data, rate=1):
    m = user_data.get("meters", {})
    for k in m: m[k] = _clip(m[k]-rate)

def tail_provenance(username: str, n=3):
    try:
        lines = user_manager.read_text_file_lines(username, "feedback/teacher_provenance.jsonl")
        return "\n".join(lines[-n:])
    except Exception:
        return ""

def reply_fn(username, user_text, state, remember_text, forget_text):
    if not username: return state, "キャラクターを選択または作成してください。", "{}"
    mem = user_manager.load_user_memory(username)
    user_data = mem.get(username, {})
    history = state.copy() if state else []
    if not isinstance(history, list): history = []
    if not isinstance(user_data, dict): user_data = {}
    user_text = user_text or ""
    
    auto = user_data.get("auto_update") or {};
    for k, v in ALLOWED_AUTO_UPDATE.items(): auto.setdefault(k, v)
    user_data["auto_update"] = auto

    ex = _extract_prefs_from_user_text(user_text)
    def _update_if_allowed(key: str, val: str):
        if not user_data.get("auto_update", {}).get(key, False): return
        if not _allow_auto_update(key): return
        if key == "likes":
            likes = user_data.get("likes") or []
            if isinstance(likes, list) and val not in likes:
                likes.append(val); user_data["likes"] = likes
        else: user_data[key] = val
    for k, item in ex.items():
        if item.get("confidence", 0) >= 0.7 and item.get("value"):
            _update_if_allowed(k, item["value"])

    if remember_text:
        if remember_text.startswith("好き:"):
            val = remember_text.split(":",1)[1].strip()
            if val:
                likes = user_data.setdefault("likes", [])
                if val not in likes: likes.append(val)
        elif remember_text.startswith("嫌い:"):
            val = remember_text.split(":",1)[1].strip()
            if val:
                dislikes = user_data.setdefault("dislikes", [])
                if val not in dislikes: dislikes.append(val)
    if forget_text:
        for k in ("likes", "dislikes"):
            arr = user_data.get(k, [])
            user_data[k] = [x for x in arr if forget_text not in x]

    def _fallback_intent(t: str) -> str:
        tl = (t or "").lower()
        if any(k in tl for k in ["すき","好き","愛してる","らぶ","chu","ぎゅ","ハグ"]): return "affection"
        if any(k in t for k in ["デート","遊び","会お","行かない","行こう","出かけ"]): return "date"
        if any(k in t for k in ["ラーメン","ご飯","オムライス","食べ","夜食"]): return "food"
        if any(k in t for k in ["ばか","ばーか","あほ"]): return "banter"
        return "neutral"
    try: local_intents = detect_intents(user_text) or []
    except Exception: local_intents = []
    main_intent = local_intents[0] if local_intents else _fallback_intent(user_text)
    user_data["__last_intent"] = main_intent

    hints = ""
    gold_reply = _pop_gold_reply(username)
    if gold_reply:
        print(f"[gold reply] using correction: {gold_reply}")
        hints = gold_reply
    else:
        picked = recall_intent_phrases(username, main_intent, k=3)
        if picked: hints = " / ".join(p for p in picked if p)

    messages = build_messages(user_text, user_data, state=history, history_turns=12, hints=hints)
    raw_bot = generate_response(messages)
    style_hint = user_data.get("style", "gentle")
    refined = teacher_refine_tone(raw_bot, style_hint)
    bot = postprocess(refined, username, user_text, user_data)

    user_data["__last_bot"] = bot
    update_meters_after_bot(user_data, bot)
    pos_turn = bool(re.search(r'(愛してる|大好き|だいすき|好き|会いたい|花火|祭り|見に行きたい|一緒に行|デート)', user_text))
    decay_meters(user_data, rate=0 if pos_turn else 1)
 
    history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": bot}]

    def _bar(n):
        n = int(max(0, min(100, int(n))))
        filled = n // 5
        return "█" * filled + "·" * (20 - filled) + f" {n:3d}"
    meters = user_data.get("meters", {})

    positive_meters = [
        meters.get('like', 50),
        meters.get('amae', 50),
        meters.get('tereru', 40),
        meters.get('yasuragi', 50)
    ]
    intimacy = sum(positive_meters) / len(positive_meters)
    meters['intimacy'] = _clip(intimacy)

    meter_view = "\n".join([f"好感度     | {_bar(meters.get('like',50))}", f"親密度     | {_bar(meters.get('intimacy',50))}", f"甘え       | {_bar(meters.get('amae',50))}", f"照れ       | {_bar(meters.get('tereru',40))}", f"安心度     | {_bar(meters.get('yasuragi',50))}", f"嫉妬       | {_bar(meters.get('jealousy',20))}"])

    mem[username] = user_data
    user_manager.save_user_memory(username, mem)
    
    profile = (
        json.dumps(user_data, ensure_ascii=False, indent=2) + 
        "\n\n[meters]\n" + meter_view +
        "\n\n[teacher_log]\n" + tail_provenance(username, 3)
    )
    return history, bot, profile

def on_send(username, user_text, state, remember_text, forget_text):
    if not username: return state, gr.update(), "", "", "", "{}"
    if not (user_text or "").strip(): return state, gr.update(), "", "", "", gr.update()
    new_state, bot, profile = reply_fn(username, user_text, state, remember_text, forget_text)
    return new_state, gr.update(value=new_state), "", "", "", gr.update(value=profile)

def on_feedback(username, kind, state, user_box, fix_box):
    if not username: return (state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "キャラクターを選択してください。", gr.update())
    last_user = ""
    if isinstance(state, list):
        for msg in reversed(state):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
    mem = user_manager.load_user_memory(username)
    u = mem.get(username, {})
    last_bot = u.get("__last_bot", "")
    last_intent = u.get("__last_intent", "")
    correction = (fix_box or "").strip() if kind == "fix" else None
    try: log_feedback(username, kind, last_user, last_bot, last_intent, correction)
    except Exception as e: print(f"[feedback error for {username}]:", e)
    latest_mem = user_manager.load_user_memory(username)
    profile_data = latest_mem.get(username, {})
    profile = json.dumps(profile_data, ensure_ascii=False, indent=2)
    status = {"good": "✅ いいね！", "bad":  "🚫 次から避けるね。", "fix":  "✍ 修正ありがとう！"}.get(kind, "OK")
    return (state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=profile), gr.update(value=status), "" if kind == "fix" else gr.update())

# OpenAIクライアントを初期化 (既存のものを再利用してもOK)
try:
    # 環境変数からAPIキーを読み込む
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"⚠️ OpenAIクライアントの初期化に失敗: {e}")
    client = None

def text_to_speech_fn(text):
    """テキストを受け取り、音声ファイルへのパスを返す関数"""
    if not client or not text:
        # クライアントがない、またはテキストが空なら何もしない
        return None
    try:
        # 他の音声と被らないように、ユニークなファイル名を生成
        speech_file_path = Path(__file__).parent / f"temp_speech_{uuid.uuid4()}.mp3"
        
        # OpenAIの音声合成APIを呼び出し
        response = client.audio.speech.create(
          model="tts-1",      # 音声合成モデル
          voice="nova",       # 声の種類
          input=text          # 音声にしたいテキスト
        )
        
        # 生成された音声データをファイルとして保存
        response.stream_to_file(speech_file_path)
        
        # 保存した音声ファイルの「場所」を返す
        return str(speech_file_path)
        
    except Exception as e:
        print(f"音声生成中にエラーが発生しました: {e}")
        return None
# === [ADD] Minimal UI handlers to satisfy click/change bindings ===
def create_user_ui(new_name: str):
    """
    新規キャラ名を受け取り、存在しなければ初期データを保存。
    戻り値は [Dropdown更新, ステータスメッセージ]
    ※ 他機能に影響しない初期値のみを設定（points/meters）
    """
    name = (new_name or "").strip()
    # Dropdownの更新用（choices / value は gr.update で返す）
    def _dropdown_update(selected=None):
        choices = list(user_manager.get_user_list() or [])
        if selected and selected not in choices:
            choices.append(selected)
        return gr.update(choices=choices, value=selected)
    if not name:
        return _dropdown_update(), "⚠️ 新規キャラクター名を入力してください。"

    # 既存メモリの取得
    mem = user_manager.load_user_memory(name)
    # 期待形：mem は { name: user_data } 形式を想定（既存コードと整合）
    user_data = {}
    if isinstance(mem, dict) and name in mem:
        user_data = mem.get(name, {}) or {}
    else:
        # まだ無ければ初期レコードを用意（他機能に干渉しない、必要最小限のみ）
        user_data = {
            "points": 100,
            "meters": {
                "like": 50, "amae": 50, "tereru": 40, "yasuragi": 50, "jealousy": 20
            }
        }
        # mem の形を合わせて保存
        mem = {name: user_data}
        user_manager.save_user_memory(name, mem)

    # Dropdown を最新化し、作成/選択メッセージを返す
    return _dropdown_update(selected=name), f"✅ キャラクター「{name}」を作成/選択しました。"


def on_user_select(username: str):
    """
    Dropdownでキャラを選んだ時に、右ペインのプロフィールや状態を更新。
    戻り値は [prof(Code), user_status(Markdown), chat(Chatbot), state(State)]
    """
    if not username:
        # prof / status / chat / state
        return "{}", "⚠️ キャラクターが選択されていません。", [], []

    mem = user_manager.load_user_memory(username)
    # mem 期待形：{ username: {...} }。無ければ空で表示だけする
    if isinstance(mem, dict):
        data = mem.get(username, {}) or {}
    else:
        data = {}

    prof_json = json.dumps(data, ensure_ascii=False, indent=2)
    status = f"✅ @{username} を選択中"
    # 既存の構成では履歴を永続保存していないので、chat/state は空配列で初期化
    return prof_json, status, [], []


# =========================
# Gradio UI Layout
# =========================
custom_css = """
:root, body, .gradio-container { background: transparent ! important; }
#bg_image{position: fixed; inset: 0; width: 100%; height: 100%; object-fit: contain; object-position: center bottom; opacity: 0.18; pointer-events: none; z-index: 0;}
#content{position: relative; z-index: 1;}
.chatbot, .gradio-container .chatbot, .gr-chatbot {background: rgba(255,255,255,0.75) !important;}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML(f"<img id='bg_image' src='data:image/png;base64,{BG_B64}' />")
    gr.Markdown("# あいちゃん WebUI")
    with gr.Column(elem_id="content"):
        with gr.Row():
            user_select = gr.Dropdown(label="キャラクター選択", choices=user_manager.get_user_list(), value=user_manager.get_user_list()[0] if user_manager.get_user_list() else None, scale=2,allow_custom_value=True,)
            new_user_name = gr.Textbox(label="新規キャラクター名", scale=1, placeholder="例: Airi")
            create_user_btn = gr.Button("作成", scale=1)
        user_status = gr.Markdown("")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(type="messages", label="あいちゃん", height=420)
                user_box = gr.Textbox(label="あなたのメッセージ", placeholder="いまなにしてる？ など", autofocus=True)
                with gr.Row():
                    remember_box = gr.Textbox(label="記憶に追加（例: 好き: ラーメン）", scale=2)
                    forget_box   = gr.Textbox(label="記憶から削除（キーワード）",    scale=1)
                send = gr.Button("送信", variant="primary")
                health = gr.Button("OpenAI接続テスト")
                health_out = gr.Markdown("")
            with gr.Column(scale=2):
                prof = gr.Code(label="キャラクター・プロフィール", language="json", interactive=False, value="{}")
                gr.Markdown("- 記憶の保存先: data/各キャラクターフォルダ")
                with gr.Row():
                    thumbs_up   = gr.Button("👍 いいね", variant="secondary")
                    thumbs_down = gr.Button("👎 いまいち", variant="secondary")
                with gr.Row():
                    fix_box     = gr.Textbox(label="✍ 手直し（この表現が良い）", placeholder="例）今日は寿司いこ？", lines=1, scale=3)
                    fix_send    = gr.Button("✍ 反映", variant="primary", scale=1)
                feedback_status = gr.Markdown("")

                # API定義 (Gradio Interface)
                gr.Interface(fn=text_to_speech_fn, inputs="text", outputs="audio", api_name="text_to_speech", visible=False)

    # UIイベントの定義
    state = gr.State([])
    create_user_btn.click(create_user_ui, inputs=[new_user_name], outputs=[user_select, user_status])
    user_select.change(on_user_select, inputs=[user_select], outputs=[prof, user_status, chat, state])
    send.click(fn=on_send, inputs=[user_select, user_box, state, remember_box, forget_box], outputs=[state, chat, user_box, remember_box, forget_box, prof])
    user_box.submit(fn=on_send, inputs=[user_select, user_box, state, remember_box, forget_box], outputs=[state, chat, user_box, remember_box, forget_box, prof])
    health.click(fn=openai_healthcheck, inputs=None, outputs=health_out)
    feedback_outputs = [state, chat, user_box, remember_box, forget_box, prof, feedback_status, fix_box]
    thumbs_up.click(fn=lambda username, state, user_box, fix_box: on_feedback(username, "good", state, user_box, fix_box), inputs=[user_select, state, user_box, fix_box], outputs=feedback_outputs)
    thumbs_down.click(fn=lambda username, state, user_box, fix_box: on_feedback(username, "bad", state, user_box, fix_box), inputs=[user_select, state, user_box, fix_box], outputs=feedback_outputs)
    fix_send.click(fn=lambda username, state, user_box, fix_box: on_feedback(username, "fix", state, user_box, fix_box), inputs=[user_select, state, user_box, fix_box], outputs=feedback_outputs)

demo.queue() 

# ---- API（include_router 方式：/backend 配下）----
from fastapi import APIRouter

router = APIRouter(prefix="/backend")


@router.get("/errors")
def list_recent_errors(limit: int = 20):
    try:
        lim = max(1, min(int(limit), 200))
    except Exception:
        lim = 20
    return JSONResponse(content={"errors": list(ERROR_BUFFER)[-lim:]})


@router.get("/get_user_data")
def get_user_data_api_endpoint(username: str):
    if not username:
        return JSONResponse(status_code=400, content={"error": "Username is required"})
    try:
        mem = user_manager.load_user_memory(username)
        user_data = mem.get(username, {}) if isinstance(mem, dict) else {}
        meters = user_data.get("meters", {})
        points = user_data.get("points", 100)
        return JSONResponse(content={
            "username": username,
            "points": points,
            "intimacy": meters.get('intimacy', 50),
            "like": meters.get('like', 50),
            "amae": meters.get('amae', 50),
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Gradio の FastAPI 本体に /backend ルーターを直接登録（mount ではなく include）
# demo.app.include_router(router)

# （デバッグ表示：登録された個別ルートが見えるはず）
print("=== Registered routes (after include_router) ===")
for r in demo.app.routes:
    try:
        print(getattr(r, "path", r))
    except Exception:
        pass
# ---- ここまで ----

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()                                 # 1) まず作る

@app.middleware("http")
async def _error_tap(request, call_next):
    ts_path  = request.url.path
    ts_method = request.method
    try:
        resp = await call_next(request)
        # ハンドラ側で500系を返した場合も拾っておく
        if getattr(resp, "status_code", 200) >= 500:
            _push_error_record(ts_path, ts_method, resp.status_code, None, note="ハンドラ内で500系レスポンス")
        return resp
    except Exception as e:
        # 例外は記録してから再スロー（既存のエラーハンドリングはそのまま動く）
        _push_error_record(ts_path, ts_method, 500, e)
        raise


# 2) ルーターを全部のせる（/backend/* 系を親に登録）
app.include_router(router)                      # ← /backend/get_user_data（あなたの既存API）
app.include_router(gifts_router)                # ← プレゼントAPI
app.include_router(images_router)               # ← 画像生成API（/backend/images/*）


print("=== FastAPI routes (app) ===")
for r in app.routes:
    try:
        print(getattr(r, "path", r))
    except Exception:
        pass


# 3) 静的ファイル：生成画像を配信（/generated/xxx.png）
gen_dir = Path(__file__).parent / "generated"
gen_dir.mkdir(exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(gen_dir)), name="generated")

# 4) Gradio UI を /app にマウント
app = gr.mount_gradio_app(app, demo, path="/app")

# 直近エラー200件を保持（プロセス内）
ERROR_BUFFER = deque(maxlen=200)

def _push_error_record(path: str, method: str, status: int, err: Exception | None, note: str = ""):
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "path": path, "method": method, "status": status,
        "error": (type(err).__name__ if err else ""),
        "detail": (str(err)[:300] if err else note)  # 返却は短め
    }
    # 追記
    ERROR_BUFFER.append(rec)

def _format_errors_markdown(limit: int = 20) -> str:
    if not ERROR_BUFFER:
        return "（直近のエラーはありません）"
    rows = list(ERROR_BUFFER)[-limit:][::-1]
    lines = [f"- {r['ts']} [{r['status']}] {r['method']} {r['path']} — {r.get('error','')} {r.get('detail','')}" for r in rows]
    return "### 直近エラー（最大{limit}件）\n" + "\n".join(lines)


if __name__ == "__main__":
    import os, uvicorn
    host = os.getenv("HOST", "0.0.0.0")      # ← 0.0.0.0 で外部公開
    port = int(os.getenv("PORT", "7860"))    # ← Render が PORT を渡します
    uvicorn.run(app, host=host, port=port)