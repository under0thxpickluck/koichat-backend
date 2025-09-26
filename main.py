# main.py (完全修正版・バックエンドサーバー)

# --- 1. 必要なライブラリのインポート ---
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
import user_manager
import re
from datetime import datetime
import time
import json
import random

# app.pyから移植が必要な関数やデータ
from intents import detect_intents
from teacher import ask_teacher
from normalizer import normalize_user_text

# --- 2. FastAPIアプリの初期化とCORS設定 ---
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. AIモデルの読み込み ---
print("AIモデルの読み込みを開始します...")
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
    print(f"✅ LoRAアダプターの読み込みに成功: {LORA_PATH}")
except Exception as e:
    print(f"⚠️ LoRAアダプターの読み込みに失敗: {e}\n→ ベースモデルで起動します")
model.eval()
print("✅ AIモデルの準備が完了しました。")


# --- 4. app.pyから必要な関数をすべて移植 ---

INTENT_ALIAS = {"POSITIVE_AFFECTION": "affection", "AFFECTION_POS": "affection", "LIKE_POSITIVE": "affection", "DATE_PLAN": "date", "FOOD_TALK": "food", "SMALL_TALK": "neutral"}
def normalize_intent(label: str) -> str:
    if not label: return "neutral"
    return INTENT_ALIAS.get(label, label).lower()

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

ALLOWED_AUTO_UPDATE = {"persona": True, "likes": True, "tone": True, "ending": True}
_AUTO_UPDATE_LAST_TS = {}
_AUTO_UPDATE_COOLDOWN_SEC = 10 * 60

def _extract_prefs_from_user_text(text: str) -> dict:
    res = {}; t = text or ""
    m = re.search(r"(?:私を|ぼくを|俺を)?(?P<val>[^「」『』\s]{1,12})と(?:呼んで|呼ばせて)", t)
    if m: res["call"] = {"value": m.group("val").strip(), "confidence": 0.9}
    if "call" not in res:
        m = re.search(r"(?P<val>[^「」『』\s]{1,12})って呼びたい", t)
        if m: res["call"] = {"value": m.group("val").strip(), "confidence": 0.8}
    m = re.search(r"(?:私は|俺は|ぼくは)?(?P<val>[^、。！.!?]{1,20})が好き", t)
    if m: res["likes"] = {"value": m.group("val").strip(), "confidence": 0.8}
    return res

def _allow_auto_update(key: str) -> bool:
    now = time.time(); last = _AUTO_UPDATE_LAST_TS.get(key, 0)
    if now - last < _AUTO_UPDATE_COOLDOWN_SEC: return False
    _AUTO_UPDATE_LAST_TS[key] = now
    return True

def recall_intent_phrases(username: str, intent: str, k: int = 3) -> list[str]:
    if not intent: return []
    db = user_manager.load_json(username, "intents.json") or {"by_intent": {}}
    learned = (db.get("by_intent") or {}).get(intent) or []
    seeds = SEED_HINTS.get(intent, SEED_HINTS.get("neutral", []))
    merged = list(dict.fromkeys([*learned, *seeds]))
    if not merged: return []
    random.shuffle(merged)
    return merged[:k]

def build_messages(user_message: str, user_data: dict, history: list = None, history_turns=12, hints: str = ""):
    call = user_data.get("call", "キミ"); ending = user_data.get("ending", "だね")
    system = f"""あなたは恋人AI「Aちゃん」。
- 直近のやり取りを踏まえ、最新の{call}の発言に応答。予定の捏造や会話の打ち切りは禁止。
- 返答は基本1文（句点1つ、改行なし）。
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
    if history:
        for m in history[-history_turns:]:
            role = 'assistant' if m.get('role') == 'assistant' else 'user'
            content = m.get('content','')
            if role == 'user': content = f"{call}：{content}"
            recent.append({'role': role, 'content': content})
    msgs = [{'role':'system','content':system}]
    msgs += recent if recent else examples
    msgs.append({'role':'user','content': f"{call}：{user_message}"})
    return msgs

def build_model_inputs(messages):
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt_text, return_tensors="pt").to(device)

def generate_response(messages, max_new_tokens=110):
    inputs = build_model_inputs(messages)
    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, trust_remote_code=True,
                penalty_alpha=0.6, top_k=4, repetition_penalty=1.15,
                no_repeat_ngram_size=3, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs['input_ids'].shape[1]
        response_ids = out[0][input_len:]
        text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return text
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "ごめん、ちょっと考えすぎちゃったみたい…。"

def log_feedback(username: str, kind: str, user_text: str, bot_text: str, main_intent: str | None, correction: str | None = None):
    rec = { "ts": datetime.now().isoformat(timespec="seconds"), "kind": kind, "intent": (main_intent or ""), "user": (user_text or ""), "bot": (bot_text or ""), "correction": (correction or "") }
    user_manager.append_jsonl(username, "feedback/feedback_log.jsonl", rec)
    if kind == "good":
        user_manager.append_jsonl(username, "feedback/good.jsonl", rec)
    elif kind == "bad":
        if bot_text: user_manager.append_jsonl(username, "feedback/avoid_phrases.jsonl", bot_text[:80])
        user_manager.append_jsonl(username, "feedback/bad.jsonl", rec)
    elif kind == "fix":
        if correction: user_manager.append_jsonl(username, "feedback/gold_replies.jsonl", correction.strip()[:120])
        user_manager.append_jsonl(username, "feedback/fix.jsonl", rec)

_ADDR_Q = re.compile(r'(どこ住み|何県|どこの県|住所|最寄り|最寄駅|何区|何市|家どこ|住まい|住んでる|どこに住|駅名)')
def _is_address_question(user_text: str) -> bool: return bool(_ADDR_Q.search(user_text or ""))
_META_Q = re.compile(r'(AI|エーアイ|人工知能|アシスタント|LLM|モデル|プロンプト|出力|生成|メタ|システム|設定)', re.IGNORECASE)
def _is_meta_question(user_text: str) -> bool: return bool(_META_Q.search(user_text or ""))
_CONTACT_Q = re.compile(r'(line|ライン|電話|通話|番号|連絡先|id|ディスコード|discord|インスタ|instagram|位置情報|住所|家)', re.IGNORECASE)
def _is_contact_request(user_text: str) -> bool: return bool(_CONTACT_Q.search(user_text or ""))

def postprocess(text: str, username: str, user_text: str = "", user_data: dict = None) -> str:
    user_data = user_data or {}
    if _is_contact_request(user_text): return "ここでは連絡先やIDの交換はできないよ、今この場所で話そ。"
    if _is_meta_question(user_text): return "そんなことは聞かないの。今の時間を一緒に楽しもう？"
    if _is_address_question(user_text): return "東京近辺だよ。女の子に細かい住所を聞くのはちょっと…内緒ね。"
    text = text.strip()
    if not text: text = "うん、そうしよ。"
    return text

def _pop_gold_reply(username: str) -> str | None:
    lines = user_manager.read_text_file_lines(username, "gold_replies.jsonl")
    if not lines: return None
    gold = lines[0].strip()
    user_manager.write_text_file_lines(username, "gold_replies.jsonl", lines[1:])
    return gold or None

def _clip(x): return int(max(0, min(100, x)))
def update_meters_after_bot(user_data, bot_text: str):
    m=user_data.get("meters", {}); t=bot_text or ""
    if re.search(r"(ぎゅ|くっつ|甘え|なで|一緒に|そばに)", t):
        m["amae"]=_clip(m.get("amae", 50)+1); m["intimacy"]=_clip(m.get("intimacy", 50)+1)
def (user_data, rate=1):
    m = user_data.get("meters", {})
    for k in m: m[k] = _clip(m.get(k, 50)-rate)


# --- 5. APIのエンドポイントを設計・作成 ---

class ChatRequest(BaseModel):
    username: str
    message: str
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    reply: str
    history: List[Dict[str, str]]
    profile: Dict

class FeedbackRequest(BaseModel):
    username: str
    kind: str
    user_text: str
    bot_text: str
    correction: Optional[str] = None

@app.post("/api/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    username = request.username
    user_text = request.message
    history = request.history if request.history is not None else []
    mem = user_manager.load_user_memory(username)
    user_data = mem.get(username, {})
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

    try: local_intents = detect_intents(user_text) or []
    except Exception: local_intents = []
    main_intent = local_intents[0] if local_intents else "neutral"
    user_data["__last_intent"] = main_intent
    hints = ""
    gold_reply = _pop_gold_reply(username)
    if gold_reply:
        hints = gold_reply
    else:
        picked = recall_intent_phrases(username, main_intent, k=3)
        if picked: hints = " / ".join(p for p in picked if p)
    messages = build_messages(user_text, user_data, history=history, hints=hints)
    raw_bot = generate_response(messages)
    style_hint = user_data.get("style", "gentle")
    refined = raw_bot
    bot = postprocess(refined, username, user_text, user_data)
    parts = re.split(r'(?<=[。！？?])', bot)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        bot = parts[0]
        for extra in parts[1:]: history.append({"role": "assistant", "content": extra})
    user_data["__last_bot"] = bot
    update_meters_after_bot(user_data, bot)
    pos_turn = bool(re.search(r'(愛してる|大好き|だいすき|好き|会いたい|花火|祭り|見に行きたい|一緒に行|デート)', user_text))
    (user_data, rate=0 if pos_turn else 1)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": bot})
    mem[username] = user_data
    user_manager.save_user_memory(username, mem)
    return ChatResponse(reply=bot, history=history, profile=user_data)


@app.post("/api/feedback")
def handle_feedback(request: FeedbackRequest):
    mem = user_manager.load_user_memory(request.username)
    user_data = mem.get(request.username, {})
    last_intent = user_data.get("__last_intent", "")
    try:
        log_feedback(
            username=request.username, kind=request.kind,
            user_text=request.user_text, bot_text=request.bot_text,
            main_intent=last_intent, correction=request.correction
        )
        return {"status": "success", "message": "フィードバックを受け付けました。"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def read_root():
    return {"message": "KoiChat APIサーバーは正常に動作しています。"}
