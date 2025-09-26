from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING
import os
import json


import os as _os
_ENABLE_MODEL = _os.getenv("ENABLE_MODEL", "0") == "1"

if _ENABLE_MODEL:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as e:
        print(f"[light-mode(images)] {e}; falling back to _ENABLE_MODEL=0")
        _ENABLE_MODEL = False
        torch = None  # type: ignore
else:
    torch = None  # type: ignore

def _torch_available() -> bool:
    return _ENABLE_MODEL and (torch is not None)


# ==== gate diffusers too ====
if _ENABLE_MODEL:
    try:
        # 使っているものをまとめてここで import（名前はあなたのコードに合わせて）
        from diffusers import (
            DiffusionPipeline,
            AutoPipelineForText2Image,
            AutoPipelineForImage2Image,
            StableDiffusionXLPipeline,
        )
    except ModuleNotFoundError as e:
        print(f"[light-mode(images:diffusers)] {e}; falling back to _ENABLE_MODEL=0")
        _ENABLE_MODEL = False
        DiffusionPipeline = AutoPipelineForText2Image = AutoPipelineForImage2Image = StableDiffusionXLPipeline = None
else:
    DiffusionPipeline = AutoPipelineForText2Image = AutoPipelineForImage2Image = StableDiffusionXLPipeline = None
# ==== end ====
if TYPE_CHECKING:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        AutoPipelineForText2Image,
        AutoPipelineForImage2Image,
    )


# Refiner の安全なフォールバック：
#  - まずネイティブの StableDiffusionXLRefinerPipeline を試す
#  - ダメなら DiffusionPipeline を使う（呼び出しは同じ）
try:
    from diffusers import StableDiffusionXLRefinerPipeline as _RefinerPipe
except Exception:
    _RefinerPipe = DiffusionPipeline  # フォールバック（ここではインスタンス化しない！）

# ------------------------------------------------------------
# 基本設定
# ------------------------------------------------------------
router = APIRouter(prefix="/backend/images", tags=["images"])

ROOT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = ROOT_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT_DIR / "data"   # data/<Username>/memory.json を想定
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SD15_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SDXL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
DEFAULT_TURBO = "stabilityai/sdxl-turbo"

_MODELS = [DEFAULT_SD15_ID, DEFAULT_SDXL_BASE, DEFAULT_SDXL_REFINER, DEFAULT_TURBO]

# キャッシュ（メモリ）
_SD15_CACHE: Dict[str, Any] = {}
_XL_SIMPLE_CACHE: Dict[str, Any] = {}
_SDXL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


@router.get("/get_user_data")
def get_user_data(username: str):
    # user_manager から読み出し（入れ子→フラットに揃える）
    from user_manager import load_user_memory
    mem = load_user_memory(username) or {}
    data = mem.get(username, mem) if isinstance(mem, dict) else {}

    return {
        "username": username,
        "points": int(data.get("points", 0)),
        "meters": data.get("meters", {}),
        "album": data.get("album", []),
        "gift_usage": data.get("gift_usage", {}),
    }


# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------
def _get_device() -> str:
    try:
        if (torch is not None) and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _save_image_to_generated(pil_image, prefix: str = "sd") -> str:
    ts = _now_stamp()
    filename = f"{prefix}-{ts}.png"
    out_path = GENERATED_DIR / filename
    pil_image.save(out_path)
    return f"/generated/{filename}"

# ---- user memory I/O（user_manager があれば使用、なければローカル直接） ----
def _load_user_memory(username: str) -> Dict[str, Any]:
    """
    user_manager の入れ子 {username:{...}} と、フラット {..} の両方に対応して返す（常にフラットで返す）
    """
    try:
        import importlib
        um = importlib.import_module("user_manager")
        if hasattr(um, "load_user_memory"):
            raw = um.load_user_memory(username) or {}
            # 入れ子 {username:{...}} → フラット {...} に変換
            if isinstance(raw, dict) and username in raw and isinstance(raw[username], dict):
                return dict(raw[username])
            return raw if isinstance(raw, dict) else {}
    except Exception:
        pass

    # フォールバック：ローカル直読み（ファイル名も user_manager に合わせる）
    udir = DATA_DIR / username
    udir.mkdir(parents=True, exist_ok=True)
    for fname in ("user_memory.json", "memory.json"):
        mfile = udir / fname
        if mfile.exists():
            try:
                raw = json.loads(mfile.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and username in raw and isinstance(raw[username], dict):
                    return dict(raw[username])
                return raw if isinstance(raw, dict) else {}
            except Exception:
                continue
    return {}

def _save_user_memory(username: str, mem: Dict[str, Any]) -> None:
    """
    渡されたフラット mem を user_manager の入れ子形式にマージ保存。
    フォールバック時は user_memory.json に保存する。
    """
    try:
        import importlib
        um = importlib.import_module("user_manager")
        if hasattr(um, "load_user_memory") and hasattr(um, "save_user_memory"):
            base = um.load_user_memory(username) or {}
            if not isinstance(base, dict):
                base = {}
            # 既存入れ子にマージ
            if username in base and isinstance(base[username], dict):
                base[username].update(mem)
            else:
                base[username] = dict(mem)
            um.save_user_memory(username, base)
            return
    except Exception:
        pass

    # フォールバック：ローカル直書き（ファイル名は user_memory.json に統一）
    udir = DATA_DIR / username
    udir.mkdir(parents=True, exist_ok=True)
    mfile = udir / "user_memory.json"
    base = {}
    if mfile.exists():
        try:
            base = json.loads(mfile.read_text(encoding="utf-8")) or {}
        except Exception:
            base = {}
    if username in base and isinstance(base[username], dict):
        base[username].update(mem)
    else:
        base[username] = dict(mem)
    mfile.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_album(username: str, url: str, tag: Optional[str]) -> None:
    if not username:
        return
    mem = _load_user_memory(username)
    album: List[Dict[str, Any]] = mem.get("album") or []
    album.insert(0, {"url": url, "tag": tag, "ts": _now_stamp()})
    # 最新20件に丸める
    album = album[:20]
    mem["album"] = album
    _save_user_memory(username, mem)


# ------------------------------------------------------------
# Pipeline ローダー
# ------------------------------------------------------------
def _load_sd15(model_id: str = DEFAULT_SD15_ID) -> StableDiffusionPipeline:
    if model_id in _SD15_CACHE:
        return _SD15_CACHE[model_id]

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    )

    # スケジューラは DPMSolver を使用（遅延 import で Free 環境でも安全）
    try:
        from diffusers import DPMSolverMultistepScheduler  # ← ここで遅延 import
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception:
        # diffusers 未導入/古い場合などは既定スケジューラのまま
        pass

    pipe.to(_get_device())
    _SD15_CACHE[model_id] = pipe
    return pipe
def _load_xl_simple(base_id: str = DEFAULT_SDXL_BASE):
    """SDXL Base（Refinerなし）を安全にロード。AutoPipelineで型ズレ事故を避ける。"""
    if base_id in _XL_SIMPLE_CACHE:
        return _XL_SIMPLE_CACHE[base_id]
    pipe = AutoPipelineForText2Image.from_pretrained(
        base_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    pipe.to(_get_device())
    _XL_SIMPLE_CACHE[base_id] = pipe
    return pipe

def _load_sdxl_pipes(base_id: str = DEFAULT_SDXL_BASE, refiner_id: str = DEFAULT_SDXL_REFINER):
    """
    SDXL Base + Refiner をペアで読み込む。
    Refiner は _RefinerPipe（StableDiffusionXLRefinerPipeline か DiffusionPipeline）で安全にロード。
    """
    key = (base_id, refiner_id)
    if key in _SDXL_CACHE:
        return _SDXL_CACHE[key]

    base = StableDiffusionXLPipeline.from_pretrained(
        base_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    ref = _RefinerPipe.from_pretrained(
        refiner_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )

    device = _get_device()
    base.to(device)
    ref.to(device)

    # ほんのり最適化（失敗は無視）
    try:
        base.unet.to(memory_format=torch.channels_last)
    except Exception:
        pass
    try:
        if hasattr(ref, "unet"):
            ref.unet.to(memory_format=torch.channels_last)
    except Exception:
        pass

    _SDXL_CACHE[key] = (base, ref)
    return base, ref


# ------------------------------------------------------------
# 生成ロジック
# ------------------------------------------------------------
def generate_sd15(
    *, prompt: str, negative: str, width: int, height: int, steps: int,
    guidance: float, seed: Optional[int]
):
    g = None
    if seed is not None:
        try:
            g = torch.Generator(device=_get_device()).manual_seed(int(seed))
        except Exception:
            g = None

    pipe = _load_sd15(DEFAULT_SD15_ID)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt, negative_prompt=negative,
            width=width, height=height,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=g,
        )
    return out.images[0]

def generate_photoreal_sdxl(
    *,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: Optional[int],
    base_id: str = DEFAULT_SDXL_BASE,
    refiner_id: str = DEFAULT_SDXL_REFINER,
    refiner_ratio: float = 0.0,
):
    """SDXL：Refiner無し or 二段（Base→Refiner）"""
    device = _get_device()
    g = None
    if seed is not None:
        try:
            g = torch.Generator(device=device).manual_seed(int(seed))
        except Exception:
            g = None

    # Refiner を使わない（プレビュー/高速）
    if not refiner_ratio or refiner_ratio <= 0.0:
        pipe = _load_xl_simple(base_id)
        with torch.inference_mode():
            img = pipe(
                prompt=prompt, negative_prompt=negative,
                width=width, height=height,
                num_inference_steps=steps, guidance_scale=guidance,
                generator=g,
            ).images[0]
        return img

    # Refiner を使う（仕上げ）
    base, ref = _load_sdxl_pipes(base_id, refiner_id)
    with torch.inference_mode():
        # 前段：Base で途中まで（latent で受け取る）
        base_out = base(
            prompt=prompt, negative_prompt=negative,
            width=width, height=height,
            num_inference_steps=steps, guidance_scale=guidance,
            denoising_end=float(refiner_ratio),
            output_type="latent",
            generator=g,
        )
        latents = base_out.images  # latent tensor

        # 後段：Refiner で仕上げ（latent を渡す）
        img = ref(
            prompt=prompt, negative_prompt=negative,
            image=latents,
            num_inference_steps=steps, guidance_scale=guidance,
            denoising_start=float(refiner_ratio),
            generator=g,
        ).images[0]
    return img


# ------------------------------------------------------------
# API
# ------------------------------------------------------------
@router.get("/health")
def health():
    return {
        "ok": True,
        "device": _get_device(),
        "models": _MODELS,
        "default": DEFAULT_SD15_ID,
    }

@router.get("/album")
def get_album(username: str, limit: int = 20):
    mem = _load_user_memory(username) if username else {}
    items = mem.get("album") or []
    return {"items": items[:max(1, min(limit, 50))]}

@router.post("/txt2img")
async def txt2img(payload: Dict[str, Any]):
    try:
        username = (payload.get("username") or "").strip()
        tag = (payload.get("tag") or "").strip() or None

        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return JSONResponse(status_code=400, content={"error": "PROMPT_REQUIRED"})

        negative = payload.get("negative_prompt") or ""
        width = int(payload.get("width") or 512)
        height = int(payload.get("height") or 512)
        steps = int(payload.get("steps") or 28)
        guidance = float(payload.get("guidance_scale") or 7.5)
        model_id = (payload.get("model_id") or DEFAULT_SD15_ID).strip()

        seed = payload.get("seed", None)
        try:
            seed = int(seed) if seed is not None else None
        except Exception:
            seed = None

        options = payload.get("options") or {}
        ref_ratio = float(options.get("refiner_denoising_end", 0.0))

        # --- 生成本体 ---
        if "xl" in model_id or model_id == DEFAULT_TURBO or model_id == DEFAULT_SDXL_BASE:
            # sdxl-turbo は Refiner なしが基本
            if "turbo" in model_id:
                pipe = _load_xl_simple(model_id)
                with torch.inference_mode():
                    image = pipe(
                        prompt=prompt, negative_prompt=negative,
                        width=width, height=height,
                        num_inference_steps=max(1, min(steps, 8)),   # turboは少ステップ
                        guidance_scale=max(0.8, min(guidance, 2.0)), # turboはCFG低め
                        generator=torch.Generator(device=_get_device()).manual_seed(seed) if seed is not None else None,
                    ).images[0]
            else:
                image = generate_photoreal_sdxl(
                    prompt=prompt, negative=negative,
                    width=width, height=height,
                    steps=steps, guidance=guidance, seed=seed,
                    base_id=DEFAULT_SDXL_BASE if model_id == DEFAULT_SDXL_BASE else model_id,
                    refiner_id=DEFAULT_SDXL_REFINER,
                    refiner_ratio=ref_ratio,
                )
            used_model = model_id
        else:
            # SD1.5 ルート
            image = generate_sd15(
                prompt=prompt, negative=negative,
                width=width, height=height,
                steps=steps, guidance=guidance, seed=seed,
            )
            used_model = DEFAULT_SD15_ID if model_id == "" else model_id

        url = _save_image_to_generated(image, prefix="sd")
        if username:
            _append_album(username, url, tag)

        return {
            "url": url,
            "meta": {
                "model_id": used_model,
                "width": width, "height": height,
                "steps": steps, "guidance_scale": guidance,
                "seed": seed, "username": username,
                "saved_to_album": bool(username),
            }
        }

    except Exception as e:
        # ここに来たら生成時の例外
        return JSONResponse(
            status_code=500,
            content={"error": "GENERATION_FAILED", "detail": str(e)},
        )
