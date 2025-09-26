# chat_transfer_kit_v1

移行用の最小パッケージです。**履歴**・**プロフィール(user_data)**・**UI返却**・**メッセージ構築**・**応答生成のインターフェース**を統一した形で収めています。

## 構成
```
chat_transfer_kit_v1/
  README.md
  requirements.txt
  run_demo.py
  chatkit/
    __init__.py
    schema.py
    storage.py
    messages.py
    generator.py
    ui_bridge.py
```
- `schema.py` : `user_data` の型・初期値・検証
- `storage.py`: 履歴・プロフィールの保存/読み込み（JSON）
- `messages.py`: `build_messages()` と質問文制御
- `generator.py`: `generate_response(messages)` のバックエンド切替（Echo/Rule/Custom）
- `ui_bridge.py`: UI側から呼ぶ統一関数 `some_function_name()` 実装（=ご提示の修正版）
- `run_demo.py`: コンソールで試す最小デモ

## 使い方
1. 依存を入れる（依存はデフォルトでゼロ。任意で `pydantic` を使う場合のみ）  
   ```bash
   pip install -r requirements.txt
   ```

2. コンソールデモを実行
   ```bash
   python run_demo.py
   ```

3. 既存UIに統合  
   - 既存の UI から `from chatkit.ui_bridge import some_function_name` を import  
   - `user_data` は `schema.default_user_data()` の初期値をベースに必要に応じて更新  
   - 履歴は `storage` で保存/読み込み可能

## 差し替えポイント
- 実際のLLM接続は `generator.py` の `CustomBackend` を差し替え（OpenAI/自前モデル/HTTPなど）
- 複数疑問文の制御は `messages.py` の `clamp_question_marks()` を調整

---

このまま別チャットに貼り付け/持ち運び可能です。
