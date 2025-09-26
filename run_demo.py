from chatkit.schema import default_user_data
from chatkit.storage import save_history, load_history, save_user_data, load_user_data
from chatkit.ui_bridge import some_function_name

HIST_PATH = "demo_data/history.json"
USER_PATH = "demo_data/user_data.json"

def main():
    import os
    os.makedirs("demo_data", exist_ok=True)

    # 初期ロード
    history = load_history(HIST_PATH)
    user_data = load_user_data(USER_PATH) or default_user_data()

    print("== chat_transfer_kit デモ ==")
    print("空行で終了。")
    while True:
        user_text = input("\nあなた> ").strip()
        if not user_text:
            break
        history, bot, profile = some_function_name(history, user_text, bot="", user_data=user_data)
        print(f"Aちゃん> {bot}")
        save_history(HIST_PATH, history)
        save_user_data(USER_PATH, user_data)

if __name__ == "__main__":
    main()
