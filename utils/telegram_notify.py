import requests


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not text:
            return False
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json=payload,
                timeout=10,
            )
            data = response.json()
        except Exception:
            return False
        return bool(data.get("ok"))
