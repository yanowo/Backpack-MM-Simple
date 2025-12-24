import requests
from typing import Optional


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.last_error: Optional[str] = None

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        self.last_error = None
        if not text:
            self.last_error = "empty message"
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
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            return False
        if not bool(data.get("ok")):
            self.last_error = str(data)
            return False
        return True
