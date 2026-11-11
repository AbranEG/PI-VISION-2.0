import requests

import os
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
resp = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates").json()
print(resp)
