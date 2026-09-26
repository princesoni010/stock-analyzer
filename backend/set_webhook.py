import asyncio
import os
import httpx
from app.config import settings

async def set_webhook():
    token = settings.TELEGRAM_BOT_TOKEN
    webhook_url = "https://bharat-market-api.onrender.com/telegram/webhook"
    
    url = f"https://api.telegram.org/bot{token}/setWebhook"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={"url": webhook_url})
        print(resp.json())

if __name__ == "__main__":
    asyncio.run(set_webhook())
