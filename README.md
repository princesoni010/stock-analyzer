# Bharat Market AI

Personal AI-Powered Indian Stock-Market Research and Telegram Alert System.

## Setup

1. Setup virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/Scripts/activate
   pip install -r requirements.txt
   ```
2. Start dependencies (PostgreSQL, Redis):
   ```bash
   docker-compose up -d
   ```
3. Copy `.env.example` to `.env` and fill in API keys.
4. Run FastAPI app:
   ```bash
   cd backend/app
   uvicorn main:app --reload
   ```
