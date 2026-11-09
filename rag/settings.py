"""Settings for code that runs on the host: real environment first, then .env."""
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent


def _dotenv():
    path = ROOT / ".env"
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    pairs = (line.split("=", 1) for line in lines if "=" in line and not line.startswith("#"))
    return {key.strip(): value.strip() for key, value in pairs}


def get(name, default=""):
    return os.environ.get(name) or _dotenv().get(name) or default


def database_url():
    if get("DATABASE_URL"):
        return get("DATABASE_URL")
    password = quote(get("POSTGRES_PASSWORD"), safe="")
    return f"postgresql://techpulse:{password}@127.0.0.1:5433/techpulse"
