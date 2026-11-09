"""Settings come only from the environment; nothing is defaulted on the user's behalf."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    database_url: str
    contact_email: str


def _required(env, name):
    value = env.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable {name}")
    return value


def load(env):
    return Config(
        database_url=_required(env, "DATABASE_URL"),
        contact_email=_required(env, "CONTACT_EMAIL"),
    )
