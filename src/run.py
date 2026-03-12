import os

import uvicorn


def _resolve_port() -> int:
    raw_port = os.getenv("PORT", "8000")
    try:
        return int(raw_port)
    except ValueError:
        return 8000


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=_resolve_port())