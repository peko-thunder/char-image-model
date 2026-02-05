from datetime import datetime
from pathlib import Path


def output_logs(logs: list[str]) -> None:
    if logs:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        log_file.write_text("\n".join(logs), encoding="utf-8")
