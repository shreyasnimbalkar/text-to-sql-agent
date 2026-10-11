"""Download the BIRD dev databases and the Mini-Dev question set into data/bird/.

Usage:  .venv\\Scripts\\python scripts\\download_bird.py

  ~350 MB download, 1.4 GB unpacked. Creates:
    data/bird/dev_databases/<db>/<db>.sqlite   11 real databases
    data/bird/mini_dev_sqlite.json             500 test questions (what the README reports)
    data/bird/dev.json                         1,534 full dev questions
    data/bird/tune.json                        dev questions NOT in Mini-Dev, used for prompt tuning,
                                               so the test questions are never used for tuning
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEV_ZIP = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
MINI_DEV = "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/mini_dev_sqlite-00000-of-00001.json"
OUT = Path(__file__).resolve().parent.parent / "data" / "bird"


def fetch(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "dev_databases").exists():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fetch(DEV_ZIP, tmp / "dev.zip")
            with zipfile.ZipFile(tmp / "dev.zip") as z:
                z.extractall(tmp)
            inner = next(tmp.glob("dev_*/dev_databases.zip"))
            print("unpacking databases (1.4 GB)...")
            with zipfile.ZipFile(inner) as z:
                z.extractall(OUT)
            for name in ("dev.json", "dev_tables.json"):
                shutil.copy(inner.parent / name, OUT / name)
        shutil.rmtree(OUT / "__MACOSX", ignore_errors=True)
        for junk in OUT.rglob(".DS_Store"):
            junk.unlink()
    if not (OUT / "mini_dev_sqlite.json").exists():
        fetch(MINI_DEV, OUT / "mini_dev_sqlite.json")

    dev = json.loads((OUT / "dev.json").read_text(encoding="utf-8"))
    mini = json.loads((OUT / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
    test_questions = {(q["db_id"], q["question"].strip()) for q in mini}
    tune = [q for q in dev if (q["db_id"], q["question"].strip()) not in test_questions]
    (OUT / "tune.json").write_text(json.dumps(tune, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done: {len(mini)} test questions, {len(tune)} tuning questions, "
          f"{len(list((OUT / 'dev_databases').iterdir()))} databases in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
