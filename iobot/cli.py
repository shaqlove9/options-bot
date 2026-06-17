"""cli.py — offline operator commands.

  python -m iobot.cli train   # retrain + walk-forward validate the meta model
  python -m iobot.cli gate     # print the validation-gate verdict
  python -m iobot.cli status   # print the latest engine status JSON
  python -m iobot.cli run       # run the trading engine (same as -m iobot.engine)
"""
from __future__ import annotations

import json
import sys

from iobot import config, gate, meta, store


def main(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "status"

    if cmd == "train":
        print(json.dumps(meta.train(store.connect()), indent=2))
    elif cmd == "gate":
        print(json.dumps(gate.evaluate(store.connect()), indent=2, default=str))
    elif cmd == "status":
        import os
        if os.path.exists(config.STATUS_FILE):
            with open(config.STATUS_FILE) as f:
                print(f.read())
        else:
            print("no status file yet")
    elif cmd == "run":
        from iobot.engine import main as run_engine
        run_engine()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
