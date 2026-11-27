"""Parallel, self-retrying wrapper around benchmark.fetch_v2.download for the 14 ETH3D indoor archives.

The ETH3D server gives ~0.1-0.5 MB/s per connection, so one thread per file. Same resumable,
size-checked .part logic as fetch_v2 (a killed run resumes where it stopped). A 60 s socket timeout
turns a dead connection into an error; failed files are retried every --wait seconds (network
outages) until all are complete or --hours elapse. Prints [ETH3D_DL_DONE] only when every file is
complete, [ETH3D_DL_INCOMPLETE] only when it gives up.

  python -m scripts.fetch_eth3d_parallel [--workers 14] [--hours 12]
"""
from __future__ import annotations

import argparse
import socket
import time
from concurrent.futures import ThreadPoolExecutor

from benchmark import fetch_v2


def _one(rf, root):
    try:
        return fetch_v2.download(rf, root)
    except Exception as ex:                      # network errors -> retried next round
        print(f"[retry-later] {rf.rel}: {type(ex).__name__}: {ex}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--wait", type=int, default=120)
    a = ap.parse_args()
    socket.setdefaulttimeout(60)
    root = fetch_v2.resolve_root(None)
    todo = list(fetch_v2.SETS["eth3d_indoor"]["files"])
    t_end = time.time() + a.hours * 3600
    rnd = 0
    while todo and time.time() < t_end:
        rnd += 1
        with ThreadPoolExecutor(a.workers) as ex:
            ok = list(ex.map(lambda rf: _one(rf, root), todo))
        todo = [rf for rf, k in zip(todo, ok) if not k]
        print(f"round {rnd}: {len(todo)} file(s) left", flush=True)
        if todo:
            time.sleep(a.wait)
    print("[ETH3D_DL_DONE]" if not todo else "[ETH3D_DL_INCOMPLETE]", flush=True)


if __name__ == "__main__":
    main()
