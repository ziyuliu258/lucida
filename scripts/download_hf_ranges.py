#!/usr/bin/env python3
"""Download one gated Hugging Face file with parallel HTTP byte ranges."""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import get_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("filename")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--progress-percent", type=int, default=5)
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise RuntimeError("no Hugging Face token is configured")
    headers = {"Authorization": f"Bearer {token}"}
    resolve = f"https://huggingface.co/{args.repo}/resolve/main/{args.filename}"
    probe = None
    for attempt in range(6):
        try:
            probe = requests.get(resolve, headers={**headers, "Range": "bytes=0-0"}, stream=True, timeout=60)
            probe.raise_for_status()
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    assert probe is not None
    total = int(probe.headers.get("content-range", "/0").split("/")[-1])
    if not total:
        total = int(probe.headers["content-length"])
    final_url = probe.url
    probe.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_CREAT | os.O_RDWR, 0o664)
    if os.fstat(fd).st_size != total:
        os.ftruncate(fd, 0)
        os.ftruncate(fd, total)
    chunk = args.chunk_mib * 1024 * 1024
    all_ranges = [(start, min(start + chunk, total) - 1) for start in range(0, total, chunk)]

    def range_present(item: tuple[int, int]) -> bool:
        """Treat a range as complete only when the sparse file fully backs it."""
        start, end = item
        try:
            data_start = os.lseek(fd, start, os.SEEK_DATA)
            if data_start != start:
                return False
            return os.lseek(fd, start, os.SEEK_HOLE) >= end + 1
        except OSError:
            return False

    completed = [item for item in all_ranges if range_present(item)]
    ranges = [item for item in all_ranges if not range_present(item)]
    if completed:
        print(f"resuming: {len(completed)}/{len(all_ranges)} ranges already present", flush=True)

    def fetch(item: tuple[int, int]) -> int:
        start, end = item
        for attempt in range(6):
            try:
                response = requests.get(final_url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                response.raise_for_status()
                data = response.content
                if len(data) != end - start + 1:
                    raise IOError(f"short range {start}-{end}: {len(data)} bytes")
                os.pwrite(fd, data, start)
                return len(data)
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        return 0

    done = sum(end - start + 1 for start, end in completed)
    next_report = args.progress_percent
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, item) for item in ranges]
        for future in as_completed(futures):
            done += future.result()
            percent = 100.0 * done / total
            if percent >= next_report or done == total:
                print(f"{args.filename}: {percent:.1f}% ({done}/{total})", flush=True)
                while next_report <= percent:
                    next_report += args.progress_percent
    os.fsync(fd)
    os.close(fd)
    print(args.output)


if __name__ == "__main__":
    main()
