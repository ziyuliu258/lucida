#!/usr/bin/env python3
"""Range-extract one scene from an official FoundationPose Google Drive ZIP."""
from __future__ import annotations

import argparse
import contextlib
import importlib
from pathlib import Path

from remotezip import RemoteZip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", default="1nQSkWFVWt1XKBML-jm0bqdoeQXm0ETsT")
    parser.add_argument("--object-archive", default="1202363524")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", help="Optional HTTP/HTTPS proxy URL.")
    args = parser.parse_args()

    module = importlib.import_module("gdown.download")
    session, cookies_file = module._get_session(
        proxy=args.proxy, use_cookies=True, user_agent="Mozilla/5.0", cookies_file=None
    )
    session.headers["Range"] = "bytes=-65536"
    session.headers["Accept-Encoding"] = "identity"
    with contextlib.ExitStack() as responses:
        response, _ = module._get_download_response(
            sess=session,
            responses=responses,
            url=f"https://drive.google.com/uc?id={args.file_id}",
            gdrive_file_id=args.file_id,
            format=None,
            verify=True,
            timeout=60,
            retry=module._RetryState(retries=3, quiet=True, cancel=None),
            use_cookies=True,
            cookies_file=cookies_file,
        )
        direct_url = response.url
    session.headers.pop("Range", None)

    prefix = f"{args.object_archive}/scene_{args.scene:08d}/"
    with session, RemoteZip(direct_url, session=session, initial_buffer_size=65536) as archive:
        names = archive.namelist()
        selected = [
            name for name in names
            if name.startswith(prefix)
            and (
                name.endswith("states.json")
                or name.endswith("scene.usd")
                or "/RenderProduct_Replicator/" in name
            )
        ]
        if not selected:
            raise ValueError(f"scene prefix absent from archive: {prefix}")
        for name in selected:
            if name.endswith("/"):
                continue
            destination = args.output / Path(name).relative_to(prefix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
            print(destination)


if __name__ == "__main__":
    main()
