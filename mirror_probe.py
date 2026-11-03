#!/usr/bin/env python3
"""Read-only mirror throughput probe.

Downloads a bounded byte range of a real wheel from each candidate PyPI mirror
and reports measured throughput. Writes nothing outside memory.
"""

import re
import sys
import time
import urllib.parse
import urllib.request

MIRRORS = [
    ("tuna", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("aliyun", "https://mirrors.aliyun.com/pypi/simple"),
    ("ustc", "https://pypi.mirrors.ustc.edu.cn/simple"),
    ("tencent", "https://mirrors.cloud.tencent.com/pypi/simple"),
]

PKG = "numpy"
PROBE_BYTES = 8 * 1024 * 1024  # 8 MiB
TIMEOUT_S = 12


def probe(name: str, base: str):
    page = f"{base}/{PKG}/"
    try:
        req = urllib.request.Request(page, headers={"User-Agent": "probe/1.0"})
        html = urllib.request.urlopen(req, timeout=TIMEOUT_S).read().decode("utf-8", "replace")
    except Exception as e:
        return name, None, f"index error: {type(e).__name__}"

    hrefs = re.findall(r'href="([^"#]+\.whl)(?:#[^"]*)?"', html)
    if not hrefs:
        return name, None, "no wheel link in index"

    url = urllib.parse.urljoin(page, hrefs[-1])
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "probe/1.0", "Range": f"bytes=0-{PROBE_BYTES - 1}"},
        )
        start = time.monotonic()
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_S)
        read = 0
        while read < PROBE_BYTES:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            read += len(chunk)
        elapsed = time.monotonic() - start
        if elapsed <= 0 or read == 0:
            return name, None, "no bytes read"
        return name, read / elapsed, f"{read / 1048576:.1f} MiB in {elapsed:.1f}s"
    except Exception as e:
        return name, None, f"download error: {type(e).__name__}"


def main() -> int:
    results = []
    for name, base in MIRRORS:
        name, speed, note = probe(name, base)
        results.append((name, speed, note))
        if speed:
            print(f"{name:8s} {speed / 1048576:6.2f} MiB/s   ({note})", flush=True)
        else:
            print(f"{name:8s}    n/a          ({note})", flush=True)

    usable = [r for r in results if r[1]]
    if not usable:
        print("\nRESULT: no mirror reachable")
        return 1
    usable.sort(key=lambda r: r[1], reverse=True)
    best = usable[0]
    print(f"\nBEST: {best[0]}  {best[1] / 1048576:.2f} MiB/s")
    for name, base in MIRRORS:
        if name == best[0]:
            print(f"URL : {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
