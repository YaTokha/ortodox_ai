#!/usr/bin/env python3
"""Снимок цен Vast.ai по выбранной GPU-модели (без токена, по публичному endpoint)."""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request


def fetch_offers(gpu_name: str, limit: int = 500):
    payload = {
        "limit": limit,
        "gpu_name": {"eq": gpu_name},
        "num_gpus": {"eq": 1},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "type": "ondemand",
    }
    req = urllib.request.Request(
        "https://console.vast.ai/api/v0/bundles/",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8")).get("offers", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="RTX 4090")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--min-reliability", type=float, default=0.98)
    args = parser.parse_args()

    offers = fetch_offers(args.gpu, args.limit)
    prices = sorted(o["dph_total"] for o in offers if isinstance(o.get("dph_total"), (int, float)))

    print(f"GPU: {args.gpu}")
    print(f"Offers found: {len(prices)}")
    if not prices:
        return

    print(f"min: {prices[0]:.6f} $/h")
    print(f"p50: {statistics.median(prices):.6f} $/h")
    print(f"p90: {prices[int(len(prices) * 0.9)]:.6f} $/h")
    print(f"max: {prices[-1]:.6f} $/h")

    filt = [
        o
        for o in offers
        if o.get("verification") == "verified"
        and (o.get("reliability2") or 0) >= args.min_reliability
        and isinstance(o.get("dph_total"), (int, float))
    ]
    f_prices = sorted(o["dph_total"] for o in filt)

    print(f"Verified + reliability>={args.min_reliability}: {len(f_prices)} offers")
    if f_prices:
        print(f"min (verified): {f_prices[0]:.6f} $/h")
        print(f"p50 (verified): {statistics.median(f_prices):.6f} $/h")


if __name__ == "__main__":
    main()
