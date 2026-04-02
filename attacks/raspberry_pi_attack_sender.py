"""Simple Raspberry Pi client for sending attack payloads to the dashboard.

Usage examples:
python raspberry_pi_attack_sender.py --server http://192.168.1.50:5000 --slug jw02736006001_02101_00001_nrs1_cal --attack-type hotspot --x 820 --y 410 --radius 25 --intensity 4000
python raspberry_pi_attack_sender.py --server http://192.168.1.50:5000 --slug jw02736006001_02101_00001_nrs1_cal --attack-type stripe_noise --axis vertical --start 1200 --width 12 --intensity 1600
"""

from __future__ import annotations

import argparse
import json
import sys

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send an attack payload to the astrophysics dashboard.")
    parser.add_argument("--server", required=True, help="Base dashboard URL, e.g. http://192.168.1.50:5000")
    parser.add_argument("--slug", required=True, help="Dataset slug to attack")
    parser.add_argument("--attack-type", required=True, choices=["hotspot", "stripe_noise", "block_dropout", "salt_pepper"])
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--intensity", type=float)
    parser.add_argument("--axis", choices=["horizontal", "vertical"])
    parser.add_argument("--start", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fill-value", type=float)
    parser.add_argument("--amount", type=float)
    parser.add_argument("--seed", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = {
        "slug": args.slug,
        "attack_type": args.attack_type,
    }

    optional_fields = {
        "x": args.x,
        "y": args.y,
        "radius": args.radius,
        "intensity": args.intensity,
        "axis": args.axis,
        "start": args.start,
        "width": args.width,
        "height": args.height,
        "fill_value": args.fill_value,
        "amount": args.amount,
        "seed": args.seed,
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value

    endpoint = args.server.rstrip("/") + "/api/attack_ingest"
    response = requests.post(endpoint, json=payload, timeout=20)
    print(json.dumps(response.json(), indent=2))
    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
