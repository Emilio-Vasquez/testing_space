"""Raspberry Pi client for sending an attack payload and waiting for defender disconnect.

Usage examples:
python raspberry_pi_attack_sender.py --server http://192.168.1.50:5000 --slug jw02736006001_02101_00001_nrs1_cal --attack-type hotspot --x 820 --y 410 --radius 25 --intensity 4000
python raspberry_pi_attack_sender.py --server http://192.168.1.50:5000 --slug jw02736006001_02101_00001_nrs1_cal --attack-type stripe_noise --axis vertical --start 1200 --width 12 --intensity 1600
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send an attack payload to the astrophysics dashboard.")
    parser.add_argument("--server", required=True, help="Base dashboard URL, e.g. http://192.168.1.50:5000")
    parser.add_argument("--slug", required=True, help="Dataset slug to attack")
    parser.add_argument("--attack-type", required=True, choices=["hotspot", "stripe_noise", "block_dropout", "salt_pepper"])
    parser.add_argument("--client-name", default="raspberry-pi", help="Human-readable client name for the session")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between session status checks after attack is sent")
    parser.add_argument("--no-wait", action="store_true", help="Send the attack and exit immediately instead of waiting for disconnect")
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


def post_json(url: str, payload: dict, timeout: float = 20.0) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def get_json(url: str, timeout: float = 20.0) -> requests.Response:
    return requests.get(url, timeout=timeout)


def main() -> int:
    args = build_parser().parse_args()
    base_url = args.server.rstrip("/")

    print(f"[INFO] Connecting to monitoring server at {base_url}")

    start_payload = {
        "slug": args.slug,
        "client_name": args.client_name,
    }
    start_response = post_json(base_url + "/api/session_start", start_payload)
    try:
        start_data = start_response.json()
    except Exception:
        print("[ERROR] Failed to parse session_start response.")
        return 1

    if not start_response.ok:
        print(json.dumps(start_data, indent=2))
        return 1

    session_id = str(start_data["session_id"])
    print(f"[INFO] Session started: {session_id}")

    payload = {
        "slug": args.slug,
        "attack_type": args.attack_type,
        "session_id": session_id,
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

    attack_response = post_json(base_url + "/api/attack_ingest", payload)
    try:
        attack_data = attack_response.json()
    except Exception:
        print("[ERROR] Failed to parse attack response.")
        return 1

    print(json.dumps(attack_data, indent=2))
    if not attack_response.ok:
        return 1

    if args.no_wait:
        print("[INFO] Attack sent. Exiting without waiting for disconnect.")
        return 0

    print("[INFO] Attack sent successfully. Waiting for defender response...")
    status_url = base_url + f"/api/session_status/{session_id}"

    while True:
        try:
            status_response = get_json(status_url, timeout=20.0)
            status_data = status_response.json()
        except KeyboardInterrupt:
            print("\n[INFO] Sender interrupted by user.")
            return 0
        except Exception as exc:
            print(f"[WARN] Session status check failed: {exc}")
            time.sleep(max(args.poll_interval, 0.5))
            continue

        if not status_response.ok:
            print(json.dumps(status_data, indent=2))
            return 1

        if not status_data.get("active", False):
            reason = status_data.get("disconnect_reason") or "ended"
            if reason == "disconnected_by_defender":
                print("[ALERT] Defender disconnected this Raspberry Pi session.")
                print("[INFO] Session terminated. Exiting sender.")
            else:
                print(f"[INFO] Session no longer active. Reason: {reason}")
            return 0

        print("[INFO] Awaiting session status...")
        time.sleep(max(args.poll_interval, 0.5))


if __name__ == "__main__":
    raise SystemExit(main())
