#!/usr/bin/env python3
"""Bind an existing project-issued key to this device; never create accounts."""

import argparse
import logging
from pathlib import Path

import requests

from motion_capture import Config, get_hardware_id


def onboard(config, issuer_email, session=None):
    key = Path(config.access_key_file).read_text(encoding="utf-8").strip()
    if not key or not issuer_email.strip():
        raise ValueError("A nonempty access key and its issuer email are required")
    unit_id = get_hardware_id(config.unique_id_path)
    response = (session or requests).post(
        f"{config.server_base_url.rstrip('/')}/onboarding",
        headers={"X-API-Key": key, "X-Email": issuer_email.strip()},
        json={"unit_id": unit_id}, timeout=config.upload_timeout,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("unit_id") != unit_id:
        raise ValueError("Onboarding returned an unexpected device identity")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    parser.add_argument("--server-base-url", default=defaults.server_base_url)
    parser.add_argument("--access-key-file", default=defaults.access_key_file)
    parser.add_argument("--unique-id-path", default=defaults.unique_id_path)
    parser.add_argument("--issuer-email", required=True)
    args = parser.parse_args()
    config = Config(server_base_url=args.server_base_url, access_key_file=args.access_key_file,
                    unique_id_path=args.unique_id_path)
    try:
        result = onboard(config, args.issuer_email)
    except (requests.RequestException, OSError, ValueError) as exc:
        logging.error("Onboarding failed: %s", exc)
        raise SystemExit(1) from exc
    print(f"Onboarded unit {result['unit_id']}; project={result['project_id']} "
          f"device={result['device_id']} idempotent={result.get('idempotent', False)}")


if __name__ == "__main__":
    main()
