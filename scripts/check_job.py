"""Check the status of a Runpod Flash training job."""

from __future__ import annotations

import argparse
import base64
import json
import sys

import cloudpickle
import requests
from runpod_flash.core.credentials import get_api_key


def main():
    parser = argparse.ArgumentParser(description="Check a Runpod job status.")
    parser.add_argument("endpoint_id", help="Runpod endpoint ID")
    parser.add_argument("job_id", help="Runpod job ID")
    parser.add_argument("--raw", action="store_true", help="Print raw API response")
    args = parser.parse_args()

    api_key = get_api_key()
    url = f"https://api.runpod.ai/v2/{args.endpoint_id}/status/{args.job_id}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    status = data.get("status", "UNKNOWN")
    print(f"Status: {status}")

    if args.raw:
        print(json.dumps(data, indent=2, default=str))
        return

    if data.get("delayTime"):
        print(f"Queue delay: {data['delayTime']}ms")
    if data.get("executionTime"):
        exec_s = data["executionTime"] / 1000
        print(f"Execution time: {exec_s:.0f}s ({exec_s/3600:.1f}h)")

    if status in ("IN_QUEUE", "IN_PROGRESS"):
        print("Job is still running. Check back later.")
        return

    if status == "COMPLETED":
        output = data.get("output", {})
        if isinstance(output, dict) and output.get("result"):
            result = cloudpickle.loads(base64.b64decode(output["result"]))
            print(json.dumps(result, indent=2, default=str))
        elif isinstance(output, dict) and output.get("json_result"):
            print(json.dumps(output["json_result"], indent=2, default=str))
        else:
            print("Output:", json.dumps(output, indent=2, default=str))
    else:
        print(f"Error: {data.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
