"""Inspect or cancel a Runpod Flash queue job by endpoint/job id."""

from __future__ import annotations

import argparse
import asyncio
import json

from runpod_flash import Endpoint
from runpod_flash.endpoint import EndpointJob


async def _status(endpoint_id: str, job_id: str) -> dict:
    endpoint = Endpoint(id=endpoint_id)
    job = EndpointJob({"id": job_id, "status": "UNKNOWN"}, endpoint)
    status = await job.status()
    return {
        "endpoint_id": endpoint_id,
        "job_id": job_id,
        "status": status,
        "done": job.done,
        "error": job.error,
        "output": job.output,
    }


async def _cancel(endpoint_id: str, job_id: str) -> dict:
    endpoint = Endpoint(id=endpoint_id)
    job = await endpoint.cancel(job_id)
    return {
        "endpoint_id": endpoint_id,
        "job_id": job.id,
        "status": job._data.get("status", "UNKNOWN"),
        "done": job.done,
        "error": job.error,
        "output": job.output,
    }


async def _main_async(args: argparse.Namespace):
    if args.action == "status":
        result = await _status(args.endpoint_id, args.job_id)
    else:
        result = await _cancel(args.endpoint_id, args.job_id)

    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Inspect or cancel a Runpod Flash queue job."
    )
    parser.add_argument("action", choices=["status", "cancel"], help="Action to perform")
    parser.add_argument("endpoint_id", help="Flash endpoint id")
    parser.add_argument("job_id", help="Queue job id")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()