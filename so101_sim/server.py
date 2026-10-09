"""Local simulated robot API for typed commands and ElevenLabs tool wiring."""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .demo import SimulationController


class PickRequest(BaseModel):
    object: str
    destination: str = "tray"


def create_app(checkpoint: Path, stage: str = "robust") -> FastAPI:
    app = FastAPI(title="SO-101 simulated controller")
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco")
    controller = worker.submit(SimulationController, checkpoint, stage).result()
    job: Future | None = None

    @app.get("/list_objects")
    def list_objects():
        if job is not None and not job.done():
            raise HTTPException(409, "simulation is executing a command")
        return {"objects": worker.submit(controller.list_objects).result()}

    @app.post("/pick_and_place")
    def pick_and_place(request: PickRequest):
        nonlocal job
        if request.object not in ("box", "bottle", "cylinder") or request.destination != "tray":
            raise HTTPException(422, "supported objects: box, bottle, cylinder; destination: tray")
        if job is not None and not job.done():
            raise HTTPException(409, "robot is busy")
        job = worker.submit(controller.pick_and_place, request.object, request.destination)
        return {"state": "accepted", "object": request.object, "destination": request.destination}

    @app.get("/status")
    def status():
        result = controller.status()
        if job is not None and job.done() and job.exception() is not None:
            result["state"] = "failed"
            result["error"] = str(job.exception())
        return result

    @app.post("/stop")
    def stop():
        return controller.stop()

    @app.on_event("shutdown")
    def shutdown():
        if job and not job.done():
            controller.stop()
        worker.submit(controller.close).result(timeout=5)
        worker.shutdown(wait=False)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", default="robust")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.checkpoint, args.stage), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
