"""HTTP inference API.

    uvicorn pdscreen.api:app --app-dir src --port 8000

POST /predict with a multipart ``file`` (WAV) returns the screening result as
JSON; GET /health reports whether the models are loaded.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import config
from .audio import AudioError

MAX_BYTES = 10 * 1024 * 1024
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    from .inference import Screener

    try:
        screener = Screener()
        screener.warm_up()
        state["screener"] = screener
    except FileNotFoundError as error:
        state["error"] = str(error)
    yield
    state.clear()


app = FastAPI(title="Parkinson's voice screening API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    screener = state.get("screener")
    return {"ready": screener is not None, "model": screener.deployment["name"] if screener else None,
            "error": state.get("error"), "disclaimer": config.DISCLAIMER}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    screener = state.get("screener")
    if screener is None:
        raise HTTPException(status_code=503, detail=state.get("error", "Models are not loaded."))
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Audio larger than 10 MB; send 5-10 seconds of audio.")
    try:
        result, _ = screener.predict_bytes(contents)
    except AudioError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return result.as_dict()
