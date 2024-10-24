import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from splatnlp import VERSION
from splatnlp.serve.inference import inference
from splatnlp.serve.load_model import load_from_env
from splatnlp.serve.tokenize import tokenize_build

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vocab, weapon_vocab, pad_token_id, model_info, model, inv_vocab
    vocab, weapon_vocab, pad_token_id, model_info, model = load_from_env()
    logger.info(f"Model info: {model_info}")
    model_info.pop("param_num")
    inv_vocab = {v: k for k, v in vocab.items()}
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceRequest(BaseModel):
    target: dict[str, int]
    weapon_id: int


class InferenceResponse(BaseModel):
    predictions: list[tuple[str, float]]
    splatgpt_info: dict
    api_version: str = "0.1.0"
    inference_time: float


@app.post("/infer")
def infer(request: InferenceRequest) -> InferenceResponse:
    logger.info(f"Received inference request: {request}")
    target = tokenize_build(request.target)
    weapon_id_str = f"weapon_id_{request.weapon_id}"
    predictions, time_taken = inference(
        model=model,
        target=target,
        weapon_id=weapon_id_str,
        vocab=vocab,
        inv_vocab=inv_vocab,
        weapon_vocab=weapon_vocab,
        pad_token_id=pad_token_id,
    )

    return InferenceResponse(
        predictions=predictions,
        splatgpt_info=model_info,
        api_version=VERSION,
        inference_time=time_taken,
    )


@app.get("/infer")
def infer_test() -> InferenceResponse:
    request = InferenceRequest(
        target={"ink_saver_main": 3, "ink_resistance_up": 3}, weapon_id=1
    )
    return infer(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
