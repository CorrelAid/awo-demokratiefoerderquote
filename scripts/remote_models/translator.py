import modal
from dotenv import load_dotenv
import os
from lib.data_augmentation import (
    ALL_PIVOTS,
    load_and_quantize,
    back_translation_augmenter_single_thread,
)
from lib.experiment_config import TRANSLATOR_APP_NAME, TRANSLATOR_FUNCTION_NAME
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


load_dotenv()

API_KEY = os.getenv("API_KEY")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers>=4.54.0",
        "rich",
        "torch",
        "fastapi",
        "python-dotenv",
        "sentencepiece",
        "nlpaug",
        "huggingface_hub[hf_transfer]",
        "sacremoses",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("lib")
)

CACHE_PATH = "/root/model_cache"

auth_scheme = HTTPBearer()
app = modal.App(TRANSLATOR_APP_NAME)

N_GPU = 1
MINUTES = 60


@app.cls(
    secrets=[modal.Secret.from_dotenv(__file__)],
    image=image,
    gpu=f"L4:{N_GPU}",
    timeout=20 * MINUTES,
    max_containers=40,
)
@modal.concurrent(max_inputs=100)
class Model:
    @modal.enter()
    def load_model(self):
        try:
            quant_pipes = {p: load_and_quantize(p, device="cuda") for p in ALL_PIVOTS}
            self.quant_pipes = quant_pipes
            print("Quantized pipelines loaded successfully")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load and quantize pipelines: {str(e)}",
            )

    @modal.fastapi_endpoint(method="POST", docs=True, label=TRANSLATOR_FUNCTION_NAME)
    def serve(
        self, request: dict, token: HTTPAuthorizationCredentials = Depends(auth_scheme)
    ):
        if token.credentials != os.getenv("API_KEY"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            texts = request["texts"]
            labels = request["labels"]
            factor = request["factor"]
            batch_size = request["batch_size"]
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request format: Missing field '{str(e)}'",
            )

        try:
            print("Starting back-translation augmentation (single-threaded)...")

            augmented_texts, augmented_labels = (
                back_translation_augmenter_single_thread(
                    self.quant_pipes,
                    texts,
                    labels,
                    factor=factor,
                    batch_size=batch_size,
                )
            )

            return {
                "augmented_texts": augmented_texts,
                "augmented_labels": augmented_labels,
            }

        except Exception as e:
            print(f"Error during back-translation augmentation:\n{str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during back-translation augmentation: {str(e)}",
            )
