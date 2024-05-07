import base64
from io import BytesIO
from typing import List, Dict

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException

from model import load_model_and_preprocessor

app = FastAPI()
model, preprocessor = load_model_and_preprocessor()


def decode_image(image_data: str) -> Image:
    """Decode a base64 image string into an image object."""
    image_bytes = base64.b64decode(image_data)
    return Image.open(BytesIO(image_bytes))


def preprocess_and_embed(images_data: List[str]) -> List[List[float]]:
    """Preprocess a list of base64 image strings and return their embeddings."""
    image_tensors = torch.stack([preprocessor(decode_image(img_data)) for img_data in images_data])
    with torch.no_grad():
        return model(image_tensors).cpu().numpy().tolist()


@app.post('/embed_image')
async def embed_image(image_data: Dict[str, str]):
    if 'image' not in image_data:
        raise HTTPException(status_code=400, detail="Image data is missing")
    vector = preprocess_and_embed([image_data['image']])[0]
    return {"vector": vector}


@app.post('/embed_images')
async def embed_images(images_data: Dict[str, List[str]]):
    if 'images' not in images_data:
        raise HTTPException(status_code=400, detail="Images data is missing")
    vectors = preprocess_and_embed(images_data['images'])
    return {"vectors": vectors}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
