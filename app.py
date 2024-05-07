import asyncio
import io
from collections import deque

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

from model import load_model_and_preprocessor

app = FastAPI()
model, preprocessor = load_model_and_preprocessor()

# Batch parameters
batch_size = 32
batch_timeout = 1.0  # Timeout in seconds

# Shared state using deque for efficient append and pop operations
current_batch = deque()
batch_lock = asyncio.Lock()  # Lock to manage access to the current_batch


async def process_images(images):
    try:
        image_tensors = torch.stack([
            preprocessor(Image.open(io.BytesIO(img))) for img in images
        ])
        with torch.no_grad():
            return model(image_tensors).cpu().numpy().tolist()
    except Exception as e:
        raise RuntimeError(f"Error processing images: {str(e)}")


async def process_batch(batch):
    images, futures = zip(*batch)
    try:
        embeddings = await process_images(list(images))
        for future, embedding in zip(futures, embeddings):
            future.set_result(embedding)
    except Exception as e:
        for future in futures:
            future.set_exception(e)


async def batch_collector():
    while True:
        await asyncio.sleep(batch_timeout)
        batch_to_process = None
        async with batch_lock:
            if current_batch:
                batch_to_process = list(current_batch)
                current_batch.clear()
        if batch_to_process:
            asyncio.create_task(process_batch(batch_to_process))


@app.on_event("startup")
async def start_batch_collector():
    asyncio.create_task(batch_collector())


@app.post("/embed_image/")
async def embed_image(image: UploadFile = File(...)):
    img_bytes = await image.read()
    result = asyncio.get_event_loop().create_future()
    async with batch_lock:
        current_batch.append((img_bytes, result))
        if len(current_batch) >= batch_size:
            batch_to_process = list(current_batch)
            current_batch.clear()
            asyncio.create_task(process_batch(batch_to_process))
    try:
        return await result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
