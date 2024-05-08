Fork of ctranspath, packaged as an embedding service

Build and run container
`docker build -t ctranspath . && docker run -p 8000:8000 ctranspath`

or pull pre-build
`docker run -p 8000:8000 gabrieldernbach/histo:cstranspath`

You can now talk to ctranspath, e.g.
```python3
import io
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image
from skimage.data import immunohistochemistry


def img2byte(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def embedding(img):
    response = requests.post(
        url='http://localhost:8000/embed_image',
        data=img2byte(img)
    )
    if response.status_code == 200:
        return np.load(io.BytesIO(response.content))
    else:
        raise Exception(f"Failed to get embedding: {response.text}")


def threadmap(fun, xs):
    with ThreadPoolExecutor() as pool:
        return pool.map(fun, xs)


image = Image.fromarray(immunohistochemistry()).resize((224, 224))
embedding(image)
threadmap(embedding, [image] * 16))
```
