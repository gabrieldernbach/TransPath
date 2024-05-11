Fork of ctranspath, packaged as an embedding service

Build and run container
`docker build -t ctranspath . && docker run -p 8000:8000 ctranspath`

or pull pre-build
`docker run -p 8000:8000 gabrieldernbach/histo:cstranspath`

You can now talk to ctranspath, e.g.
```python3
import io
from tenacity import retry, stop_after_attempt, wait_random_exponential
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image
from skimage.data import immunohistochemistry


def img2byte(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()

@retry(stop=stop_after_attempt(20), wait=wait_random_exponential(multiplier=0.5, max=240))
def embedding(img):
    response = requests.post(
        url='http://localhost:8000/embed_image',
        data=img2byte(img)
    )
    return np.load(io.BytesIO(response.content))

def threadmap(fun, xs):
    with ThreadPoolExecutor() as pool:
        return pool.map(fun, xs)


image = Image.fromarray(immunohistochemistry()).resize((224, 224))
embedding(image)
threadmap(embedding, [image] * 16))
```



---

Can be deployed distributedly in kubernetes, use
`kubectl apply -f deployment.yaml`
listens to requetsts on `http://ctranspath-embedder-service:8000/embed_image`