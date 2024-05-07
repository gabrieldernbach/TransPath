Fork of ctranspath, packaged as an embedding service

Build and run container
`docker build -t ctranspath . && docker run -p 8000:8000 ctranspath`

or pull pre-build
`docker run -p 8000:8000 gabrieldernbach/histo:cstranspath`

You can now talk to ctranspath, e.g.
```python3
import io

import numpy as np
import requests
from PIL import Image
from skimage.data import immunohistochemistry


def img2byte(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def embeddings(imgs):
    uri = 'http://localhost:8000/embed_images'
    files = [("images", img2byte(img)) for img in imgs]
    response = requests.post(uri, files=files)
    return np.array(response.json())

sample_image = Image.fromarray(immunohistochemistry()).resize((224, 224))
embeddings([sample_image] * 5)
```
