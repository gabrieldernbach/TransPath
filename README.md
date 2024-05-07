Fork of ctranspath, packaged as an embedding service

Build and run container
`docker build -t ctranspath . && docker run -p 8000:8000 ctranspath`

or pull pre-build
`docker run -p 8000:8000 gabrieldernbach/histo:cstranspath`

You can now talk to ctranspath, e.g.
```python3
import base64
from io import BytesIO

import requests
from PIL import Image
from skimage.data import immunohistochemistry
from tqdm.contrib.concurrent import thread_map

def encode_image(image: Image) -> str:
    """Converts an image to a base64-encoded JPEG string."""
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def request_image_embeddings(images: [Image]) -> list:
    """Posts a list of images to an embedding service and returns the embeddings."""
    data = {'images': [encode_image(i) for i in images]}
    response = requests.post('http://localhost:8000/embed_images', json=data)
    return response.json()['vectors']

# Prepare example image and replicate it for multiple requests
sample_image = Image.fromarray(immunohistochemistry())
image_batch = [sample_image] * 4

# Use concurrent threads to request embeddings for multiple batches
embeddings = thread_map(request_image_embeddings, [image_batch] * 4)
```
