FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
RUN curl https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view -o ctranspath.pth
RUN curl https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view -o timm-0.5.4.tar
RUN pip install timm-0.5.4.tar
