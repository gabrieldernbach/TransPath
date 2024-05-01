FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN apt-get update; apt-get install curl vim -y

RUN pip install gdown
RUN gdown https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view?usp=sharing --fuzzy
RUN pip install timm-0.5.4.tar && rm timm-0.5.4.tar

RUN mkdir /app
RUN gdown https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view?usp=sharing --fuzzy -O /app/ctranspath.pth 
COPY script.py /app/script.py

WORKDIR /app
