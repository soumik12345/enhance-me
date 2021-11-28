# Pull Base Image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set Working Directory
RUN mkdir /usr/src/enhance-me
WORKDIR /usr/src/enhance-me

# Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip setuptools wheel
RUN pip install gdown matplotlib streamlit tqdm wandb

COPY . /usr/src/enhance-me/

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
