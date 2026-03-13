FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]