# 
FROM python:3.9.7

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app


# 
COPY ./gradient_boosting_model /code/gradient_boosting_model

# 
COPY ./data /code/data

# train_pipeline
#CMD ["python", "gradient_boosting_model/train_pipeline.py"]

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
