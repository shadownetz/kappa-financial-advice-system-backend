# Use a tiangolo base image that runs FastAPI on Uvicorn/Gunicorn
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9


# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-build the matplotlib font cache
RUN python -c "import matplotlib.pyplot"

# Copy the rest of the code
COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]