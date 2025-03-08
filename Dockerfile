# Use a tiangolo base image that runs FastAPI on Uvicorn/Gunicorn
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Let the base image know our main file is `views.py`
ENV MODULE_NAME=views

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Pre-build the matplotlib font cache
RUN python -c "import matplotlib.pyplot"

# Copy the rest of the code
COPY . /app
