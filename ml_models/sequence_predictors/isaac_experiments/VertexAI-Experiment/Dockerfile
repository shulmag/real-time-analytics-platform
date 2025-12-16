FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8.py37

WORKDIR / 

COPY . . 

# RUN cd ficc_python && \
#     pip install -r requirements.txt && \
#     pip install . -U

RUN pip install -r requirements.txt

RUN pip install numpy --upgrade

ENTRYPOINT ["python", "train.py"]