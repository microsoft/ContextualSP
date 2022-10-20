FROM python:3.6.8
RUN pip install numpy==1.19.2
RUN pip install scikit-learn==0.23.2
WORKDIR /eqasc
ENV PYTHONPATH .
ENV PYTHONUNBUFFERED yes
COPY allennlp_reasoning_explainqa /eqasc/allennlp_reasoning_explainqa
