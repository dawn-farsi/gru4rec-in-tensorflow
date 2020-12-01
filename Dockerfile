FROM python:3.6.6
WORKDIR /working
ADD . /working
#ADD requirements.txt /working
RUN pip install -r requirements.txt

CMD ["python3", "run/article_reco_gru_training.py"]
