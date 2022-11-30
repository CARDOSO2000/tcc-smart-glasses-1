#Deriving the latest base image
FROM python:3.10.8


LABEL Maintainer="roushan.me17"
RUN apt-get update
RUN apt-get install libasound-dev -y
RUN apt-get install libsndfile-dev -y
RUN apt-get install libportaudio2 -y
RUN pip install sounddevice
RUN pip install soundfile
RUN pip install tensorflow
RUN pip install python_speech_features
RUN pip install scipy


COPY run.py ./
COPY cnn_v2.h5 ./
COPY cnn_4_classes_v2.h5 ./
COPY deploy.wav ./

CMD [ "python", "run.py"]