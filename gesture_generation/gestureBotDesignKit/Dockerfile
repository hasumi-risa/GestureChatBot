# docker build -t gesbot:1.0 .

FROM python:3.8-buster

# Python Libraries
RUN pip3 install \
        pandas==1.2.0 \
        numpy==1.19.3 \
        seaborn==0.11.1 \
        xgboost==1.3.1 \
        scikit-learn==0.24.0 \
        bayesian-optimization==1.2.0 \
        jupyter==1.0.0 \
        mlxtend==0.18.0 \
        corenlp-python==3.4.1.post1 \
        transformers==3.4.0 \
        openpyxl \
        librosa \
        tqdm \
        scipy \
        torch \
        torchvision \
        torchaudio \
        torchtext==0.6.0 \
        requests==2.25.1 \
        azure-cognitiveservices-speech \
        opencv-python

# Java
RUN     apt-get update  -y \
    &&  apt-get install -y software-properties-common \
    &&  add-apt-repository ppa:openjdk-r/ppa \
    &&  apt-get install -y openjdk-11-jdk \
    &&  rm -rf /var/lib/apt/lists/*


# java for CoreNLP
# RUN apt-get install openjdk-11-jdk -y \
#     && wget http://nlp.stanford.edu/software/stanford-corenlp-full-2013-06-20.zip \
#     && unzip ./stanford-corenlp-full-2013-06-20.zip -d ./imagistic_generator/src/NLP/ \
#     && rm ./stanford-corenlp-full-2013-06-20.zip


# ffmpeg
# RUN     apt update -y \ 
#     &&  apt install ffmpeg -y

