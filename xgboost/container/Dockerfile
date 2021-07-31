FROM python:3.8

# Install the necessary libraries
RUN pip3 install numpy sklearn xgboost pandas sagemaker-training

# Copy the script that will serve as the entrypoint for 
# training
COPY train.py /opt/ml/code/train.py 

ENV SAGEMAKER_PROGRAM train.py
