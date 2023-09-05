#!/bin/bash

MODELS_PATH=$PWD/models

if test -d $MODELS_PATH; then
    echo "Setup complete"
else
    mkdir $MODELS_PATH
    cd $MODELS_PATH
    wget https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip
    unzip vosk-model-small-pl-0.22.zip
    rm vosk-model-small-pl-0.22.zip
    echo 'export PYTHONPATH="${PYTHONPATH}:'$PWD'"' >> ~/.bashrc
    echo "Setup complete"
fi
