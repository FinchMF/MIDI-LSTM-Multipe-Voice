#!/bin/bash/

echo [+] Modeling $1;

if [ -d './data' ];
then 
    if [ -f '.models/'$1'_piano_model.h5' ];
    then
        echo [i] $1 model already exists
        # engage composer
        python composer_write.py

    else
        echo [i] $1 model needs to be trained
        # train the composer
        python composer_train.py $1
        # engage composer
        python composer_write.py $1

    fi

else
    echo [+] Building Directories and Gathering Midi Data
    # setup directories 
    bash setup.sh
    # gather data
    python fetch_midi_data.py
    # train the composer
    python composer_train.py $1
    # engage composer
    python composer_write.py $1

fi;




