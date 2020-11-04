# MIDI-LSTM-Multipe-Voice

## Additions

Supplementing and developing off the MIDI-LSTM---TRAINING PIPELINE, this program generates piano midi. Where as Cello midi focuses on melody as a combination of a vector of pitchs and a vector of durations - here we address midi as a 128x300 matrix of pitch and duration. Additionally, I've added in classifier, LinearSVC, to critique the LSTM's generated midi.

## CONTENTS OF Program

* Web MIDI Scraper
* MIDI parser
* MIDI data transformer
* RNN-ATTN model with Stacked LSTMs 

RNN-ATTN with Stacked LSTMs:
* Model's architeture below

         ​(Embedded Pitch & Duration Input Matrix)
                        ​|
                        ​|
                        ​|
                    *---​LSTM
                    |    -Dropout
                    |    ​|
                    |    ​LSTM
                    |    ​-Dropout
                    |    ​|
                    |    ​ATTN MODEL
                    |    ​|
                    *---LSTM
                    |    -Dropout
                    |    |
                    |    ATTN MODEL
                    |    |
                    |    ​Fully Connected Layer
                    |    -Dropout
                    |    ​|
                    *---LSTM
                    |    -Dropout
                    |    |
                    |    Fully Connected Layer
                    |    -Dropout
                    |    |
                    |    ATTN MODEL
                    |    |
                    |    Fully Connected Layer
                    |    -Dropout
                    |    |
                    |    Fully Connected Layer
                    |    -Dropout
                    |    |
                    *---LSTM
                    |    -Dropout
                    |    |
                    |    Fully Connected Layer
                    |    -Dropout
                    |   ​|
        ​Fully Connected Layer (softmax output)
                ​        |                      
            (​Pitch & Duration Output Matrix)

#### Webscraper:

* MIDI is scraped from www.piano_midi.com and www.midi_world.com. All piano peices are scraped. 

Parameters and Model Configurations, see: 

    composer_RNN.py 
    utils/params.py


## EXECUTE Program

### Aim of Program:

The expection for the pipeline is to:
* webscrape midi 
* transform and store data as necessary
* construct NN architecture 
* engineer the data to achitecture input
* train model on scraped midi
* save weights of best trained model
* generate midi file

### How to Use
    
    git clone https://github.com/FinchMF/MIDI-LSTM-Multiple-Voice.git
    pip install -r requirements.txt

There are two operational shell scripts:

    $ bash reset.sh 
    $ bash execute.sh '<composername>'

## List of Composers

* albeniz		
* bartok		
* borodin		
* chopin		
* godowsky	
* handel	
* liszt		
* mj_starke	
* muss		
* mw_liszt	
* pacchioni	
* ravel
* schum		
* sinding 
* bach		
* beeth		
* brahms		
* clementi
* grana		
* haydn		
* mendelssohn	
* moszkowski	
* mw_bach	
* mw_mozrt
* rach		
* scarlatti	
* schumann	
* tschai  
* balak	
* beethoven 
* burgm		
* debuss		
* grieg		
* mozart	
* mw_byrd		
* mw_tchai	
* rachmaninov	
* schub		
* shostakovich

### RESET
Reset will ease folders: 'data' , 'run' and 'models' . This resets repo.

### EXECUTE
After the initial repo clone, run: 
        
    $ bash execute.sh '<composername>'
