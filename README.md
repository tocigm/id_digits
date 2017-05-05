How to run:

A. Training & validation
  + Prepare data: id_gray_digit_data.py
              - set DATA_DIR to the data directory
              - TRAINING_SIZE: % data for training, the rest is for validation
              - DENOISE_THRESHOLD: all pixels over this values are considerred as noise and set to 255 (white background)
  + Train/validation: id_digit_svm.py
        All possible models are tested here, the data can be normalised to 0-1 or -1:1.
        
B. Predict: see test() in id_digit_svm.py

# id_digits
Without normalise:

SVM    0.166427134234

QDA    0.361598160391

NB     0.846795056051

DCT    0.918942224777

LDA    0.989364759989

RF     0.990514515665

LG     0.994826099454

Normalise 0:1

QDA    0.443518252371

NB     0.843633227939

DCT    0.91434320207

SVM    0.986202931877

LDA    0.989364759989

RF     0.990514515665

LG     0.994538660535


Normalise -1:1

NB    -1

QDA    0.422822650187

DCT    0.917505030181

LDA    0.989364759989

RF     0.989364759989

LG     0.994538660535

SVM    0.998850244323  ------> BEST 


CNN:

Lenet:


