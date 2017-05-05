from id_digit_cnn import ID_DIGIT_CNN
from id_digits import ID_DIGIT

class Config:
    learning_rate = 0.5
    MAX_ITER = 100
    LR_DEC_LIMIT = 5
    EARLY_STOPPING = 5

    batch_size = 100
if __name__=="__main__":
    data = ID_DIGIT("./")
    conf = Config
    model = ID_DIGIT_CNN(conf,data)
    model.run()
    
    
    
