import pickle
import gzip

class ID_DIGIT(object):
    def __init__(self,data_path):
        print("Loading ID digit data")
        with gzip.open(data_path+'/data.pkl.gz','rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data_dict = data_dict = u.load()
            self.train_dat  = data_dict["X_train"]
            self.train_lab  = data_dict["y_train"]
            self.valid_dat  = data_dict["X_valid"]
            self.valid_lab  = data_dict["y_valid"]

            print(self.train_dat.shape)
            print(self.train_lab.shape)
            print(self.valid_dat.shape)
            print(self.valid_lab.shape)

        self.start_inx = self.end_inx = 0
    def next(self,batch_size=100):
        if self.start_inx>=self.train_dat.shape[0]:
            self.start_inx = self.end_inx = 0
            return None,None
        self.end_inx = min(self.end_inx+batch_size,self.train_dat.shape[0])
        x = self.train_dat[self.start_inx:self.end_inx,:,:,:]
        y = self.train_lab[self.start_inx:self.end_inx]
        self.start_inx = self.end_inx

        return x,y
    def get_train(self):
        return self.train_dat,self.train_lab
    def get_valid(self):
        return self.valid_dat,self.valid_lab
    @property
    def max_col(self):
        return self.train_dat.shape[2]
    @property
    def max_row(self):
        return self.train_dat.shape[1]

if __name__=="__main__":
    data = ID_DIGIT("/home/tra161/WORK/Data/bagiks/ID_DIGITS/labelled")
    while True:
        x,y = data.next()
        if x is not None:
            print(x.shape)
            print(y.shape)
        else:
            print("None")
        input("")
    
