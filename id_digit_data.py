import glob
import pickle
import gzip
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2

DATA_DIR = "/home/tra161/WORK/Data/bagiks/ID_DIGITS/gray_labelled"
TRAINING_SIZE = 0.6 # 60% for training, 40% for validation
MAX_ROW = 39
MAX_COL = 34

def prepare_data():
    train_dat = np.empty((0,MAX_ROW,MAX_COL,3),float)
    train_lab = []
    valid_dat = np.empty((0,MAX_ROW,MAX_COL,3),float)
    valid_lab = []
    max_row = 0
    max_col = 0
    for i in range(9):
        count = 0
        fs = glob.glob(DATA_DIR+"/"+str(i)+"/*.jpg")
        for f in fs:
            sample = np.zeros((MAX_ROW,MAX_COL,3),dtype=float)
            #dat = cv2.imread(f)
            dat = Image.open(f)
            dat = np.asarray(dat)
            dat = np.where(dat>0,dat,0)
            plt.imshow(dat)
            plt.show()
            print(dat.shape)
            input("")
            #if max_row< dat.shape[0]:
            #    max_row = dat.shape[0]
            #if max_col< dat.shape[1]:
            #    max_col = dat.shape[1]
            sample[:dat.shape[0],:dat.shape[1],:] = dat
            plt.imshow(sample)
            plt.show()
            sample = sample[np.newaxis,:,:,:]
            count+=1
            if count<TRAINING_SIZE*len(fs):                
                train_dat = np.append(train_dat,sample,axis=0)
                train_lab.append(i)
            else:
                valid_dat = np.append(valid_dat,sample,axis=0)
                valid_lab.append(i)
    train_lab = np.array(train_lab)
    valid_lab = np.array(valid_lab)
    
    print(train_dat.shape)
    print(valid_dat.shape)
    pkl_file = DATA_DIR+"/gray_data.pkl.gz"    
    with gzip.open(pkl_file,"wb") as f:
        pickle.dump({"X_train":train_dat,
                     "y_train":train_lab,
                     "X_valid":valid_dat,
                     "y_valid":valid_lab},f)    
    
def test_tf_data():
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(DATA_DIR+"/"+str(0)+"/*.jpg"))
    image_reader = tf.WholeFileReader()

    _,image_file = image_reader.read(filename_queue)

    image = tf.image.decode_jpeg(image_file)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        image_tensor = sess.run([image])
        
        print(image_tensor.shape)
        
if __name__=="__main__":
    prepare_data()
    
