# Carregar o dataset MNIST

# Obs: Este script é baseado na versão do livro http://neuralnetworksanddeeplearning.com/, com a devida autorização do autor.

# Imports
import pickle
import gzip
import numpy as np
from numpy import trapz

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def salvaImagens(pklLoad):
   # print(type(training_data[0][0]))
    #print(len(training_data[0][0]))
   # print(training_data[0][0])
    i=0
    aux = pklLoad
    for conjImg in aux:
       for img in conjImg:
           for p in img:
               p = p * 255
           img.astype(np.uint8)
           data = img.reshape(28, 28)
           image = np.asarray(data).squeeze()
           #plt.imshow(img)
           plt.imsave("img1/img_"+ str(i) +".png", image, cmap='gray')
           #plt.savefig("img1/img_"+ str(i) +".png")
           i+=1

def lerImagens(numImg=5):
    # imgplot = plt.imshow(img)
    # img.show()
    imgs =[]
    for i in range(numImg):
        img = Image.open("img1/img_"+str(i)+".png").convert('L')
        data = np.asarray(img,dtype=np.float32)
        data = data.reshape(784,1)
        for i,p in enumerate(data):
            data[i] = p/2
        imgs.append(data)

    return imgs

def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

  #  salvaImagens(training_data)

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
