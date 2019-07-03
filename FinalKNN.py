import struct as st
import numpy as np
from operator import itemgetter
from collections import Counter

def knn_sliding(k, train, test,label_test):
    pred_labels = []
    index = 0
    correct = 0
    for sample_test in test:
        dist = []
        for t in train:
            minim = 999999999999
            label = t[-1]
            for each in sample_test:
                euclid_dist = np.linalg.norm(np.array(t[:-1])- np.array(each))
                if euclid_dist < minim:
                    minim = euclid_dist
            dist.append([minim,label])
        counts = [i[1] for i in sorted(dist)[:k]]
        max_count_label = Counter(counts).most_common(1)[0][0]
        pred_labels.append(max_count_label)
        if max_count_label == label_test[index]:
            correct += 1
        index += 1

    total = np.size(test,0)
    accuracy = (correct/float(total))*100.00
    return accuracy, pred_labels

def knn_euclidean(k, data, test):
    predicted_labels = []
    true_counter = 0

    for tt in test:
        dist = []
        for d in data:
            label = d[-1]
            euclid_dist = np.linalg.norm(np.array(d[:-1]) -  np.array(tt[:-1]))
            dist.append([euclid_dist, label])
        counts = [i[1] for i in sorted(dist)[:k]]
        # print(Counter(counts).most_common(1))
        max_count_label = Counter(counts).most_common(1)[0][0]
        predicted_labels.append(max_count_label)
        if max_count_label == tt[-1]:
            # print(max_count_label)
            true_counter += 1

    total = np.size(test,0)
    accuracy = (true_counter/float(total))*100.00
    return accuracy, predicted_labels

def load_data(filename):
    if filename == "train-images-idx3-ubyte" or filename == "t10k-images-idx3-ubyte":
        path = "data/"+filename
        with open(path, 'rb') as f:
            byte = f.read(16)   #read first 16 byes
            #first 4 bytes magic num, then 4byes =total num of images,
            #the, 4bytes = number of rows, 4bytes = number of cols
            magic,size,nrows, ncols = st.unpack(">IIII", byte)

            #stores the rest of the file which is pixels of the images in the format: 60000 rows of 28*28 matrix/array
            img = np.asarray(st.unpack(('>'+'B'*size*nrows*ncols), f.read(size*nrows*ncols))).reshape((size,nrows,ncols))
            return magic,size,nrows,ncols,img

    if filename == "train-labels-idx1-ubyte" or filename== "t10k-labels-idx1-ubyte":
        path = "data/"+filename
        with open(path, 'rb') as f:
            byte = f.read(8)
            magic,size = st.unpack(">II", byte)
            label = np.asarray(st.unpack(('>'+'B'*size), f.read(size)))
        return label

def pad_image(test_image, target_size):
    '''This function operates on the given image and increases the dimension by 2.
        In this case, input is the test image of 28*28 dimension, which is then padded
        to make it 30*30.'''
    size_test = np.size(test_image,0)
    b = np.zeros((target_size,),dtype=int)
    padded_img_test = np.insert(test_image[0],0,0,axis=1) #adds an extra row in the last of 28 zeros
    padded_img_test = np.insert(padded_img_test,np.size(padded_img_test,1),0,axis=1)#adds one extra zero infront of every row of the 28*28 matrix
    padded_img_test = np.insert(padded_img_test,0,0,axis = 0) #adds one extra zero at last of every row of the modified matrix
    padded_img_test = np.append(padded_img_test, [b], axis= 0) #adds last row of 30 zeros

    for i in range(1,np.size(test_image,0)):
        a = np.insert(test_image[i],0,0,axis=1)
        a = np.insert(a,np.size(a,1),0,axis=1)
        a = np.insert(a,0,0,axis = 0)
        a = np.append(a, [b], axis= 0)
        padded_img_test = np.insert(padded_img_test, np.size(padded_img_test,0),a , axis =0)
        # print(padded_img_test)
        # break
    padded_img_test = np.reshape(padded_img_test,(size_test,target_size,target_size))
    # print(padded_img_test.shape)
    return padded_img_test



if __name__ == '__main__':

    ##load the data sets
    magic_train,size_train,rows_train,cols_train,img_train = load_data("train-images-idx3-ubyte")
    label_train = load_data("train-labels-idx1-ubyte")

    magic_test,size_test,rows_test,cols_test,img_test = load_data("t10k-images-idx3-ubyte")
    label_test = load_data("t10k-labels-idx1-ubyte")

    train = np.asarray([np.reshape(x, (784)) for x in img_train])
    dataset = np.column_stack((train, label_train))

    ##Shuffling data
    np.random.seed(0)
    np.random.shuffle(img) #randomly shuffles the 28*28 images

    ##Cross-vaidate to find optimal k
    final_acc = []
    validation_start = 0
    validation_end = 6000
    for k in range(1,11,2):
        for i in range(1,10):
            sum = 0
            if validation_start == 60000:
                break
            print("Epoch: ",i,"k=",k)
            validation_set = dataset[validation_start:validation_end]
            if validation_start == 0:
                train_set = dataset[validation_end:]
            elif validation_start == 54000:
                train_set = dataset[0:validation_start]
            else:
                train_set = np.concatenate((dataset[:validation_start],dataset[validation_end:    ]), axis=1)
            accuracy = knn_euclidean(k, train_set, validation_set)
            print("accuracy = ", accuracy)
            sum += accuracy
            validation_start = validation_end
            validation_end = validation_start + 6000
        final_acc.append([sum/float(10), k])

    optimal_k = final_acc[final_acc.index(max(final_acc))][1]
    print(optimal_k)

    ##Testing on Test Images
    test = np.asarray([np.reshape(x, 784) for x in img_test])
    testset = np.column_stack((test, label_test))
    acc_test, label_t_euclidean = knn_euclidean(optimal_k, dataset, testset)

    ##Testing using Sliding Window

    ###padding the images with zeros so that now the dimension is 30*30
    padded_test_images = pad_image(img_test, 30)
    flattened_images = []

    for i in range(np.size(padded_test_images,0)):
        for row in range(3):
            for col in range(3):
                temp_img = padded_test_images[i][row:row+28,col:col+28]
                flattened_images.append(np.reshape(temp_img,784))

    flat_imgs = np.asarray(flattened_images)
    flat_imgs = np.reshape(flat_imgs, (size_test, 9, 784))
    acc_sliding, lab_sliding = knn_sliding(optimal_k, dataset, flat_imgs, label_test)
    # print(acc)
    # print(lab)
