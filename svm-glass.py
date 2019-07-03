
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
import numpy as np
import collections
import time as watch
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier


# from sklearn.model_selection import

def load_data(filename):
    final_path = './glass_data/'+ filename
    data = []
    # label = []
    with open(final_path, 'r') as fh:
        for line in fh:
            data.append(line.rstrip().split(',')[1:])

    return data

def find_weights(data):
    w = {}
    label_count = collections.Counter(data[:,(np.shape(data)[1]-1)])
    for k in label_count.keys():
        w[k] = 1.0 -  float(label_count[k])/np.shape(data)[0]

    return w

def within_fold_train(kernel, x_train, y_train, x_test, y_test,parameter, type , weight=None):
    '''This function helps to operate on train and testing data form the 4-folds so
    as to find the optimal hyperparameters'''

    if type == 'ovr':
        if kernel == 'linear':
            max_acc = 0
            max_param = []
            for ind in parameter['C']:
                # print(ind)
                clf = OneVsRestClassifier(svm.SVC(C=ind, kernel=kernel,class_weight=weight))
                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)
                acc = accuracy_score(y_test, pred)

                if acc > max_acc:
                    max_acc = acc
                    max_param = [ind]
            return max_param

        elif kernel == 'poly':
            max_acc = 0
            optimal_params = []
            for c in parameter['C']:
                for g in parameter['gamma']:
                    for coef in parameter['coef0']:
                        for deg in parameter['degree']:
                            clf = OneVsRestClassifier(svm.SVC(C=c, kernel=kernel,degree=deg, gamma=g, coef0=coef, class_weight=weight))
                            clf.fit(x_train,y_train)
                            pred = clf.predict(x_test)
                            acc = accuracy_score(y_test,pred)

                            if acc > max_acc:
                                max_acc = acc
                                optimal_params= [c,g,coef,deg]
            return optimal_params

        elif kernel == 'sigmoid':
            max_acc = 0
            optimal_params = []
            for c in parameter['C']:
                for g in parameter['gamma']:
                    for coef in parameter['coef0']:
                        clf = OneVsRestClassifier(svm.SVC(C=c,kernel=kernel,gamma=g,coef0=coef,class_weight=weight))
                        clf.fit(x_train,y_train)
                        pred = clf.predict(x_test)
                        acc = accuracy_score(y_test, pred)

                        if acc> max_acc:
                            max_acc = acc
                            optimal_params = [c,g,coef]
            return optimal_params

        elif kernel == 'rbf':
            max_acc = 0
            optimal_params = []
            for c in parameter['C']:
                for g in parameter['gamma']:
                    clf = OneVsRestClassifier(svm.SVC(C=c,kernel=kernel,gamma=g,class_weight=weight))
                    clf.fit(x_train,y_train)
                    pred = clf.predict(x_test)
                    acc = accuracy_score(y_test, pred)

                    if acc> max_acc:
                        max_acc = acc
                        optimal_params = [c,g]
            return optimal_params

        else:
            print("No such kernel")
            raise ValueError('A very specific bad thing happened.')


    else:
        if kernel == 'linear':
            max_acc = 0
            max_param = []
            for ind in parameter['C']:
                # print(ind)
                clf = svm.SVC(C=ind, kernel=kernel, decision_function_shape=type,  class_weight=weight)
                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)
                acc = accuracy_score(y_test, pred)

                if acc > max_acc:
                    max_acc = acc
                    max_param = [ind]
            return max_param


        elif kernel == 'poly':
            max_acc = 0
            optimal_params = []
            for c in parameter['C']:
                for g in parameter['gamma']:
                    for coef in parameter['coef0']:
                        for deg in parameter['degree']:
                            clf = svm.SVC(C=c, kernel=kernel,degree=deg, gamma=g, coef0=coef,decision_function_shape=type,  class_weight=weight).fit(x_train,y_train)
                            pred = clf.predict(x_test)
                            acc = accuracy_score(y_test,pred)

                            if acc > max_acc:
                                max_acc = acc
                                optimal_params= [c,g,coef,deg]
            return optimal_params

        elif kernel == 'sigmoid':
            max_acc = 0
            optimal_params = []
            for c in parameter['C']:
                for g in parameter['gamma']:
                    for coef in parameter['coef0']:
                        clf = svm.SVC(C=c,kernel=kernel,gamma=g,coef0=coef,decision_function_shape=type,  class_weight=weight).fit(x_train,y_train)
                        pred = clf.predict(x_test)
                        acc = accuracy_score(y_test, pred)

                        if acc> max_acc:
                            max_acc = acc
                            optimal_params = [c,g,coef]
            return optimal_params

        elif kernel == 'rbf':
            max_acc = 0
            optimal_params = []
            for c in parameter['C']:
                for g in parameter['gamma']:
                    clf = svm.SVC(C=c,kernel=kernel,gamma=g,decision_function_shape=type,  class_weight=weight).fit(x_train,y_train)
                    pred = clf.predict(x_test)
                    acc = accuracy_score(y_test, pred)

                    if acc> max_acc:
                        max_acc = acc
                        optimal_params = [c,g]
            return optimal_params

        else:
            print("No such kernel")
            raise ValueError('A very specific bad thing happened.')


def fold_train(kernel,x_train, y_train, x_test, y_test,optimal_hyperparameter, type, weight=None):
    '''This function utilizes the optimal hyperparameters obtained from previous training,
    trains a new model on the entire 4-folds of data and tests on the remaining one fold,and
    returns back accuracy values'''

    if type == 'ovr':
        if kernel == 'linear':
            for ind in optimal_hyperparameter:
                # print(ind)
                clf = OneVsRestClassifier(svm.SVC(C=ind, kernel=kernel,class_weight=weight))
                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)
                acc = accuracy_score(y_test, pred)

            return acc

        elif kernel == 'poly':
            c = optimal_hyperparameter[0]
            g = optimal_hyperparameter[1]
            coef = optimal_hyperparameter[2]
            deg = optimal_hyperparameter[3]
            clf = OneVsRestClassifier(svm.SVC(C=c, kernel=kernel, degree=deg, gamma=g,coef0=coef,class_weight=weight))
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            acc = accuracy_score(y_test, pred)
            return acc

        elif kernel == 'sigmoid':
            c = optimal_hyperparameter[0]
            g = optimal_hyperparameter[1]
            coef = optimal_hyperparameter[2]
            clf = OneVsRestClassifier(svm.SVC(C=c,kernel=kernel,gamma=g,coef0=coef,class_weight=weight))
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            acc = accuracy_score(y_test, pred)
            return acc

        elif kernel == 'rbf':
            c = optimal_hyperparameter[0]
            g = optimal_hyperparameter[1]
            clf = OneVsRestClassifier(svm.SVC(C=c,kernel=kernel,gamma=g,class_weight=weight))
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            acc = accuracy_score(y_test, pred)
            return acc
        else:
            raise ValueError('Invalid value')

    else:
        if kernel =='linear':
            for ind in optimal_hyperparameter:
                # print(ind)
                clf = svm.SVC(C=ind, kernel=kernel, decision_function_shape=type, class_weight=weight).fit(x_train, y_train)
                pred = clf.predict(x_test)
                acc = accuracy_score(y_test, pred)

            return acc

        elif kernel =='poly':
            c = optimal_hyperparameter[0]
            g = optimal_hyperparameter[1]
            coef = optimal_hyperparameter[2]
            deg = optimal_hyperparameter[3]
            clf = svm.SVC(C=c, kernel=kernel, degree=deg, gamma=g,coef0=coef,decision_function_shape=type, class_weight=weight).fit(x_train, y_train)
            pred = clf.predict(x_test)
            acc = accuracy_score(y_test, pred)
            return acc

        elif kernel =='sigmoid':
            c = optimal_hyperparameter[0]
            g = optimal_hyperparameter[1]
            coef = optimal_hyperparameter[2]
            clf = svm.SVC(C=c, kernel=kernel, gamma=g,coef0=coef,decision_function_shape=type, class_weight=weight).fit(x_train, y_train)
            pred = clf.predict(x_test)
            acc = accuracy_score(y_test, pred)
            return acc

        elif kernel =='rbf':
            c = optimal_hyperparameter[0]
            g = optimal_hyperparameter[1]
            clf = svm.SVC(C=c, kernel=kernel,gamma=g,decision_function_shape=type,  class_weight=weight).fit(x_train, y_train)
            pred = clf.predict(x_test)
            acc = accuracy_score(y_test, pred)
            return acc
        else:
            raise ValueError('Invalid value')



def main():
    glass_data= load_data('glass.data')
    # print glass_data[:5]
    total_samples = np.shape(glass_data)[0]
    # print(total_samples)
    #convert to numpy array
    glass_data = np.asarray(glass_data,dtype=float)
    #find the weights based on counts of each label
    weight_list = find_weights(glass_data)

    #scale/normalize the data points to range [0,1] note: not the class label
    min_max_scaler = preprocessing.MinMaxScaler()
    glass_data_scaled = min_max_scaler.fit_transform(glass_data[:,:-1])
    labels = glass_data[:,-1]
    #appending labels back to this scaled data
    glass_data_scaled = np.column_stack((glass_data_scaled,labels))

    #shuffle the data
    np.random.seed(0)
    np.random.shuffle(glass_data_scaled)

    cv_folds = 5
    size_of_one_fold = total_samples/cv_folds#42

    parameter_grid = {'gamma':[2,3,0.1, 0.01,0.001,0.0001], 'C':[1,10,100,200, 1000],
                        'coef0':[0.1,0.3, 0.5, 1, 1.5,2],'degree':[2, 3, 4, 5, 6, 7]}
    type_svm = ['ovo', 'ovr']
    kernels = ['linear', 'poly', 'sigmoid', 'rbf']
    ##Here's the oVo and oVr methods efficiency and time comparison
    ##different folds training
    print("###############")
    print("Using no weights")
    for type in type_svm:
        print('Type of svm: %s'%type)
        t = watch.time()
        val_start = 0
        val_end = size_of_one_fold
        acc_list = {'linear':[], 'poly':[], 'rbf':[], 'sigmoid':[]}
        for i in range(cv_folds):
            if val_start == 0:
                val_set = glass_data_scaled[val_start:val_end]
                train_set = glass_data_scaled[val_end:]
            elif val_start == 168:
                train_set = glass_data_scaled[0:val_start]
                val_set = glass_data_scaled[val_start:]
            else:
                train_set = np.concatenate((glass_data_scaled[:val_start],glass_data_scaled[val_end:]), axis=0)
                val_set = glass_data_scaled[val_start:val_end]

            val_start = val_end
            val_end = val_end + 42

            #separating data and labels
            val_data = val_set[:, :-1]
            val_label = val_set[:,-1]
            train_data = train_set[:, :-1]
            train_label = train_set[:, -1]

            #80% and 20% splitting
            t_train, t_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, shuffle=False)
            for k in kernels:
                optimal_hyperparam = within_fold_train(k,t_train, y_train, t_test, y_test,parameter_grid, type )
                # print(optimal_hyperparam)
                accuracy = fold_train(k,train_data, train_label, val_data, val_label,optimal_hyperparam, type)
                acc_list[k].append((accuracy,optimal_hyperparam))

        # print(acc_list)
        for key,val in acc_list.items():
            print("Avg. accuracy for %s: "%key)
            avg = (val[0][0]+val[1][0]+val[2][0]+val[3][0]+val[4][0])/5 * 100.00
            print avg
        print("Time taken : %f"% (watch.time()- t))


    ##Since unbalanced dataset, we use weights now and again compare
    ##for both oVo and ovr methods
    print('########')
    print("Using Weights Now to balance the unbalanced dataset")
    print("Only One V One SVM is used in this case:")
    type = "ovo"
    print('Type of svm: %s'%type)
    t = watch.time()
    val_start = 0
    val_end = size_of_one_fold
    acc_list = {'linear':[], 'poly':[], 'rbf':[], 'sigmoid':[]}
    for i in range(cv_folds):
        if val_start == 0:
            val_set = glass_data_scaled[val_start:val_end]
            train_set = glass_data_scaled[val_end:]
        elif val_start == 168:
            train_set = glass_data_scaled[0:val_start]
            val_set = glass_data_scaled[val_start:]
        else:
            train_set = np.concatenate((glass_data_scaled[:val_start],glass_data_scaled[val_end:]), axis=0)
            val_set = glass_data_scaled[val_start:val_end]

        val_start = val_end
        val_end = val_end + 42

        #separating data and labels
        val_data = val_set[:, :-1]
        val_label = val_set[:,-1]
        train_data = train_set[:, :-1]
        train_label = train_set[:, -1]

        #80% and 20% splitting
        t_train, t_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, shuffle=False)
        for k in kernels:
            optimal_hyperparam = within_fold_train(k,t_train, y_train, t_test, y_test,parameter_grid, type, 'balanced' )
            # print(optimal_hyperparam)
            accuracy = fold_train(k,train_data, train_label, val_data, val_label,optimal_hyperparam, type, 'balanced')
            acc_list[k].append((accuracy,optimal_hyperparam))

    for key,val in acc_list.items():
        print("Avg. accuracy for %s:" % key)
        avg = (val[0][0]+val[1][0]+val[2][0]+val[3][0]+val[4][0])/5 * 100.00
        print avg
    print("Time taken : %f"% (watch.time()- t))




if __name__ == '__main__':
    main()
