import os
import h5py
import numpy as np

# extract data and label, transform the data
def data_process(data):
    ### ***  Data from mat.file:  *** ###
    testdata, testlabel = data['testdata'], data['testlabel']       # testdata: (5, 62, 1384)  testlabel: (1384, 1)
    traindata, trainlabel = data['traindata'], data['trainlabel']   # traindata: (5, 62, 2010)  trainlabel: (2010, 1)

    ### ***  After Transpose:  *** ###
    testdata_tran = np.transpose(testdata, (2, 0, 1))               # testdata: (1384, 5, 62)  
    traindata_tran =  np.transpose(traindata, (2, 0, 1))            # traindata: (2010, 5, 62)


    ### ***  After Reshape:  *** ###
    testdata_re = testdata_tran.reshape(len(testdata_tran), -1)     # testdata: (1384, 310)
    traindata_re =  traindata_tran.reshape(len(traindata_tran), -1) # traindata: (2010, 310)

    return testdata_re, traindata_re, testlabel, trainlabel


def get_SEED_data_label(path='../SEED/'):
    data_path = path

    for root, dirs, files in os.walk(data_path):
        filename_list = files
    print('Numbers of SEED files:', len(filename_list))

    print('********* Data processing begin: *********\n')
    counter = 0
    for filename in filename_list:
        
        file_path = data_path + filename
        data_h5py = h5py.File(file_path, 'r')
        
        testdata, traindata, testlabel, trainlabel = data_process(data_h5py)
        
        # print('***** For', filename, ': *****')
        # print('testdata:\t', testdata.shape)
        # print('traindata:\t', traindata.shape)
        # print('testlabel:\t', testlabel.shape)
        # print('trainlabel:\t', trainlabel.shape, '\n')
        
        if counter == 0:
            seed_data = np.vstack((testdata, traindata))
            seed_data_label = np.vstack((testlabel, trainlabel))
        else:
            seed_data = np.vstack((seed_data, testdata))        
            seed_data = np.vstack((seed_data, traindata))
            
            seed_data_label = np.vstack((seed_data_label, testlabel))
            seed_data_label = np.vstack((seed_data_label, trainlabel))
            
        counter = counter + 1
    
    seed = np.hstack((seed_data_label, seed_data))

    print('********* Data processing end: *********')
    print('seed:\t\t\t', seed.shape)
    print('seed_data:\t\t', seed_data.shape)
    print('seed_data_label:\t', seed_data_label.shape)

    return seed_data, seed_data_label, seed