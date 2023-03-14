import os
import tensorflow
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore", category=Warning)

import model
import utils
import argparse
from ipdb import set_trace
import data_preprocessing_forSEED

import h5py

parser = argparse.ArgumentParser(description="Argparse of SEED training.")
parser.add_argument("-g", "--gpu_ids", type=str, default='0', help="choose GPU index for training")
parser.add_argument("-n", "--data_nums", type=int, default=5000, help="the numbers of SEED data for training")
parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size for traning")
parser.add_argument("-f", "--gpu_fraction", type=float, default=1.0, help="per_process_gpu_memory_fraction")
parser.add_argument("-p", "--data_path", type=str, default='../SEED/', help="SEED data path")


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids) # 使用 GPU 0，1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dirname = '../'
## mention paths
data_folder     = os.path.join(os.path.dirname(dirname), 'data_folder')
summaries       = os.path.join(os.path.dirname(dirname), 'summaries')
output_folder   = os.path.join(os.path.dirname(dirname), 'output')
model_dir       = os.path.join(os.path.dirname(dirname), 'models')

## transformation task params
noise_param         = 15                        #noise_amount
scale_param         = 1.1                       #scaling_factor
permu_param         = 20                        #permutation_pieces
tw_piece_param      = 9                         #time_warping_pieces
twsf_param          = 1.05                      #time_warping_stretch_factor
no_of_task          = ['original_signal', 'noised_signal', 'scaled_signal', 'negated_signal', 'flipped_signal', 'permuted_signal', 'time_warped_signal'] 

transform_task      = [0, 1, 2, 3, 4, 5, 6]     #transformation labels

single_batch_size   = len(transform_task)

## hyper parameters
batchsize               = args.batch_size  
actual_batch_size       = batchsize * single_batch_size
log_step                = 100
epoch                   = 100
initial_learning_rate   = 0.0001
drop_rate               = 0.4
regularizer             = 1
L2                      = 0.0001
lr_decay_steps          = 10000
lr_decay_rate           = 0.9
loss_coeff              = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]

window_size             = 310 ## 2560
extract_data            = 0
current_time            = utils.current_time()
data_num                = args.data_nums
# seed_path               = '../SEED_for_test/'
seed_path               = args.data_path

graph = tf.Graph()
print('creating graph...')
with graph.as_default():
    ## initialize tensor
    
    input_tensor        = tf.placeholder(tf.float32, shape = (None, window_size, 1), name = "input")
    y                   = tf.placeholder(tf.float32, shape = (None, np.shape(transform_task)[0]), name = "output") 
    drop_out            = tf.placeholder_with_default(1.0, shape=(), name="Drop_out")
    isTrain             = tf.placeholder(tf.bool, name = 'isTrain')
    global_step         = tf.Variable(0, dtype=np.float32, trainable=False, name="steps")

    conv1, conv2, conv3, main_branch, task_0, task_1, task_2, task_3, task_4, task_5, task_6 = model.self_supervised_model(input_tensor, isTraining= isTrain, drop_rate= drop_out)
    logits          = [task_0, task_1, task_2, task_3, task_4, task_5, task_6]

    supervised_model = model.supervised_model(input_dimension=128, output_dimension=3, hidden_nodes = 512, dropout = 0.1, lr_super = 0.00001, L2=0.01)
    ## main branch is the output after all conv layers
    featureset_size = main_branch.get_shape()[1]
    y_label         = utils.get_label(y=y, actual_batch_size=actual_batch_size)
    all_loss        = utils.calculate_loss(y_label, logits)
    output_loss     = utils.get_weighted_loss(loss_coeff, all_loss)  
    
    if regularizer:
        l2_loss = 0
        weights = []
        for v in tf.trainable_variables():
            weights.append(v)
            if 'kernel' in v.name:
                l2_loss += tf.nn.l2_loss(v)
        output_loss = output_loss + l2_loss * L2
        
    y_pred                = utils.get_prediction(logits = logits)
    learning_rate         = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate, staircase=True)

    optimizer             = tf.train.AdamOptimizer(learning_rate) 
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op    = optimizer.minimize(output_loss, global_step, colocate_gradients_with_ops=True)
        
    with tf.variable_scope('Session_saver'):
        saver       = tf.train.Saver(max_to_keep=10)

    tf.summary.scalar('learning_rate/lr', learning_rate)
    tf.summary.scalar('loss/training_batch_loss', output_loss)
    
    summary_op      = tf.summary.merge_all()    
    
print('graph creation finished')

for root, dirs, files in os.walk(seed_path):
    filename_list = files
print('Numbers of SEED files:', len(filename_list))

# 各模型最好的结果
# best_tr_loss = ['tr_loss']
# best_tr_accuracy = ['tr_accuracy']
# best_tr_f1_score = ['tr_f1_score']

# best_te_loss = ['te_loss']
# best_te_accuracy = ['te_accuracy']
# best_te_f1_score = ['te_f1_score']

# best_supervised_tr_score_accuracy = ['supervised_tr_score_accuracy']
# best_supervised_tr_score_precision = ['supervised_tr_score_precision']
# best_supervised_tr_score_recall = ['supervised_tr_score_recall']
# best_supervised_tr_score_f1_score = ['supervised_tr_score_f1_score']

# best_supervised_te_score_accuracy = ['supervised_te_score_accuracy']
# best_supervised_te_score_precision = ['supervised_te_score_precision']
# best_supervised_te_score_recall = ['supervised_te_score_recall']
# best_supervised_te_score_f1_score = ['supervised_te_score_f1_score'] 

best_tr_loss = []
best_tr_accuracy = []
best_tr_f1_score = []

best_te_loss = []
best_te_accuracy = []
best_te_f1_score = []

best_supervised_tr_score_accuracy = []
best_supervised_tr_score_precision = []
best_supervised_tr_score_recall = []
best_supervised_tr_score_f1_score = []

best_supervised_te_score_accuracy = []
best_supervised_te_score_precision = []
best_supervised_te_score_recall = []
best_supervised_te_score_f1_score = [] 

utils.makedirs('../record')
for index, filename in enumerate(filename_list):
    tr_loss = []
    tr_accuracy = []
    tr_f1_score = []
    
    te_loss = []
    te_accuracy = []
    te_f1_score = []

    supervised_tr_score_accuracy = []
    supervised_tr_score_precision = []
    supervised_tr_score_recall = []
    supervised_tr_score_f1_score = []

    supervised_te_score_accuracy = []
    supervised_te_score_precision = []
    supervised_te_score_recall = []
    supervised_te_score_f1_score = []    
    max(supervised_tr_score_accuracy,default=0)
    max(supervised_te_score_accuracy, default=0)
    savepath = '../record/' +  filename.split('.')[0]
    utils.makedirs(savepath)

    print('**************************************\n')
    print('********* For', filename, ': *********\n')
    print('**************************************\n')
    file_path = seed_path + filename
    data_h5py = h5py.File(file_path, 'r')
    
    testdata, traindata, testlabel, trainlabel = data_preprocessing_forSEED.data_process(data_h5py)

    ## extract training and testing EGG data
    train_EGG = traindata
    train_EGG = shuffle(train_EGG)
    test_EGG = testdata
    
    ## fetch emotion recognition labels
    train_label, test_label = utils.one_hot_encoding_45(train_label=trainlabel, test_label=testlabel)

    training_length = train_EGG.shape[0]
    testing_length  = test_EGG.shape[0]
    print('Training Data shape:', train_EGG.shape)
    print('Testing Data shape:', test_EGG.shape)
    print('Training Label shape:', train_label.shape)
    print('Testing Label shape:', test_label.shape)

    ## save STR results
    tr_ssl_result_filename  =  os.path.join(output_folder, "STR_result"   , str("tr_" + str(index) +"_"  + current_time + ".npy"))
    te_ssl_result_filename  =  os.path.join(output_folder, "STR_result"   , str("te_" + str(index) +"_"  + current_time + ".npy"))
    tr_ssl_loss_filename    =  os.path.join(output_folder, "STR_loss"     , str("tr_" + str(index) +"_"  + current_time + ".npy"))
    te_ssl_loss_filename    =  os.path.join(output_folder, "STR_loss"     , str("te_" + str(index) +"_"  + current_time + ".npy"))
            
    str_logs        = os.path.join(summaries, "STR", current_time)
    er_logs         = os.path.join(summaries, "ER", current_time)
    utils.makedirs(str_logs)
    
    print('Initializing all parameters.')
    tf.reset_default_graph()

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)
    config = tf.ConfigProto()                                               # 对session进行参数配置
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction  # 分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True                                  # 按需分配显存，这个比较重要

    with tf.Session(graph=graph, config=config) as sess:   
        summary_writer = tf.summary.FileWriter(str_logs, sess.graph)
    
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        print('self supervised training started')
        
        train_loss_dict     = {}
        test_loss_dict      = {}
    
        tr_ssl_result       = {}
        te_ssl_result       = {}    
        
        ## epoch loop
        for epoch_counter in tqdm(range(epoch)):
            
            tr_loss_task    = np.zeros((len(transform_task), 1), dtype  = np.float32)
            train_pred_task = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32) -1
            train_true_task = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32) -1
            tr_output_loss  = 0
    
           
            tr_total_gen_op = utils.make_total_batch(data = train_EGG, length = training_length, batchsize = batchsize, 
                                               noise_amount=noise_param, 
                                               scaling_factor=scale_param, 
                                               permutation_pieces=permu_param, 
                                               time_warping_pieces=tw_piece_param, 
                                               time_warping_stretch_factor= twsf_param, 
                                               time_warping_squeeze_factor= 1/twsf_param)
            
            print('***** self supervised training *****')
            for training_batch, training_labels, tr_counter, tr_steps in tr_total_gen_op:
                # print('***** Training *****')
                ## run the model here 
                training_batch, training_labels = utils.unison_shuffled_copies(training_batch, training_labels)
                training_batch  = training_batch.reshape(training_batch.shape[0], training_batch.shape[1], 1)
                fetches         = [all_loss, output_loss, y_pred, train_op]
                if tr_counter % log_step == 0:
                    fetches.append(summary_op)
                    
                fetched = sess.run(fetches, {input_tensor: training_batch, y: training_labels, drop_out: drop_rate, isTrain: True})
                
                if tr_counter % log_step == 0: # 
                    summary_writer.add_summary(fetched[-1], tr_counter)
                    summary_writer.flush()
    
                tr_loss_task = utils.fetch_all_loss(fetched[0], tr_loss_task) 
                tr_output_loss += fetched[1]
                
                train_pred_task = utils.fetch_pred_labels(fetched[2], train_pred_task)
                train_true_task = utils.fetch_true_labels(training_labels, train_true_task)

            ## loss after epoch
            tr_epoch_loss = np.true_divide(tr_loss_task, tr_steps)
            train_loss_dict.update({epoch_counter: tr_epoch_loss})
            tr_output_loss = np.true_divide(tr_output_loss, tr_steps)
            
            ## performance matrix after each epoch
            tr_epoch_accuracy, tr_epoch_f1_score = utils.get_results_ssl(train_true_task, np.asarray(train_pred_task, int))
            tr_ssl_result = utils.write_result(tr_epoch_accuracy, tr_epoch_f1_score, epoch_counter, tr_ssl_result)
            utils.write_summary(loss = tr_epoch_loss, total_loss = tr_output_loss, f1_score = tr_epoch_f1_score, epoch_counter = epoch_counter, isTraining = True, summary_writer = summary_writer)
            utils.write_result_csv(index, epoch_counter, os.path.join(output_folder, "STR_result", "tr_str_f1_Score.csv"), tr_epoch_f1_score)
    
            model_path = os.path.join(model_dir , "epoch_" + str(epoch_counter))
            utils.makedirs(model_path)
            save_path = saver.save(sess, os.path.join(model_path, "SSL_model.ckpt"))
            # print('epoch {}: testing loss is {}'.format(epoch, avg_loss))
            tr_accuracy_avg, tr_f1_score_avg = tr_epoch_accuracy.sum() / single_batch_size, tr_epoch_f1_score.sum() / single_batch_size
            print('【Epoch {} (Self supervised training)】Loss : {:.4f} | Accuracy : {:.4f} | F1 score : {:.4f} '.format(epoch_counter, tr_output_loss, tr_accuracy_avg, tr_f1_score_avg))
            print("Self-supervised trained model is saved in path: %s" % save_path) 
            tr_loss.append(tr_output_loss)
            tr_accuracy.append(tr_accuracy_avg)
            tr_f1_score.append(tr_f1_score_avg)

            ## initialize array
            te_loss_task    = np.zeros((len(transform_task), 1), dtype  = np.float32)
            test_pred_task  = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32)-1
            test_true_task  = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32)-1
            te_output_loss  = 0
           
            te_total_gen_op = utils.make_total_batch(data = test_EGG, 
                                                     length = testing_length, 
                                                     batchsize = batchsize, 
                                                     noise_amount=noise_param, 
                                                     scaling_factor=scale_param, 
                                                     permutation_pieces=permu_param, 
                                                     time_warping_pieces=tw_piece_param, 
                                                     time_warping_stretch_factor= twsf_param, 
                                                     time_warping_squeeze_factor= 1/twsf_param)
            
            print('***** self supervised testing *****')    
            for testing_batch, testing_labels, te_counter, te_steps in te_total_gen_op:
                # print('***** Testing *****')
                ## run the model here 
                fetches = [all_loss, output_loss, y_pred]
                    
                fetched = sess.run(fetches, {input_tensor: testing_batch, y: testing_labels, drop_out: 0.0, isTrain: False})
    
                te_loss_task = utils.fetch_all_loss(fetched[0], te_loss_task)
                te_output_loss += fetched[1]
                test_pred_task = utils.fetch_pred_labels(fetched[2], test_pred_task)
                test_true_task = utils.fetch_true_labels(testing_labels, test_true_task)
    
            ## loss after epoch
            te_epoch_loss = np.true_divide(te_loss_task, te_steps)
            test_loss_dict.update({epoch_counter: te_epoch_loss})
            te_output_loss = np.true_divide(te_output_loss, te_steps)
    
            ## performance matrix after each epoch
            te_epoch_accuracy, te_epoch_f1_score = utils.get_results_ssl(test_true_task, test_pred_task)            
            te_ssl_result = utils.write_result(te_epoch_accuracy, te_epoch_f1_score, epoch_counter, te_ssl_result)    
            utils.write_summary(loss = te_epoch_loss, total_loss = te_output_loss, f1_score = te_epoch_f1_score, epoch_counter = epoch_counter, isTraining = False, summary_writer = summary_writer)
            utils.write_result_csv(index, epoch_counter, os.path.join(output_folder, "STR_result", "te_str_f1_score.csv"), te_epoch_f1_score)
           
            te_accuracy_avg, te_f1_score_avg = te_epoch_accuracy.sum() / single_batch_size, te_epoch_f1_score.sum() / single_batch_size
            print('【Epoch {} (Self supervised testing)】 Loss : {:.4f} | Accuracy : {:.4f} | F1 score : {:.4f} '.format(epoch_counter, te_output_loss, te_accuracy_avg, te_f1_score_avg))
            te_loss.append(te_output_loss)
            te_accuracy.append(te_accuracy_avg)
            te_f1_score.append(te_f1_score_avg)        
            

            print('***** supervised started *****')        
            if 0==1:
                """
                supervised task of self supervised learning
                """
                """  seed """
                
                tr_score_accuracy = []
                tr_score_precision = []
                tr_score_recall = []
                tr_score_f1_score = []
            
                te_score_accuracy = []
                te_score_precision = []
                te_score_recall = []
                te_score_f1_score = []    

                ## training - testing ECG
                x_tr = traindata
                x_te = testdata
                
                ## features extracted from conv layers
                x_tr_feature = utils.extract_feature(x_original = x_tr, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                x_te_feature = utils.extract_feature(x_original = x_te, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)

                model_for_train = supervised_model
                for epoch_counter in tqdm(range(10)):
                ## supervised emotion recognition
                    
                    model_for_train, tr_score, te_score = model.supervised_model_seed(x_tr_feature = x_tr_feature, y_tr = train_label, x_te_feature = x_te_feature, y_te = test_label, identifier = 'seed_egg', kfold = index, result = output_folder, summaries = er_logs, current_time = current_time, model = model_for_train, epoch_super = 10, batch_super=16)        
                    print('Supervised learning epoch:', epoch_counter)

                #     tr_score_accuracy.append(tr_score[0])
                #     tr_score_precision.append(tr_score[1])
                #     tr_score_recall.append(tr_score[2])
                #     tr_score_f1_score.append(tr_score[3])

                #     te_score_accuracy.append(te_score[0])
                #     te_score_precision.append(te_score[1])
                #     te_score_recall.append(te_score[2])
                #     te_score_f1_score.append(te_score[3])
                
                # supervised_tr_score_accuracy.append(tr_score_accuracy)
                # supervised_tr_score_precision.append(tr_score_precision)
                # supervised_tr_score_recall.append(tr_score_recall)
                # supervised_tr_score_f1_score.append(tr_score_f1_score)

                # supervised_te_score_accuracy.append(te_score_accuracy)
                # supervised_te_score_precision.append(te_score_precision)
                # supervised_te_score_recall.append(te_score_recall)
                # supervised_te_score_f1_score.append(te_score_f1_score)  

        supervised_savepath = savepath + '/supervised_score/'
        utils.makedirs(supervised_savepath)

        supervised_tr_score_accuracy_savepath = supervised_savepath + 'supervised_tr_score_accuracy.xls'
        if os.path.exists(supervised_tr_score_accuracy_savepath):
            os.remove(supervised_tr_score_accuracy_savepath)
        output = open(supervised_tr_score_accuracy_savepath, 'w', encoding='gbk')
        for i in range(len(supervised_tr_score_accuracy)):
            for j in range(len(supervised_tr_score_accuracy[i])):
                output.write(str(supervised_tr_score_accuracy[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
                output.write('\t')                                      #相当于Tab一下，换一个单元格
            output.write('\n')                                          #写完一行立马换行
        output.close()

        supervised_tr_score_precision_savepath = supervised_savepath + 'supervised_tr_score_precision.xls'
        if os.path.exists(supervised_tr_score_precision_savepath):
            os.remove(supervised_tr_score_precision_savepath)
        output = open(supervised_tr_score_precision_savepath, 'w', encoding='gbk')
        for i in range(len(supervised_tr_score_precision)):
            for j in range(len(supervised_tr_score_precision[i])):
                output.write(str(supervised_tr_score_precision[i][j]))    
                output.write('\t')                                      
            output.write('\n')                                          
        output.close()

        supervised_tr_score_recall_savepath = supervised_savepath + 'supervised_tr_score_recall.xls'
        if os.path.exists(supervised_tr_score_recall_savepath):
            os.remove(supervised_tr_score_recall_savepath)
        output = open(supervised_tr_score_recall_savepath, 'w', encoding='gbk')
        for i in range(len(supervised_tr_score_recall)):
            for j in range(len(supervised_tr_score_recall[i])):
                output.write(str(supervised_tr_score_recall[i][j]))    
                output.write('\t')                                      
            output.write('\n')                                          
        output.close()

        supervised_tr_score_f1_score_savepath = supervised_savepath + 'supervised_tr_score_f1_score.xls'
        if os.path.exists(supervised_tr_score_f1_score_savepath):
            os.remove(supervised_tr_score_f1_score_savepath)
        output = open(supervised_tr_score_f1_score_savepath, 'w', encoding='gbk')
        for i in range(len(supervised_tr_score_f1_score)):
            for j in range(len(supervised_tr_score_f1_score[i])):
                output.write(str(supervised_tr_score_f1_score[i][j]))    
                output.write('\t')                                      
            output.write('\n')                                          
        output.close()

        supervised_te_score_accuracy_savepath = supervised_savepath + 'supervised_te_score_accuracy.xls'
        if os.path.exists(supervised_te_score_accuracy_savepath):
            os.remove(supervised_te_score_accuracy_savepath)
        output = open(supervised_te_score_accuracy_savepath, 'w', encoding='gbk')
        for i in range(len(supervised_te_score_accuracy)):
            for j in range(len(supervised_te_score_accuracy[i])):
                output.write(str(supervised_te_score_accuracy[i][j]))    
                output.write('\t')                                      
            output.write('\n')                                          
        output.close()
        
        supervised_te_score_precision_savepath = supervised_savepath + 'supervised_te_score_precision.xls'
        if os.path.exists(supervised_te_score_precision_savepath):
            os.remove(supervised_te_score_precision_savepath)
        output = open(supervised_te_score_precision_savepath, 'w', encoding='gbk')
        for i in range(len(supervised_te_score_precision)):
            for j in range(len(supervised_te_score_precision[i])):
                output.write(str(supervised_te_score_precision[i][j]))    
                output.write('\t')                                      
            output.write('\n')                                          
        output.close()

        supervised_te_score_recall_savepath = supervised_savepath + 'supervised_te_score_recall.xls'
        if os.path.exists(supervised_te_score_recall_savepath):
            os.remove(supervised_te_score_recall_savepath)
        output = open(supervised_te_score_recall_savepath, 'w', encoding='gbk')
        for i in range(len(supervised_te_score_recall)):
            for j in range(len(supervised_te_score_recall[i])):
                output.write(str(supervised_te_score_recall[i][j]))    
                output.write('\t')                                      
            output.write('\n')                                          
        output.close()

        supervised_te_score_f1_score_savepath = supervised_savepath + 'supervised_te_score_f1_score.xls'
        if os.path.exists(supervised_te_score_f1_score_savepath):
            os.remove(supervised_te_score_f1_score_savepath)
        output = open(supervised_te_score_f1_score_savepath, 'w', encoding='gbk')
        for i in range(len(supervised_te_score_f1_score)):
            for j in range(len(supervised_te_score_f1_score[i])):
                output.write(str(supervised_te_score_f1_score[i][j]))    
                output.write('\t')                                      
            output.write('\n')                                          
        output.close()

        tr_loss_savepath = savepath + '/tr_loss.xls'
        if os.path.exists(tr_loss_savepath):
            os.remove(tr_loss_savepath)
        output = open(tr_loss_savepath, 'w', encoding='gbk')
        for i in range(len(tr_loss)):
            output.write(str(tr_loss[i]))    
            output.write('\t')                                      
        output.write('\n')                                          
        output.close()

        tr_accuracy_savepath = savepath + '/tr_accuracy.xls'
        if os.path.exists(tr_accuracy_savepath):
            os.remove(tr_accuracy_savepath)
        output = open(tr_accuracy_savepath, 'w', encoding='gbk')
        for i in range(len(tr_accuracy)):
            output.write(str(tr_accuracy[i]))    
            output.write('\t')                                      
        output.write('\n')                                          
        output.close()

        tr_f1_score_savepath = savepath + '/tr_f1_score.xls'
        if os.path.exists(tr_f1_score_savepath):
            os.remove(tr_f1_score_savepath)
        output = open(tr_f1_score_savepath, 'w', encoding='gbk')
        for i in range(len(tr_f1_score)):
            output.write(str(tr_f1_score[i]))    
            output.write('\t')                                      
        output.write('\n')                                          
        output.close()

        te_loss_savepath = savepath + '/te_loss.xls'
        if os.path.exists(te_loss_savepath):
            os.remove(te_loss_savepath)
        output = open(te_loss_savepath, 'w', encoding='gbk')
        for i in range(len(te_loss)):
            output.write(str(te_loss[i]))    
            output.write('\t')                                      
        output.write('\n')                                          
        output.close()

        te_accuracy_savepath = savepath + '/te_accuracy.xls'
        if os.path.exists(te_accuracy_savepath):
            os.remove(te_accuracy_savepath)
        output = open(te_accuracy_savepath, 'w', encoding='gbk')
        for i in range(len(te_accuracy)):
            output.write(str(te_accuracy[i]))    
            output.write('\t')                                      
        output.write('\n')                                          
        output.close()
        
        te_f1_score_savepath = savepath + '/te_f1_score.xls'
        if os.path.exists(te_f1_score_savepath):
            os.remove(te_f1_score_savepath)
        output = open(te_f1_score_savepath, 'w', encoding='gbk')
        for i in range(len(te_f1_score)):
            output.write(str(te_f1_score[i]))    
            output.write('\t')                                      
        output.write('\n')                                          
        output.close()
        
        best_tr_loss.append(max(tr_loss))
        best_tr_accuracy.append(max(tr_accuracy))
        best_tr_f1_score.append(max(tr_f1_score))

        best_te_loss.append(max(te_loss))
        best_te_accuracy.append(max(te_accuracy))
        best_te_f1_score.append(max(te_f1_score))

        best_supervised_tr_score_accuracy.append(max(max(supervised_tr_score_accuracy)))
        best_supervised_tr_score_precision.append(max(max(supervised_tr_score_precision)))
        best_supervised_tr_score_recall.append(max(max(supervised_tr_score_recall)))
        best_supervised_tr_score_f1_score.append(max(max(supervised_tr_score_f1_score)))

        best_supervised_te_score_accuracy.append(max(max(supervised_te_score_accuracy)))
        best_supervised_te_score_precision.append(max(max(supervised_te_score_precision)))
        best_supervised_te_score_recall.append(max(max(supervised_te_score_recall)))
        best_supervised_te_score_f1_score.append(max(max(supervised_te_score_f1_score)))

        ## save str loss, acc and f1 score    
        np.save(tr_ssl_loss_filename, train_loss_dict)
        np.save(te_ssl_loss_filename, test_loss_dict)
    
        np.save(tr_ssl_result_filename, tr_ssl_result)
        np.save(te_ssl_result_filename, te_ssl_result)


best_result = [best_tr_loss, best_tr_accuracy, best_tr_f1_score, best_te_loss, best_te_accuracy, best_te_f1_score, best_supervised_tr_score_accuracy, best_supervised_tr_score_precision, best_supervised_tr_score_recall, best_supervised_tr_score_f1_score, best_supervised_te_score_accuracy, best_supervised_te_score_precision, best_supervised_te_score_recall, best_supervised_te_score_f1_score]
best_savepath = '../record/' + 'best_result.xls'
if os.path.exists(best_savepath):
    os.remove(best_savepath)
output = open(best_savepath, 'w', encoding='gbk')
for i in range(len(best_result)):
    for j in range(len(best_result[i])):
        output.write(str(best_result[i][j]))    
        output.write('\t')                                      
    output.write('\n')                                          
output.close()


print('Tr Loss : {:.4f} | Tr Accuracy : {:.4f} | Tr F1 score : {:.4f} '.format(np.mean(best_tr_loss), np.mean(best_tr_accuracy), np.mean(best_tr_f1_score)))
print('te Loss : {:.4f} | te Accuracy : {:.4f} | te F1 score : {:.4f} '.format(np.mean(best_te_loss), np.mean(best_te_accuracy), np.mean(best_te_f1_score)))
print('Supervised Tr Accuracy : {:.4f} | Supervised Tr Precision : {:.4f} | Supervised Tr Recall : {:.4f} | Supervised Tr F1 score : {:.4f} '.format(np.mean(best_supervised_tr_score_accuracy), np.mean(best_supervised_tr_score_precision), np.mean(best_supervised_tr_score_recall), np.mean(best_supervised_tr_score_f1_score)))
print('Supervised Te Accuracy : {:.4f} | Supervised Te Precision : {:.4f} | Supervised Te Recall : {:.4f} | Supervised Te F1 score : {:.4f} '.format(np.mean(best_supervised_te_score_accuracy), np.mean(best_supervised_te_score_precision), np.mean(best_supervised_te_score_recall), np.mean(best_supervised_te_score_f1_score)))

print('*********  Finish  *********')