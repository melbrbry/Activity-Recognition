import os
import numpy as np
import tensorflow as tf
import utilities as ut
import pickle
import random
from tensorflow import set_random_seed

set_random_seed(1)
random.seed(2)
np.random.seed(3)
os.environ['PYTHONHASHSEED'] = '0'

class LSTM_Config(object):
    
    def __init__(self):
        self.dropout = 0.8
        self.hidden_dim = 400
        self.batch_size = 256
        self.lr = 0.001
        self.frame_dim = 39 * 8
        self.no_of_activities = 10
        self.no_of_layers = 1
        self.min_no_of_epochs = 20
        self.max_no_of_epochs = 100
        self.model_name = "model_keep=%.2f_batch=%d_hidden_dim=%d_layers=%d" % (self.dropout,
                    self.batch_size, self.hidden_dim,
                    self.no_of_layers)
        self.model_dir = "models/%s" % self.model_name
        self.patience = 10

class LSTM_Model(object):

    def __init__(self, config):
        self.config = config
        self.create_model_dirs()
        self.load_utilities_data()
        self.add_placeholders()
        self.add_logits()
        self.add_loss_op()
        self.add_training_op()
        self.add_pred_op()
        self.add_accuracy_op()
        
    def create_model_dirs(self):
        if not os.path.exists(self.config.model_dir):
            os.mkdir(self.config.model_dir)
        if not os.path.exists("%s/weights" % self.config.model_dir):
            os.mkdir("%s/weights" % self.config.model_dir)
        if not os.path.exists("%s/losses" % self.config.model_dir):
            os.mkdir("%s/losses" % self.config.model_dir)
        if not os.path.exists("%s/metrics" % self.config.model_dir):
            os.mkdir("%s/metrics" % self.config.model_dir)
        if not os.path.exists("%s/plots" % self.config.model_dir):
            os.mkdir("%s/plots" % self.config.model_dir)
            
    def load_utilities_data(self):
        file = './JSON dataset/train/data'
        with open(file, 'rb') as filehandle:  
            data = pickle.load(filehandle)
#            scaler = Normalizer().fit(data)
#            normalized_data = scaler.transform(data)
#            self.train_data = normalized_data 
            self.train_data = data 

        file = './JSON dataset/train/labels'
        with open(file, 'rb') as filehandle:  
            self.train_labels = pickle.load(filehandle)                         
        file = './JSON dataset/val/data'
        with open(file, 'rb') as filehandle:  
            data = pickle.load(filehandle)
#            scaler = Normalizer().fit(data)
#            normalized_data = scaler.transform(data)
#            self.val_data = normalized_data
            self.val_data = data
        file = './JSON dataset/val/labels'
        with open(file, 'rb') as filehandle:  
            self.val_labels = pickle.load(filehandle) 
        file = './JSON dataset/test/data'
        with open(file, 'rb') as filehandle:  
            data = pickle.load(filehandle)
#            scaler = Normalizer().fit(data)
#            normalized_data = scaler.transform(data)
#            self.test_data = normalized_data
            self.test_data = data
        file = './JSON dataset/test/labels'
        with open(file, 'rb') as filehandle:  
            self.test_labels = pickle.load(filehandle)                         

    def add_placeholders(self):
        self.dropout_ph = tf.placeholder(tf.float32, name="dropout_ph") 
        self.labels_ph = tf.placeholder(tf.float32,
                                        shape=[None, self.config.no_of_activities],
                                        name="labels_ph") 
        self.input_ph =  tf.placeholder(tf.float32,
                                        shape=[None, None, self.config.frame_dim],
                                        name="input_ph")
        self.predictions_ph = tf.placeholder(tf.float32,
                                        shape=[None, self.config.no_of_activities],
                                        name="predictions_ph")
        self.sequence_length_ph =  tf.placeholder(tf.float32,
                                        shape=[None],
                                        name="sequence_length_ph")
        
    def create_feed_dict(self, input_batch, labels_batch, dropout, sequence_length):
        feed_dict = {}
        feed_dict[self.input_ph] = input_batch
        feed_dict[self.labels_ph] = labels_batch
        feed_dict[self.dropout_ph] = dropout
        feed_dict[self.sequence_length_ph] = sequence_length
        return feed_dict 
   
    def create_pred_feed_dict(self, data_inst, dropout, sequence_length):
        feed_dict = {}
        feed_dict[self.input_ph] = np.array([data_inst])
        feed_dict[self.dropout_ph] = dropout        
        feed_dict[self.sequence_length_ph] = sequence_length
        return feed_dict
    
    def create_acc_feed_dict(self, predictions, labels):
        feed_dict = {}
        feed_dict[self.predictions_ph] = predictions
        feed_dict[self.labels_ph] = labels
        return feed_dict
    
    def build_LSTM_cell(self):
        LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim)
        LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(LSTM_cell,
                    input_keep_prob=self.dropout_ph,
                    output_keep_prob=self.dropout_ph)
        return LSTM_cell
    
    def add_logits(self):
        stacked_LSTM_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [self.build_LSTM_cell() for _ in range(self.config.no_of_layers)])
        initial_state = stacked_LSTM_cell.zero_state(tf.shape(self.input_ph)[0],
                    tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(stacked_LSTM_cell,
                    self.input_ph, initial_state=initial_state, sequence_length=self.sequence_length_ph)
        lstm_h = final_state[0].h

        output = tf.reshape(lstm_h, [-1, self.config.hidden_dim])

        with tf.variable_scope("logits"):
            W_logits = tf.get_variable("W_logits",
                        shape=[self.config.hidden_dim,self.config.no_of_activities],
                        initializer=tf.contrib.layers.xavier_initializer())
            b_logits = tf.get_variable("b_logits",
                        shape=[1,self.config.no_of_activities], 
                        initializer=tf.constant_initializer(0))
            self.logits = tf.matmul(output, W_logits) + b_logits

    def add_loss_op(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels_ph))

    def add_training_op(self):    
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        self.train_op = optimizer.minimize(self.loss)
    
    def add_pred_op(self):
        self.predict_op = tf.one_hot(tf.argmax(tf.nn.softmax(self.logits), axis=1), depth=self.config.no_of_activities)
     
    def add_accuracy_op(self):
        self.accuracy_op = tf.contrib.metrics.accuracy(tf.argmax(self.labels_ph,1), tf.argmax(self.predictions_ph, 1))
        
    def run_epoch(self, session):
        batch_losses = []
        for step, (data_batch, labels_batch, sequence_length) in enumerate(ut.train_data_iterator(self)):
            feed_dict = self.create_feed_dict(data_batch,
                                              labels_batch, self.config.dropout, sequence_length)
            batch_loss, _ = session.run([self.loss, self.train_op],
                                        feed_dict=feed_dict)
            batch_losses.append(batch_loss)
        return batch_losses
 
    def predict_activities(self, session, data_inst):
        sequence_length = [len(data_inst)]
        feed_dict = self.create_pred_feed_dict(data_inst, self.config.dropout, sequence_length)
        prediction = session.run(self.predict_op, feed_dict=feed_dict)
        return prediction
    
    def compute_accuracy(self, session, mode='train', no_of_instances=100):
        if mode == 'train':     
            data = np.array(self.train_data)
            labels = np.array(self.train_labels)
        if mode == 'val':
            data = np.array(self.val_data)
            labels = np.array(self.val_labels)
        if mode == 'test':
            data = np.array(self.test_data)
            labels = np.array(self.test_labels)
            no_of_instances=len(data)
        predictions = []
        for step, data_inst in enumerate(data):
            if step < no_of_instances:
                prediction = self.predict_activities(session, data_inst)
                predictions.append(list(prediction[0]))
        feed_dict = self.create_acc_feed_dict(predictions, labels[:no_of_instances])
        accuracy = session.run(self.accuracy_op, feed_dict=feed_dict)
        return accuracy
    
    def early_stopping(self, val_scores, best_val_score, patience):
        val_scores.reverse()
        tolerance = 1e-3
        for step, score in enumerate(val_scores):
            if score - tolerance < best_val_score:
                return False
        return True
            
    def compute_val_loss(self, session):
        val_ids = [i for i in range(len(self.val_data))]
        data, labels, sequence_length = ut.get_batch_ph_data(self, val_ids, "val")
        feed_dict = self.create_feed_dict(data, labels, self.config.dropout, sequence_length)
        val_loss = session.run(self.loss, feed_dict=feed_dict)
        return val_loss
         
def main():    
    tf.reset_default_graph()
    config = LSTM_Config()
    model = LSTM_Model(config)
    saver = tf.train.Saver(max_to_keep=model.config.max_no_of_epochs)
    
    with tf.Session() as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)
        train_epochs_losses, val_epochs_losses, train_accuracies, val_accuracies = [], [], [], []
        best_val_accuracy = 0
        best_acc_loss = float("INF")
        best_val_loss = float("INF")
        for epoch in range(config.max_no_of_epochs):
            print("epoch: %d/%d" % (epoch+1, config.max_no_of_epochs))
            ut.log("epoch: %d/%d" % (epoch+1, config.max_no_of_epochs))
            batch_losses = model.run_epoch(sess)
#            model.config.lr = model.config.initial_lr/(1+(epoch/30))
            epoch_loss = np.mean(batch_losses)
            val_epoch_loss = model.compute_val_loss(sess)
            print("train loss = %f | val loss = %f" % (epoch_loss, val_epoch_loss))
            ut.log("train loss = %f | val loss = %f" % (epoch_loss, val_epoch_loss))
            train_epochs_losses.append(epoch_loss)
            pickle.dump(train_epochs_losses, open("%s/losses/train_epochs_losses"\
                        % model.config.model_dir, "wb"))
            val_epochs_losses.append(val_epoch_loss)
            pickle.dump(val_epochs_losses, open("%s/losses/val_epochs_losses"\
                        % model.config.model_dir, "wb"))
            
            train_accuracy = model.compute_accuracy(sess, mode='train') 
            val_accuracy = model.compute_accuracy(sess, mode='val')
            print("train accuracy = %f | val accuracy = %f" % (train_accuracy, val_accuracy))
            ut.log("train accuracy = %f | val accuracy = %f" % (train_accuracy, val_accuracy))
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            pickle.dump(train_accuracies, open("%s/metrics/train_accuracies"\
                        % model.config.model_dir, "wb"))
            pickle.dump(val_accuracies, open("%s/metrics/val_accuracies"\
                        % model.config.model_dir, "wb"))
            
            if val_accuracy > best_val_accuracy or \
            (val_accuracy == best_val_accuracy and val_epoch_loss < best_acc_loss):
                saver.save(sess, "%s/weights/model" % (model.config.model_dir))
                best_val_accuracy = val_accuracy
                best_acc_loss = val_epoch_loss
            
            best_val_loss = min(val_epoch_loss, best_val_loss)
    
#            if epoch > model.config.min_no_of_epochs and \
#            model.early_stopping(val_epochs_losses, best_val_loss, model.config.patience):
#                break

        saver.restore(sess,  "%s/weights/model" % (model.config.model_dir))
        test_accuracy = model.compute_accuracy(sess, mode='test')
        print("test accuracy = %f" % (test_accuracy))
        ut.log("test accuracy = %f" % (test_accuracy))
        ut.plot_performance(config.model_dir)
        
        
if __name__ == '__main__':
    main()
