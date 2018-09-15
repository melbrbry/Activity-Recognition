import os
import numpy as np
import tensorflow as tf
import utilities as ut
import pickle

class LSTM_Config(object):
    
    def __init__(self):
        self.dropout = 0.75
        self.hidden_dim = 400 
        self.batch_size = 2
        self.lr = 0.001
#        self.frames_per_vid = ut.get_max_frames('./dataset/')
#        self.frames_per_vid = 70
        self.frame_dim = 39
        self.no_of_activities = 10 
        self.no_of_layers = 1
        self.max_no_of_epochs = 20
        self.model_name = "model_keep=%.2f_batch=%d_hidden_dim=%d_layers=%d" % (self.dropout,
                    self.batch_size, self.hidden_dim,
                    self.no_of_layers)
        self.model_dir = "models/%s" % self.model_name

class LSTM_Model(object):

    def __init__(self, config):
        self.config = config
        self.create_model_dirs()
        print("Creating Model Directories PASSED!")
        self.load_utilities_data()
        print ("Loading Utilities Data PASSED!")
        self.add_placeholders()
        print("Adding Placeholders PASSED!")
        self.add_logits()
        print("Adding Logits PASSED!")
        self.add_loss_op()
        print("Adding Loss op PASSED!")
        self.add_training_op()
        print("Adding Trainign op PASSED!")
        self.add_pred_op()
        self.add_accuracy_op()
        
    def create_model_dirs(self):
        if not os.path.exists(self.config.model_dir):
            os.mkdir(self.config.model_dir)
        if not os.path.exists("%s/weights" % self.config.model_dir):
            os.mkdir("%s/weights" % self.config.model_dir)
        if not os.path.exists("%s/losses" % self.config.model_dir):
            os.mkdir("%s/losses" % self.config.model_dir)

    def load_utilities_data(self):
        file = './dataset/train/data'
        with open(file, 'rb') as filehandle:  
            self.train_data = pickle.load(filehandle)
        file = './dataset/train/labels'
        with open(file, 'rb') as filehandle:  
            self.train_labels = pickle.load(filehandle)                         
        file = './dataset/val/data'
        with open(file, 'rb') as filehandle:  
            self.val_data = pickle.load(filehandle)
        file = './dataset/val/labels'
        with open(file, 'rb') as filehandle:  
            self.val_labels = pickle.load(filehandle)                         

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
        
    def create_train_feed_dict(self, input_batch, labels_batch, dropout):
        feed_dict = {}
        feed_dict[self.input_ph] = input_batch
        feed_dict[self.labels_ph] = labels_batch
        feed_dict[self.dropout_ph] = dropout        
        return feed_dict 
   
    def create_pred_feed_dict(self, val_data_inst, dropout):
        feed_dict = {}
        feed_dict[self.input_ph] = [val_data_inst]
        feed_dict[self.dropout_ph] = dropout        
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
                    self.input_ph, initial_state=initial_state)
        output = outputs[:,-1,:] 
        
#        print("outputs", outputs, "output", output)
        output = tf.reshape(output, [-1, self.config.hidden_dim])

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
#        self.predict_op = self.logits
        
    def add_accuracy_op(self):
        self.accuracy_op = tf.metrics.accuracy(self.labels_ph, self.predictions_ph)
        
    def run_epoch(self, session):
        batch_losses = []
        for step, (data_batch, labels_batch) in enumerate(ut.train_data_iterator(self)):
#            print(step)
#            print("data batch \n", [i+1 for i in range(len(data._batch[0][0])) if data_batch[0][0][i]==1])
#            print("labels batch \n", labels_batch[0])
            feed_dict = self.create_train_feed_dict(data_batch,
                                              labels_batch, self.config.dropout)
            
            batch_loss, _ = session.run([self.loss, self.train_op],
                                        feed_dict=feed_dict)
            batch_losses.append(batch_loss)
        return batch_losses
 
    def predict_activities(self, session, val_data_inst):
        feed_dict = self.create_pred_feed_dict(val_data_inst, self.config.dropout)
        prediction = session.run(self.predict_op, feed_dict=feed_dict)
        return prediction
    
    def calc_accuracy_on_val(self, session):
        val_data = np.array(self.val_data)
        val_labels = np.array(self.val_labels)
        print("true labels ", val_labels)
        predictions = []
        for val_data_inst in val_data:
            prediction = self.predict_activities(session, val_data_inst)
            predictions.append(list(prediction[0]))
        print("predictions ", list(predictions))
        feed_dict = self.create_acc_feed_dict(predictions, val_labels)
        accuracy = session.run(self.accuracy_op, feed_dict=feed_dict)
#        accuracy = 0
        return accuracy
def main():    
    tf.reset_default_graph()
    config = LSTM_Config()
    print("Model Configure PASSED!")
    model = LSTM_Model(config)
    print("Model Create PASSED!")    
#    saver = tf.train.Saver(max_to_keep=model.config.max_no_of_epochs)
    
    with tf.Session() as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)

        for epoch in range(config.max_no_of_epochs):
            print ("epoch: %d/%d" % (epoch, config.max_no_of_epochs-1))
            batch_losses = model.run_epoch(sess)
#            print(batch_losses)
            epoch_loss = np.mean(batch_losses)
            print("loss:", epoch_loss)
            val_accuracy = model.calc_accuracy_on_val(sess)
            print("val accuracy:", val_accuracy)
#            if epoch%10 == 0:
#                saver.save(sess, "%s/weights/model" % model.config.model_dir,
#                            global_step=epoch)

if __name__ == '__main__':
    main()
