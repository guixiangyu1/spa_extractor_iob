import os
import tensorflow as tf
from random import shuffle


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None


    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)  #搜寻scopename范围内的variable
        init = tf.variables_initializer(variables)
        self.sess.run(init)


    def add_train_op(self, lr_method, lr, loss, clip=-1, indicate=None):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive  对梯度进行裁剪，防止梯度爆炸
                if indicate == "train":
                    grads, vs     = zip(*optimizer.compute_gradients(loss, [v for v in tf.trainable_variables() if v.name != "words/_word_embeddings:0"]))
                    grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                elif indicate == "fine_tuning":
                    grads, vs     = zip(*optimizer.compute_gradients(loss,[v for v in tf.trainable_variables() if v.name == "words/_word_embeddings:0"] ))
                    grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                # self.train_op = optimizer.minimize(loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "words/_word_embeddings:0"))
                #var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "words/_word_embeddings:0")
                if indicate == "train":
                    
                    grads = optimizer.compute_gradients(loss, [v for v in tf.trainable_variables() if v.name != "words/_word_embeddings:0"])
                    # self.train_op = optimizer.minimize(loss)
                    # print([v.name for v in tf.trainable_variables()])

                    # opt_vars = [v for v in tf.trainable_variables() if v.name != "words/_word_embeddings:0"]
                    # print([v.name for v in opt_vars])
                    # self.train_op = optimizer.minimize(loss, var_list=opt_vars)

                if indicate == "fine_tuning":
                    print([v.name for v in tf.trainable_variables()])
                    grads = optimizer.compute_gradients(loss, [v for v in tf.trainable_variables() if v.name == "words/_word_embeddings:0"])
                    # opt_vars = [v for v in tf.trainable_variables() if v.name == "words/_word_embeddings:0"]
                    # print(opt_vars)
                    # self.train_op = optimizer.minimize(loss, var_list=opt_vars)

                # elif indicate==None:
                #     grads = optimizer.compute_gradients(loss, [v for v in tf.trainable_variables() if
                #                                                v.name == "words/_word_embeddings:0"])
                self.train_op = optimizer.apply_gradients(grads)



    def initialize_session(self, indicate=None):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # if indicate == "fine_tuning":
        #     vars_restore = [v for v in tf.trainable_variables() if v.name != "words/_word_embeddings:0"]
        #     self.saver = tf.train.Saver(vars_restore)
        # else:
        #     self.saver = tf.train.Saver()
        self.saver = tf.train.Saver()
#         self.saver = tf.train.Saver()

        # variables_names = [v.name for v in tf.all_variables()]
        # values = self.sess.run(variables_names)
        # for k, v in zip(variables_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)

    def restore_session(self, dir_model, indicate=None):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")
        if indicate=="fine_tuning":
            vars_restore = [v for v in tf.trainable_variables() if v.name != "words/_word_embeddings:0"]
            self.saver = tf.train.Saver(vars_restore)
        self.saver.restore(self.sess, dir_model)
        self.saver = tf.train.Saver()




    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)


    def close_session(self):
        """Closes the session"""
        self.sess.close()


    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged      = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                self.sess.graph)


    def train(self, train, dev, test):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score = 0
        nepoch_no_imprv = 0 # for early stopping
        # self.add_summary() # tensorboard

        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                        self.config.nepochs))

            train = list(train)
            shuffle(train)
            score = self.run_epoch(train, dev, epoch)     #f1的值

            self.evaluate(test)

            self.config.lr *= self.config.lr_decay # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                self.logger.info("- new best score!- r")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    break


    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test)   #在子类ner_model中已经实现了该方法
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)
