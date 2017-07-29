import tensorflow as tf

class VariantNet(object):

    def __init__(self, input_shape = (15, 4, 3),
                       output_shape = (4, 5, 2, 6 ),
                       kernel_size1 = (2, 4),
                       kernel_size2 = (2, 4),
                       poll_size1 = (2, 1),
                       poll_size2 = (2, 1),
                       filter_num = 48,
                       hidden_layer_unit_number = 96):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.poll_size1 = poll_size1
        self.poll_size2 = poll_size2
        self.filter_num = filter_num
        self.hidden_layer_unit_number = hidden_layer_unit_number
        self.g = tf.Graph()
        self._build_graph()
        self.session = tf.Session(graph = self.g)


    def _build_graph(self):
        with self.g.as_default():
            X_in = tf.placeholder(tf.float32, [None, self.input_shape[0],
                                                     self.input_shape[1], 
                                                     self.input_shape[2]])

            Y_out = tf.placeholder(tf.float32, [None, sum(self.output_shape)])

            self.X_in = X_in
            self.Y_out = Y_out

            conv1 = tf.layers.conv2d(
                inputs=X_in,
                filters=self.filter_num,
                kernel_size=self.kernel_size1,
                padding="same",
                activation=tf.nn.elu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=self.poll_size1, strides=1)

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=self.filter_num,
                kernel_size=self.kernel_size2,
                padding="same",
                activation=tf.nn.elu)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=self.poll_size2, strides=1)

            flat_size = ( 15 - (self.poll_size1[0] - 1) - (self.poll_size2[0] - 1))
            flat_size *= ( 4 - (self.poll_size1[1] - 1) - (self.poll_size2[1] - 1))
            flat_size *= self.filter_num

            conv2_flat =  tf.reshape(pool2, [-1,  flat_size])

            unit_num = self.hidden_layer_unit_number
            h1 = tf.layers.dense(inputs=conv2_flat, units=unit_num, activation=tf.nn.elu)
            dropout1 = tf.layers.dropout(inputs=h1, rate=0.50)

            h2 = tf.layers.dense(inputs=dropout1, units=unit_num, activation=tf.nn.elu)
            dropout2 = tf.layers.dropout(inputs=h2, rate=0.50)

            h3 = tf.layers.dense(inputs=dropout2, units=unit_num, activation=tf.nn.elu)
            dropout3 = tf.layers.dropout(inputs=h3, rate=0.50)

            h4 = tf.layers.dense(inputs=dropout3, units=unit_num, activation=tf.nn.elu)
            dropout4 = tf.layers.dropout(inputs=h4, rate=0.50)

            Y1_ = tf.layers.dense(inputs=dropout4, units=self.output_shape[0], activation=tf.nn.elu)
            Y2_ = tf.layers.dense(inputs=dropout4, units=self.output_shape[1], activation=tf.nn.elu)
            Y3_ = tf.layers.dense(inputs=dropout4, units=self.output_shape[2], activation=tf.nn.elu)
            Y4_ = tf.layers.dense(inputs=dropout4, units=self.output_shape[3], activation=tf.nn.elu)

            self.Y1 = tf.nn.softmax(Y1_)
            self.Y2 = tf.nn.softmax(Y2_)
            self.Y3 = tf.nn.softmax(Y3_)
            self.Y4 = tf.nn.softmax(Y4_)

            outs = self.output_shape
            si = 0
            loss1 = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits( logits=Y1_, labels=tf.slice( Y_out, [0, si], [-1, outs[0]] ) ) )
            si += outs[0]
            loss2 = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits( logits=Y2_, labels=tf.slice( Y_out, [0, si], [-1, outs[1]] ) ) )
            si += outs[1]
            loss3 = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits( logits=Y3_, labels=tf.slice( Y_out, [0, si], [-1, outs[2]] ) ) )
            si += outs[2]
            loss4 = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits( logits=Y4_, labels=tf.slice( Y_out, [0, si], [-1, outs[3]] ) ) )
            self.loss = loss1 + loss2 + loss3 + loss4
             
            learning_rate = 0.00025
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(self.loss)
            
            self.training_op = training_op
            self.init_op = tf.global_variables_initializer()

    def init(self):
        self.session.run( self.init_op )

    def close(self):
        self.session.close()

    def train(self, batchX, batchY):
        loss = 0
        X_in = self.X_in
        Y_out = self.Y_out
        loss, _ = self.session.run( (self.loss, self.training_op), feed_dict={X_in:batchX, Y_out:batchY})    
        return loss

    def get_loss(self, batchX, batchY):
        loss = 0
        X_in = self.X_in
        Y_out = self.Y_out
        loss  = self.session.run( self.loss, feed_dict={X_in:batchX, Y_out:batchY})    
        return loss

    def save_parameters(self, fn):
        with self.g.as_default():
            self.saver = tf.train.Saver()
            self.saver.save(self.session, fn)

    def restore_parameters(self, fn):
        with self.g.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, fn)

    def predict(self, Xarray):
        with self.g.as_default():
            Y1, Y2, Y3, Y4  = self.session.run( (self.Y1, self.Y2, self.Y3, self.Y4), feed_dict={self.X_in:Xarray})
            return Y1, Y2, Y3, Y4

    def __del__(self):
        self.session.close()

