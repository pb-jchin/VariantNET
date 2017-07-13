import tensorflow as tf

class VariantNet(object):

    def __init__(self, input_shape = (15, 4, 3),
                       output_shape1 = (4, ),
                       output_shape2 = (4, ),
                       kernel_size1 = (2, 4),
                       kernel_size2 = (3, 4),
		       poll_size1 = (7, 1),
                       poll_size2 = (3, 1),
                       filter_num = 48,
                       hidden_layer_unit_number = 48):
        self.input_shape = input_shape
        self.output_shape1 = output_shape1
        self.output_shape2 = output_shape2
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

	    Y_out = tf.placeholder(tf.float32, [None, self.output_shape1[0] + self.output_shape2[0]])

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


	    Y1 = tf.layers.dense(inputs=dropout3, units=self.output_shape1[0], activation=tf.nn.sigmoid)
	    Y2 = tf.layers.dense(inputs=dropout3, units=self.output_shape2[0], activation=tf.nn.elu)
	    Y3 = tf.nn.softmax(Y2)

            self.Y1 = Y1
            self.Y3 = Y3

	    loss = tf.reduce_sum(  tf.pow( Y1 - tf.slice(Y_out,[0,0],[-1,self.output_shape1[0]] ), 2) )  +\
		   tf.reduce_sum(  tf.nn.softmax_cross_entropy_with_logits( logits=Y2,  
									    labels=tf.slice( Y_out, [0,self.output_shape1[0]],
                                                                                                    [-1,self.output_shape2[0]] ) ) ) 
            self.loss = loss
	     
	    learning_rate = 0.0005
	    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	    training_op = optimizer.minimize(loss)
            
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
            bases_, type_  = self.session.run( (self.Y1, self.Y3), feed_dict={self.X_in:Xarray})
            return bases_, type_

    def __del__(self):
        self.session.close()

