#!/usr/bin/python3

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from sklearn import metrics

tf.enable_eager_execution()
tf.compat.v1.set_random_seed(0)



class Model_Base():
    """
    A neural net base class, containing functions for training
    """
    def __init__(self):
        self.layers = []
        self.checkpoint = None


    def save(self, model_name):
        location = './ckpt_'+model_name+'/'
        if(not hasattr(self,'checkpoint')):
            #create dict of variables
            vars_list = [(str(i),var) for i,var in enumerate(self.trainable_variables)]
            vars_dict = dict(vars_list)
            self.checkpoint = tf.train.Checkpoint(opt=self.opt, **vars_dict)
            self.manager = tf.train.CheckpointManager(self.checkpoint, location,max_to_keep=3)

        self.manager.save()
        print(f"Checkpoint {self.checkpoint.save_counter.numpy()} saved")


    def load(self, model_name):
        location = './ckpt_'+model_name+'/'
        if(not hasattr(self,'checkpoint')):
            #create dict of variables
            vars_list = [(str(i),var) for i,var in enumerate(self.trainable_variables)]
            vars_dict = dict(vars_list)
            self.checkpoint = tf.train.Checkpoint(opt=self.opt, **vars_dict)
            self.manager = tf.train.CheckpointManager(self.checkpoint, location,max_to_keep=3)
        self.checkpoint.restore(self.manager.latest_checkpoint)
        if(self.manager.latest_checkpoint):
            print(f"Checkpoint restored from {self.manager.latest_checkpoint}")
        else:
            print("Initialized model from scratch")

    @property
    def trainable_variables(self):
        if(len(self.layers) == 0):
            raise NotImplementedError(
                    "No layers in 'self.layers', you should have put them there in __init__"
                    )
        trainable_variables = []
        for l in self.layers:
            trainable_variables += l.trainable_variables
        return trainable_variables

    def train(self, X, Y, num_epochs=10, batch_size=128,step_size=0.001):
        """
        Batch SGD
        Yields control back every epoch, returning the average loss across batches
        """
        assert(hasattr(self,'opt')) # you haven't set up an optimizer

        N = tf.shape(X)[0]

        for i in range(num_epochs):
            loss_history = []
            for batch in range(tf.math.ceil(N/batch_size)):
                start = batch*batch_size
                end = min((batch+1)*batch_size, N)

                with tf.GradientTape() as tape:
                    loss = self.loss(X[start:end],Y[start:end])

                loss_history.append(tf.reduce_sum(loss).numpy())

                weights = self.trainable_variables
                gradients = tape.gradient(loss, weights)
                self.opt.apply_gradients(zip(gradients, weights))

            yield np.mean(loss_history)

    def loss(self, X, Y):
        raise NotImplementedError()
    def predict(self, X):
        raise NotImplementedError()
    def batch_predict(self, X, batch_size=128):
        ''' for doing predictions without killing GPU memory '''
        N = tf.shape(X)[0]
        prediction_history = []
        for batch in range(tf.math.ceil(N/batch_size)):
            start = batch*batch_size
            end = min((batch+1)*batch_size, N)
            predictions = self.predict(X[start:end])
            prediction_history.append(predictions)
        return tf.concat(prediction_history, axis=0)


class Layer():
    def build(self, *args, **kwargs):
        # Must create trainable_variables attribute
        raise NotImplementedError()
    def call(self, *args, **kwargs):
        raise NotImplementedError()
    def __call__(self, *args, **kwargs):    
        try:
            self.output_shape
        except AttributeError:
            self.build(*args, **kwargs)
        return self.call(*args,**kwargs)

class Conv2D(Layer):
    ''' 
    A wrapper for tf.nn.conv2d. Takes a tensor of shape 
    [batch, height, width, channels] and outputs a tensor of shape
    [batch, out_height, out_width, out_channels]
    '''
    def build(self, x, filter_shape=[9,9], output_channels=5, strides=1):
        input_shape = x.shape
        self.strides=strides
        weights_shape=[filter_shape[0],filter_shape[1],input_shape[-1],output_channels]
        #  self.w = tf.Variable(tf.random.uniform(),dtype=tf.float32)
        #  self.w = (self.w - 0.5)/100
        self.w = tf.Variable(
                tf.random.truncated_normal(weights_shape)/10000.,
                dtype=tf.float32
                )

        #set batch size to 1
        input_shape = list(input_shape)
        self.output_shape = [-1, 
                            ((input_shape[1]-filter_shape[0])//strides)+1,
                            ((input_shape[2]-filter_shape[1])//strides)+1,
                            output_channels,
                            ]
        self.trainable_variables = [self.w]


    def call(self, x, filter_shape=[9,9], output_channels=5, strides=1):
        return tf.nn.conv2d(x, self.w, self.strides, padding="VALID")

class Dense(Layer):
    def build(self, x, output_size=10):
        input_shape = x.shape
        assert(len(input_shape) == 2)
        self.w = tf.Variable(
                tf.random.truncated_normal((input_shape[-1],output_size))/10000.,
                dtype=tf.float32
                )
        self.b = tf.Variable(tf.random.truncated_normal([output_size])/10000.,dtype=tf.float32)

        self.output_shape = (-1,output_size)
        self.trainable_variables = [self.w, self.b]

    def call(self, x, output_size=10):
        return x@self.w + self.b

class DenseCaps(Layer):
    '''
    Example usage:
    d = denseCaps(x, out_caps=8, out_atoms=5, routing_iterations=2)

    Takes a tensor x of shape [batch, in_capsules, in_atoms],
    where in_atoms is the sive of the vector that makes up each capsule in the previous 
    layer.
    Outputs a tensor d of shape [batch, out_capsules, out_atoms]
    Contains weights of shape [in_capsules, in_atoms, out_capsules, out_dims]
    '''
    def build(self, x, out_caps=5, out_atoms=4,routing_iterations=3):
        input_shape = x.shape
        # initialise weights
        weight_shape = (input_shape[1],input_shape[2],out_caps,out_atoms)
        self.w = tf.Variable(
                tf.random.truncated_normal(weight_shape)/10000.,
                dtype=tf.float32)

        self.routing_iterations = routing_iterations
        self.in_caps = input_shape[1]
        self.out_caps = out_caps
        
        # set trainable_variables and output_shape
        self.trainable_variables = [self.w]
        self.output_shape = (-1,out_caps,out_atoms)


    def call(self, x, out_caps=5, out_atoms=4,routing_iterations=3):
        batch_size = tf.shape(x)[0]
        # initialise routing weights b
        b = tf.Variable(tf.zeros((batch_size,self.in_caps,self.out_caps,1)),dtype=tf.float32)

        for r in range(self.routing_iterations):
            # get softmax over b, called c
            c = tf.nn.softmax(b,axis=-2)
            # get votes of shape [batch, in_caps, out_caps, out_atoms], called $u$ in paper
            votes = tf.einsum('abc,bcde->abde', x, self.w)
            # sum (axis 2) over votes*expand_dims(c)
            s = tf.reduce_sum(votes*c,axis=1)
            # squash, currently of shape [batch, out_caps, out_atoms]
            squashed = self.squash(s)
            if(r < self.routing_iterations-1):
                # dot with votes
                reshaped_squashed = tf.expand_dims(squashed,1)
                a = tf.reduce_sum(reshaped_squashed*votes,axis=-1)
                # add to b, updating routing weights
                tf.compat.v1.assign_add(b, tf.expand_dims(a,-1))

        return squashed

    def squash(self, x):
        # [batch, ..., capsules, atoms] -> same shape, but if you norm along the atoms axis
        # it's now always less than one
        norms = tf.expand_dims(tf.linalg.norm(x,axis=-1),-1)
        squash = norms**2/(1+norms**2)*(x/norms)
        return squash

#TODO assemble this class
class CapsNet(Model_Base):
    '''
    Takes an MNIST batch of shape [batch, 28, 28] and 
    outputs softmax prediction of shape [batch, 10]
    '''
    def __init__(self, learning_rate=0.001):
        # initialise layers
        self.opt = tf.train.AdamOptimizer(learning_rate)

        self.conv1 = Conv2D()
        self.conv2 = Conv2D()
        self.capsules = DenseCaps()

        self.dense1 = Dense()
        self.dense2 = Dense()
        self.dense3 = Dense()

        self.layers = [self.conv1, self.conv2, self.capsules,
                        self.dense1, self.dense2, self.dense3]

        x = np.zeros((1,28,28),dtype=np.float32)
        y = np.zeros((1,10),dtype=np.float32)
        self.loss(x,y)


    def model(self, X, training=False):
        X = tf.expand_dims(X,-1)
        # X is now [batch, h, w, channels]
        h = tf.nn.relu(self.conv1(X, (9,9), 256))
        h2 = tf.nn.relu(self.conv2(h, filter_shape=(9,9), output_channels=32*8,strides=2))
        s = self.conv2.output_shape
        h2 = tf.reshape(h2, (-1,s[1]*s[2]*32,8))
        # shape is [batch, h*w*caps, atoms]
        h3 = self.capsules(h2, out_caps=10, out_atoms=16)
        # shape is [batch, 10, 16]
        return h3

    def reconstruction(self, h3, X, Y):
        #mask all except correct capsule
        # h3 [batch, 10, 16]
        # Y [batch, 10]
        Y = tf.expand_dims(Y, -1)
        # Y [batch, 10, 1]
        masked = h3*Y
        masked = tf.reshape(masked, (-1, 10*16))
        h1 = self.dense1(masked, 512)
        h2 = self.dense2(h1, 1024)
        prediction = self.dense3(h2, 784)
        # shape [batch, 784]
        X = tf.reshape(X, (-1, 784))
        return tf.reduce_sum((X-prediction)**2, axis=-1)
        
    def predict(self, X):
        h3 = self.model(X)
        digit_cap_lengths = tf.linalg.norm(h3, axis=-1)
        return digit_cap_lengths

    def loss(self, X, Y, training=True):
        h3 = self.model(X, training=training)
        digit_cap_lengths = tf.linalg.norm(h3, axis=-1)
        reconstruction_loss = self.reconstruction(h3, X, Y)
        margin_loss = Y*tf.nn.relu(0.9-digit_cap_lengths)**2 + \
                0.5*(1-Y)*tf.nn.relu(digit_cap_lengths-0.1)**2
        margin_loss = tf.reduce_sum(margin_loss, axis=-1)
        total_loss = margin_loss + 0.0005*reconstruction_loss
        return total_loss
        

class ConvNet(Model_Base):
    '''
    Takes an MNIST batch of shape [batch, 28, 28] and 
    outputs softmax prediction of shape [batch, 10]
    '''
    def __init__(self,learning_rate=0.001):

        self.opt = tf.train.AdamOptimizer(learning_rate)

        # initialise layers
        self.conv1 = Conv2D()
        self.conv2 = Conv2D()
        self.dense = Dense()

        self.layers = [self.conv1, self.conv2, self.dense]

        # Cheap hack to make sure the model is fully initialized
        # before the first time it is run. Hopefully won't be necessary 
        # after bugs are ironed out of tf2.0
        x = np.zeros((1,28,28),dtype=np.float32)
        y = np.zeros((1,10),dtype=np.float32)
        self.loss(x,y)


    def model(self, X, training=False):
        X = tf.expand_dims(X,-1)
        # X is now [batch, h, w, channels]
        h = tf.nn.relu(self.conv1(X, (7,7), 50))
        h2 = tf.nn.relu(self.conv2(h, (7,7), 30))
        s = self.conv2.output_shape
        new_shape = (-1, s[1]*s[2]*s[3])
        h3 = tf.reshape(h2, new_shape)
        logits = self.dense(h3,10)
        return logits

    def predict(self, X):
        logits = self.model(X)
        return tf.nn.softmax(logits)

    def loss(self, X, Y, training=True):
        logits = self.model(X, training=training)
        return tf.losses.softmax_cross_entropy(Y,logits)


def main():
    (X, Y), (X_test, Y_test) = mnist.load_data()

    Y_test = tf.one_hot(Y_test, 10, 1.0, 0.0)
    X_test = tf.cast(X_test,tf.float32)/255.
    Y = tf.one_hot(Y, 10, 1.0, 0.0)
    X = tf.cast(X,tf.float32)/255.
    
    # Split validation set off of training set
    # Training size:    50,000
    # Validation size:  10,000
    # Test size:        10,000
    val_size = len(Y_test)
    Y_val = Y[0:val_size,:]
    X_val = X[0:val_size:,:,:]
    X = X[val_size:,:,:]
    Y = Y[val_size:,:]
    
    model_name = 'caps1'
    model = CapsNet(learning_rate=0.003)
    model.load(model_name)
    for epoch,loss in enumerate(model.train(X,Y,num_epochs=20,batch_size=64)):
        print(f"Epoch {epoch}")
        print(f"Training Loss: {loss}")

        testPredictions = model.batch_predict(X_val)
        accuracy = get_accuracy(testPredictions, Y_val)
        print(f"Validation Accuracy: {accuracy*100:.2F}%")
        model.save(model_name)
        print()
    get_accuracy(testPredictions, Y_val, True)
    
    
    
def get_accuracy(predictions, y, confusion=False):
    y_true = np.argmax(y.numpy(),axis=-1)
    y_pred = np.argmax(predictions,axis=-1)
    accuracy = np.sum(np.equal(y_pred, y_true))/len(y_true)
    if confusion:
        print(metrics.confusion_matrix(y_true, y_pred))
    return accuracy


'''
Takes batch of mnist images a numpy array, stacked along the first axis, and displays them
Expected shape of batch_of_images is [num_images, 784]
'''
def display_images(batch_of_images):
    if(len(batch_of_images.shape) == 1):
        batch_of_images = np.expand_dims(batch_of_images, 0)
    assert(len(batch_of_images.shape) == 2)
    assert(batch_of_images.shape[1] == 784)
    batch_of_images = np.reshape(batch_of_images, [-1, 28, 28])

    num_images = batch_of_images.shape[0]
    row_size = np.ceil(np.sqrt(num_images))
    plt.figure(1)
    plt.title('MNIST')
    for i in range(num_images):
        plt.subplot(row_size,row_size, i+1)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.imshow(batch_of_images[i], cmap='gray')
    plt.show()



if __name__ == "__main__":
    main()

