# This code is heavily based on Jan Mentzen's implementation of a VAE (https://jmetzen.github.io/2015-11-27/vae.html)

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

# define some useful functions

def init_xavier(n_in,n_out):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer()
    variable = tf.Variable(initializer(shape=[n_in,n_out]))
    return variable

def cost_mat(X,Y,N,M):
    XX = tf.reduce_sum(tf.multiply(X,X),axis=1)
    YY = tf.reduce_sum(tf.multiply(Y,Y),axis=1)
    C1 = tf.transpose(tf.reshape(tf.tile(XX,[M]),[M,N]))
    C2 = tf.reshape(tf.tile(YY,[N]),[N,M])
    C3 = tf.transpose(tf.matmul(Y,tf.transpose(X)))
    C = C1 + C2 - 2*C3;
    return C

def K_tild(u,v,C,N,M,epsilon):
    C_tild = C - tf.transpose(tf.reshape(tf.tile(u[:,0],[M]),[M,N])) - tf.reshape(tf.tile(v[:,0],[N]),[N,M])
    K_tild = tf.exp(-C_tild/epsilon)
    return K_tild

def sinkhorn_step_log(j,u,v,C, N,M,epsilon,Lambda = 1):
    mu = tf.cast(1/N, tf.float32)
    nu = tf.cast(1/M, tf.float32)
    Ku = tf.reshape( tf.reduce_sum(K_tild(u,v,C,N,M,epsilon),axis = 1) ,[N,1] )
    u = Lambda * ( epsilon*(tf.log(mu) - tf.log(Ku +10**(-6))) + u )
    Kv = tf.reshape( tf.reduce_sum(K_tild(u,v,C,N,M,epsilon),axis = 0), [M,1] )
    v = Lambda * ( epsilon*(tf.log(nu) - tf.log(Kv +10**(-6))) + v )
    j += 1
    return j,u,v,C,N,M,epsilon

def sinkhorn_loss(X,Y):
    epsilon = tf.constant(1.) # smoothing sinkhorn
    Lambda = tf.constant(1.) # unbalanced parameter
    k = tf.constant(50) # number of iterations for sinkhorn
    N = tf.shape(X)[0] # sample size from mu_theta
    M = tf.shape(Y)[0] # sample size from \hat nu
    D = tf.shape(Y)[1] # dimension of the obervation space
    C = cost_mat(X,Y,N,M)
    K = tf.exp(-C/epsilon)
    #sinkhorn iterations
    j0 = tf.constant(0)
    u0 = tf.zeros([N,1])
    v0 = tf.zeros([M,1])
    cond_iter = lambda j, u, v, C, N, M, epsilon: j < k
    j,u,v,C,N,M,epsilon = tf.while_loop(
    cond_iter, sinkhorn_step_log, loop_vars=[j0, u0, v0,C, N,M,epsilon])
    gamma_log = K_tild(u,v,C,N,M,epsilon)
    final_cost = tf.reduce_sum(gamma_log*C)
    return final_cost
    


# Variational Autoencoder class


class VariationalAutoencoder(object):
    
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [batch_size, network_architecture["n_input"]])
      
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        self.network_weights = self._initialize_weights(**self.network_architecture)

        # Draw one sample z from uniform in latent space
        n_z = self.network_architecture["n_z"]
        self.z = tf.random_uniform((self.batch_size, n_z), dtype=tf.float32)
        
        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr =   self._generator_network(self.network_weights["weights_gener"],
                                    self.network_weights["biases_gener"])
    
    def _initialize_weights(self, n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_gener'] = {
            'h1': init_xavier(n_z, n_hidden_gener_1),
            'h2': init_xavier(n_hidden_gener_1, n_hidden_gener_2),
            'out_var': init_xavier(n_hidden_gener_2, n_input)}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_var': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights        
   

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_reconstr = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_var']), 
                                 biases['out_var']))
        return x_reconstr
            
    def _create_loss_optimizer(self):
        # Sinkhorn loss
        self.cost = sinkhorn_loss(self.x, self.x_reconstr)   # average over batch
        # Use ADAM optimizer
        self.optimizer =             tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    
   
    def generate(self, z_sample):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        
        zz = tf.placeholder(tf.float32, [1, network_architecture["n_z"]])

        
        weights = self.network_weights["weights_gener"]
        biases = self.network_weights["biases_gener"]
        
        layer_1 = self.transfer_fct(tf.add(tf.matmul(zz, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_reconstr = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_var']), 
                                 biases['out_var']))
        
        return self.sess.run(x_reconstr,feed_dict={zz: np.reshape(z_sample,[1,network_architecture["n_z"]])})
    


# Training

def train(network_architecture, learning_rate=0.005,
          batch_size=300, training_epochs=10, display_step=5):
    print('Compiling...')
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Training cycle
    print('Training...')

    for epoch in range(training_epochs):
        print(epoch)
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", "{:.9f}".format(avg_cost))
    return vae



# training the model

network_architecture =     dict(n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=2)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=70)




# Visualizing manifold in 2D

print('Generating images...')

nx = ny = 15
x_values = np.linspace(0, 1, nx)
y_values = np.linspace(0, 1, ny)

canvas = np.empty((28*ny, 28*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([xi, yi])
        x_mean = vae.generate(z_mu)
        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

plt.figure(figsize=(8, 10))        
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.tight_layout()
plt.savefig("img/manifold.png", bbox_inches="tight")


