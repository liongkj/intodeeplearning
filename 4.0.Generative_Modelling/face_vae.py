import tensorflow as tf

import IPython
import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import mitdeeplearning as mdl

# Get the training data: both images from CelebA and ImageNet
from face_cnn import make_standard_classifier, standard_classifier_probs, keys, test_faces

path_to_training_data = tf.keras.utils.get_file(
    'train_face.h5', 'https://www.dropbox.com/s/l5iqduhe0gwxumq/train_face.h5?dl=1')
# Instantiate a TrainingDatasetLoader using the downloaded dataset
loader = mdl.lab2.TrainingDatasetLoader(path_to_training_data)

### Defining the VAE loss function ###

''' Function to calculate VAE loss given:
      an input x, 
      reconstructed output x_recon, 
      encoded means mu, 
      encoded log of standard deviation logsigma, 
      weight parameter for the latent loss kl_weight
'''


def vae_loss_function(x, x_recon, mu, logsigma, kl_weight=0.0005):
    latent_loss = 0.5 * \
                  tf.reduce_sum(tf.exp(logsigma) + tf.square(mu) -
                                1.0 - logsigma, axis=1)

    reconstruction_loss = tf.reduce_mean(tf.abs(x - x_recon), axis=(1, 2, 3))
    vae_loss = kl_weight * latent_loss + reconstruction_loss

    return vae_loss


### VAE Reparameterization ###

"""Reparameterization trick by sampling from an isotropic unit Gaussian.
# Arguments
    z_mean, z_logsigma (tensor): mean and log of standard deviation of latent distribution (Q(z|X))
# Returns
    z (tensor): sampled latent vector
"""


def sampling(z_mean, z_logsigma):
    # By default, random.normal is "standard" (ie. mean=0 and std=1.0)
    batch, latent_dim = z_mean.shape
    epsilon = tf.random.normal(shape=(batch, latent_dim))

    z = z_mean + tf.math.exp(0.5 * z_logsigma) * epsilon
    return z


### Loss function for DB-VAE ###

"""Loss function for DB-VAE.
# Arguments
    x: true input x
    x_pred: reconstructed x
    y: true label (face or not face)
    y_logit: predicted labels
    mu: mean of latent distribution (Q(z|X))
    logsigma: log of standard deviation of latent distribution (Q(z|X))
# Returns
    total_loss: DB-VAE total loss
    classification_loss = DB-VAE classification loss
"""


def debiasing_loss_function(x, x_pred, y, y_logit, mu, logsigma):
    vae_loss = vae_loss_function(x, x_pred, mu, logsigma)

    classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=y_logit)

    face_indicator = tf.cast(tf.equal(y, 1), tf.float32)

    total_loss = tf.reduce_mean(
        classification_loss +
        face_indicator * vae_loss
    )

    return total_loss, classification_loss


### Define the decoder portion of the DB-VAE ###

n_filters = 12  # base number of convolutional filters, same as standard CNN
latent_dim = 100  # number of latent variables


def make_face_decoder_network():
    # Functionally define the different layer types we will use
    Conv2DTranspose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding='same', activation='relu')
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu')
    Reshape = tf.keras.layers.Reshape

    # Build the decoder network using the Sequential API
    decoder = tf.keras.Sequential([
        # Transform to pre-convolutional generation
        Dense(units=4 * 4 * 6 * n_filters),  # 4x4 feature maps (with 6N occurances)
        Reshape(target_shape=(4, 4, 6 * n_filters)),

        # Upscaling convolutions (inverse of encoder)
        Conv2DTranspose(filters=4 * n_filters, kernel_size=3, strides=2),
        Conv2DTranspose(filters=2 * n_filters, kernel_size=3, strides=2),
        Conv2DTranspose(filters=1 * n_filters, kernel_size=5, strides=2),
        Conv2DTranspose(filters=3, kernel_size=5, strides=2),
    ])

    return decoder


### Defining and creating the DB-VAE ###

class DB_VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(DB_VAE, self).__init__()
        self.latent_dim = latent_dim

        # Define the number of outputs for the encoder. Recall that we have
        # `latent_dim` latent variables, as well as a supervised output for the
        # classification.
        num_encoder_dims = 2 * self.latent_dim + 1

        self.encoder = make_standard_classifier(num_encoder_dims)  # this is the cnn from previous
        self.decoder = make_face_decoder_network()

    # function to feed images into encoder, encode the latent space, and output
    #   classification probability
    def encode(self, x):
        # encoder output
        encoder_output = self.encoder(x)

        # classification prediction
        y_logit = tf.expand_dims(encoder_output[:, 0], -1)
        # latent variable distribution parameters
        z_mean = encoder_output[:, 1:self.latent_dim + 1]
        z_logsigma = encoder_output[:, self.latent_dim + 1:]

        return y_logit, z_mean, z_logsigma

    # VAE reparameterization: given a mean and logsigma, sample latent variables
    def reparameterize(self, z_mean, z_logsigma):
        # TODO: call the sampling function defined above
        z = sampling(z_mean, z_logsigma)
        # z = # TODO
        return z

    # Decode the latent space and output reconstruction
    def decode(self, z):
        # TODO: use the decoder to output the reconstruction
        reconstruction = self.decoder(z)
        # reconstruction = # TODO
        return reconstruction

    # The call function will be used to pass inputs x through the core VAE
    def call(self, x):
        # Encode input to a prediction and latent space
        y_logit, z_mean, z_logsigma = self.encode(x)

        # TODO: reparameterization
        z = self.reparameterize(z_mean, z_logsigma)
        # z = # TODO

        # TODO: reconstruction
        recon = self.decode(z)
        # recon = # TODO
        return y_logit, z_mean, z_logsigma, recon

    # Predict face or not face logit for given input x
    def predict(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)
        return y_logit


dbvae = DB_VAE(latent_dim)


# adaptive resampling for under-represented data
# Function to return the means for an input image batch
# {dbvae}
def get_latent_mu(images, dbvae, batch_size=1024):
    N = images.shape[0]
    mu = np.zeros((N, latent_dim))
    for start_ind in range(0, N, batch_size):
        end_ind = min(start_ind + batch_size, N + 1)
        batch = (images[start_ind:end_ind]).astype(np.float32) / 255.
        _, batch_mu, _ = dbvae.encode(batch)
        mu[start_ind:end_ind] = batch_mu
    return mu


### Resampling algorithm for DB-VAE ###

'''Function that recomputes the sampling probabilities for images within a batch
      based on how they distribute across the training data'''


# {smoothing_fac}: tunes the degree of debiiasing, v = 0: ex
def get_training_sample_probabilities(images, dbvae, bins=10, smoothing_fac=0.001):
    print("Recomputing the sampling probabilities")

    # TODO: run the input batch and get the latent variable means
    mu = get_latent_mu(images, dbvae)

    # sampling probabilities for the images
    training_sample_p = np.zeros(mu.shape[0])

    # consider the distribution for each latent variable
    for i in range(latent_dim):
        latent_distribution = mu[:, i]
        # generate a histogram of the latent distribution
        hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)

        # find which latent bin every data sample falls in
        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')

        # TODO: call the digitize function to find which bins in the latent distribution
        #    every data sample falls in to
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
        bin_idx = np.digitize(latent_distribution, bin_edges)
        # bin_idx = np.digitize('''TODO''', '''TODO''') # TODO

        # smooth the density function
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

        # invert the density function
        p = 1.0 / (hist_smoothed_density[bin_idx - 1])

        # TODO: normalize all probabilities
        p = p / np.sum(p)
        # p = # TODO

        # TODO: update sampling probabilities by considering whether the newly
        #     computed p is greater than the existing sampling probabilities.
        training_sample_p = np.maximum(p, training_sample_p)
        # training_sample_p = # TODO

    # final normalization
    training_sample_p /= np.sum(training_sample_p)

    return training_sample_p


### Training the DB-VAE ###

# Hyperparameters
batch_size = 32
learning_rate = 5e-4
latent_dim = 100

# DB-VAE needs slightly more epochs to train since its more complex than
# the standard classifier so we use 6 instead of 2
num_epochs = 6

# instantiate a new DB-VAE model and optimizer
dbvae = DB_VAE(100)
optimizer = tf.keras.optimizers.Adam(learning_rate)


# To define the training operation, we will use tf.function which is a powerful tool
#   that lets us turn a Python function into a TensorFlow computation graph.
@tf.function
def debiasing_train_step(x, y):
    with tf.GradientTape() as tape:
        # Feed input x into dbvae. Note that this is using the DB_VAE call function!
        y_logit, z_mean, z_logsigma, x_recon = dbvae(x)

        '''TODO: call the DB_VAE loss function to compute the loss'''
        loss, class_loss = debiasing_loss_function(x, x_recon, y, y_logit, z_mean, z_logsigma)
        # loss, class_loss = debiasing_loss_function('''TODO arguments''') # TODO

    '''TODO: use the GradientTape.gradient method to compute the gradients.
       Hint: this is with respect to the trainable_variables of the dbvae.'''
    grads = tape.gradient(loss, dbvae.trainable_variables)
    # grads = tape.gradient('''TODO''', '''TODO''') # TODO

    # apply gradients to variables
    optimizer.apply_gradients(zip(grads, dbvae.trainable_variables))
    return loss


# get training faces from data loader
all_faces = loader.get_all_train_faces()

if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists

# The training loop -- outer loop iterates over the number of epochs
for i in range(num_epochs):

    IPython.display.clear_output(wait=True)
    print("Starting epoch {}/{}".format(i + 1, num_epochs))

    # Recompute data sampling proabilities
    '''TODO: recompute the sampling probabilities for debiasing'''
    p_faces = get_training_sample_probabilities(all_faces, dbvae)
    # p_faces = get_training_sample_probabilities('''TODO''', '''TODO''') # TODO

    # get a batch of training data and compute the training step
    for j in tqdm(range(loader.get_train_size() // batch_size)):
        # load a batch of data
        (x, y) = loader.get_batch(batch_size, p_pos=p_faces)
        # loss optimization
        loss = debiasing_train_step(x, y)

        # plot the progress every 200 steps
        if j % 500 == 0:
            mdl.util.plot_sample(x, y, dbvae)



dbvae_logits = [dbvae.predict(np.array(x, dtype=np.float32)) for x in test_faces]
dbvae_probs = tf.squeeze(tf.sigmoid(dbvae_logits))

xx = np.arange(len(keys))
plt.bar(xx, standard_classifier_probs.numpy().mean(1), width=0.2, label="Standard CNN")
plt.bar(xx + 0.2, dbvae_probs.numpy().mean(1), width=0.2, label="DB-VAE")
plt.xticks(xx, keys);
plt.title("Network predictions on test dataset")
plt.ylabel("Probability")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left");
