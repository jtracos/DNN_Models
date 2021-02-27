import tensorflow as tf
from tensorflow.keras import layers, optimizers, models, initializers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfpl = tfp.layers
##
## Defining model parameters
##
encoded_size = 16
base_depth = 32

K.clear_session()

### Creating prior distribution(encoder)
###
prior = tfd.Independent( tfd.Normal(loc = tf.zeros(encoded_size), scale = 1), \
            reinterpreted_batch_dims = 1, name = "prior")

encoder = models.Sequential([ 
    layers.InputLayer(input_shape = input_shape),
    layers.Lambda( lambda x: tf.cast(x,tf.float32) - 0.5),
    layers.Conv2D(base_depth, 5, strides = 1, padding = "same", 
        activation = tf.nn.leaky_relu),
     layers.Conv2D(base_depth, 5, strides = 2, padding = "same", 
         activation = tf.nn.leaky_relu),
     layers.Conv2D(2*base_depth,5 ,strides = 1, padding = "same", 
         activation = tf.nn.leaky_relu),
     layers.Conv2D(2*base_depth,5 ,strides = 2, padding = "same",  
         activation = tf.nn.leaky_relu),
     layers.Conv2D(4*base_depth, 7, strides = 1,\
                    padding = "same", activation = tf.nn.leaky_relu),
     layers.Flatten(),
     layers.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size), 
                activation = None),
     tfpl.MulticariateNormalTriL( encoded_size,
                activity_regulizer = tfpl.LKDivergenceRegularizer(prior)
            ], name = "encoder")

##Creating de Deconvolutional Decoder
decoder = models.Sequential([
    layers.InputLayer(input_shape = [encoded_size]),
    layers.Conv2DTranspose(2*base_depth, 7, strides = 1,\
                    padding = "valid", activation = tf.nn.leaky_relu),

    layers.Conv2DTranspose(2*base_depth, 5, strides = 1,\
                    padding = "same", activation = tf.nn.leaky_relu),

    layers.Conv2DTranspose(2*base_depth, 5, strides = 2,\
                    padding = "same", activation = tf.nn.leaky_relu),

     layers.Conv2DTranspose(base_depth, 5, strides = 1,\
                    padding = "same", activation = tf.nn.leaky_relu),

     layers.Conv2DTranspose(base_depth, 5, strides = 2,\
                    padding = "same", activation = tf.nn.leaky_relu),

     layers.Conv2DTranspose(base_depth, 5, strides = 1,\
                    padding = "same", activation = None),
     layers.Flatten(),
     tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits)
    ], name = "Decoder")

vae = models.Model(inputs = encoder.inputs, outputs = decoder(encoder.outputs[0]), 
    name = "VAE")
# from tensorflow.keras import utils
#utils.plot_model(encoder, show_shapes = True, show_layer_names = False)
#utils.plot_model(decoder, show_shapes = True, show_layer_names = False)
#utils.plot_model(vae, show_shapes = True, show_layer_names = False)
#
#

negloglik = lambda x. rv_x: -rv_x.log_prob(x)
vae.compile(optimizer = optimizers.Adam(learning_rate = 1e-3),loss = negloglik)
history = vae.fit(train_dataset, epochs = 15, validation_data = eval_dataset)

## TEST THE MODEL
##
#Visualizing random selected images
x = next(iter(eval_dataset))[0][:10]
xhat = vae(x)


import matplotlib.pyplot as plt


# function to visualize
def display_imgs(x, y=None):
  if not isinstance(x, (np.ndarray, np.generic)):
    x = np.array(x)
  plt.ioff()
  n = x.shape[0]
  fig, axs = plt.subplots(1, n, figsize=(n, 1))
  if y is not None:
    fig.suptitle(np.argmax(y, axis=1))
  for i in range(n):
    axs.flat[i].imshow(x[i].squeeze(), interpolation='none', cmap='gray')
    axs.flat[i].axis('off')
  plt.show()
  plt.close()
  plt.ion()


print('Originals:')
display_imgs(x)

print('Decoded Random Samples:')
display_imgs(xhat.sample())

print('Decoded Modes:')
display_imgs(xhat.mode())

print('Decoded Means:')
display_imgs(xhat.mean())


# Now, let's generate ten never-before-seen digits.
z = prior.sample(10)
xtilde = decoder(z)


print('Randomly Generated Samples:')
display_imgs(xtilde.sample())

print('Randomly Generated Modes:')
display_imgs(xtilde.mode())

print('Randomly Generated Means:')
display_imgs(xtilde.mean())
