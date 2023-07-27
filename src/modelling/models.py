from keras.models import Sequential
import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Dense, RNN
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors

""" library codes for building Bayesian Neural Network """


# define the prior weight distribution using standard Gaussian N(0, 1)


def sg_prior(kernel_size, bias_size, dtype=None):
    """ define a standard Gaussian prior """

    n = kernel_size + bias_size
    prior_model = Sequential([
        tfpl.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(
                loc=tf.zeros(n), scale_diag=tf.ones(n))
        )
    ])
    return prior_model


def gmm_prior(num_modes, latent_dim):
    prior = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=[1 / num_modes,] * num_modes),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=tf.Variable(tf.random.normal(shape=[num_modes, latent_dim])),
            scale_diag=tfp.util.TransformedVariable(tf.Variable(
                tf.ones(shape=[num_modes, latent_dim])), bijector=tfb.Softplus())
        )
    )
    return prior


# define variational posterior weight distribution -- multivariate Gaussian with full covariance
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = Sequential([
        tfpl.VariableLayer(
            tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfpl.MultivariateNormalTriL(n)
    ])
    return posterior_model


def posterior2(kernel_size, bias_size, dtype=None):
    """ mean-field Gaussian posterior """

    n = kernel_size + bias_size
    posterior_model = Sequential([
        tfpl.VariableLayer(
            tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
        tfp.layers.IndependentNormal(n)
    ])
    return posterior_model


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


##### Define a base Bayesian dense neural network #####

def BDNN_svi(norm, N):
    """ A function to create a BDNN model for scalar regression

    Parameters
    ----------
    norm : tf.keras.layers.LayerNormalization
        normalization layer;
    N : int
        the number of training samples, i.e. x_train.shape[0];
    """

    model = Sequential([
        norm,
        tfpl.DenseVariational(
            units=16,
            make_prior_fn=sg_prior,
            make_posterior_fn=posterior,
            kl_weight=1/N,
            activation='sigmoid'),
        tfpl.DenseVariational(
            units=tfpl.IndependentNormal.params_size(1),
            make_prior_fn=sg_prior,
            make_posterior_fn=posterior,
            kl_weight=1/N,
        ),
        tfpl.IndependentNormal(1),  # scalar regression
    ])
    return model


def BDNN_flipout(N, norm):
    # for compute KL divergence

    def kernel_divergence_fn(
        q, p, _): return tfp.distributions.kl_divergence(q, p) / (N * 1.0)
    def bias_divergence_fn(
        q, p, _): return tfp.distributions.kl_divergence(q, p) / (N * 1.0)

    model_flipout = Sequential([
        norm,
        tfpl.DenseFlipout(units=16,
                          activation='relu',
                          kernel_divergence_fn=kernel_divergence_fn,
                          bias_divergence_fn=bias_divergence_fn,),
        tfpl.DenseFlipout(tfpl.IndependentNormal.params_size(1),
                          kernel_divergence_fn=kernel_divergence_fn,
                          bias_divergence_fn=bias_divergence_fn,),
        tfpl.IndependentNormal(1),  # scalar regression
    ])
    # tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    return model_flipout


def create_and_compile_aleatoric_model(input_shape, norm):
    """ compute the sigma of the aleatoric uncertainty """

    model = tf.keras.Sequential([
        norm,
        tf.keras.layers.Dense(16, activation='relu',
                              input_shape=(input_shape,)),
        tf.keras.layers.Dense(2, activation='relu'),
        tfpl.IndependentNormal(1),
    ])

    model.compile(
        loss=nll,
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005),
        metrics=['mae', 'mape'])

    return model


def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mae', 'mape']
                  )
    return model


### temporal models ###


def build_lstm(units=16):
    """ build a simple LSTM model """

    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        # tf.keras.layers.LSTM(16, return_sequences=True),
        tf.keras.layers.LSTM(units, return_sequences=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    return model


def build_BLSTM_svi(latent_dim, KL_scaling_factor):
    ''' build a mixed-uncertainty Bayesian LSTM model by svi '''

    def kernel_divergence_fn(q, p, _): return tfp.distributions.kl_divergence(
        q, p) * KL_scaling_factor

    model = tf.keras.models.Sequential([
        RNN(ed.layers.LSTMCellFlipout(
            latent_dim,
            kernel_regularizer=ed.regularizers.NormalKLDivergence(
                scale_factor=KL_scaling_factor),
            recurrent_regularizer=ed.regularizers.NormalKLDivergence(
                scale_factor=KL_scaling_factor),
            input_shape=(None, 5))),
        Dense(tfpl.IndependentNormal.params_size(1)),
        tfpl.IndependentNormal(1),
    ], name='svi_BLSTM')
    return model


def build_BLSTM_svi2(latent_dim, KL_scaling_factor):
    ''' build a mixed-uncertainty Bayesian LSTM model by svi '''

    def kernel_divergence_fn(q, p, _): return tfp.distributions.kl_divergence(
        q, p) * KL_scaling_factor

    model = tf.keras.models.Sequential([
        RNN(ed.layers.LSTMCellFlipout(
            latent_dim,
            kernel_regularizer=ed.regularizers.NormalKLDivergence(
                scale_factor=KL_scaling_factor),
            recurrent_regularizer=ed.regularizers.NormalKLDivergence(
                scale_factor=KL_scaling_factor),
            input_shape=(None, 5))),
        Dense(1),
    ], name='svi_BLSTM')
    return model


def compile_and_fit_lstm(model, window, loss, EPOCHS, VERBOSE=0, early_stopping=False, patience=2):
    """ compile and fit temporal models """

    if early_stopping == True:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min')

    model.compile(
        loss=loss,  # mse
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', 'mape'])  # mae

    history = model.fit(
        window.train,
        validation_data=window.val,
        epochs=EPOCHS,
        verbose=VERBOSE)
    return history


def lstm_SRT(units=16):

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units, return_sequences=False, dropout=0.5),
        tf.keras.layers.Dense(units=1)
    ])
    return model
