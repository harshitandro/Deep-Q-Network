import numpy as np

np.random.seed(1024)
import keras as K


class DQNetwork:
    """Standard QNetwork implementation : Actor(Policy) Model"""

    def __init__(self, state_size, action_size, action_high=1.0, action_low=0.0, layer_sizes=(64, 64),
                 batch_norm_options=(True, True), dropout_options=(0, 0), learning_rate=0.0001, logger=None):
        """Initialise the Network Model with given number of layes defined with given size.
        Parameters
        ==========
        :param state_size : size of the state space.
        :type state_size : int
        :param action_size : size of the action space.
        :type action_size : int
        :param action_high : Upper bound of the action space.
        :type action_high : float
        :param action_low : Lower bound of the action space.
        :type action_low : float
        :param layer_sizes : list of ints defining the size of each layer used in the model
        :type layer_sizes : list
        :param batch_norm_options : list of bool defining whether to use Batch Normalisation in layers used in the
        model. Index of element corresponds to number of layer to set.
        :type batch_norm_options : list
        :param dropout_options : list of float defining how much dropout is to be applied(to drop this much fraction)
        to the output of layers used in the model. Index of element corresponds to number of layer to set.
        :type dropout_options : list
        :param learning_rate : Learning Rate for the model.
        :type learning_rate : float
        """

        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        self.layer_sizes = layer_sizes
        self.batch_norm_options = batch_norm_options
        self.dropout_options = dropout_options
        self.learning_rate = learning_rate
        self.logger = logger

        # Build the model
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = K.layers.Input(shape=(self.state_size,), name='states')
        net = states
        # Add the hidden layers
        for layer_count in range(len(self.layer_sizes)):
            net = K.layers.Dense(units=self.layer_sizes[layer_count])(net)
            net = K.layers.Activation('relu')(net)
            if self.batch_norm_options[layer_count]:
                net = K.layers.BatchNormalization()(net)
            net = K.layers.Dropout(self.dropout_options[layer_count])(net)

        # Add final output layer with sigmoid activation
        actions = K.layers.Dense(units=self.action_size, activation='linear',
                                 name='raw_actions')(net)

        # Create Keras model
        self.model = K.models.Model(inputs=states, outputs=actions)

        # Print the created model summary
        self.logger.debug("Model Summery:")
        self.model.summary(print_fn=self.logger.debug)

        # Define optimizer and training function
        self.optimizer = K.optimizers.Adam(lr=self.learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)
