import numpy as np

np.random.seed(1024)
import random
from collections import namedtuple, deque
from model.DQNetwork import DQNetwork


class DDQNAgent:
    """DDQN Agent implementation using Deep Q Network"""

    def __init__(self, env, buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, callbacks=(),
                 logger=None):
        """Initialise the Agent
        Parameters
        ==========
        :param env : OpenAI Gym Env.
        :param buffer_size : size of the replay memory buffer.
        :type buffer_size : int
        :param batch_size : batch size of experience to be used for training.
        :type batch_size : int
        :param gamma : Discount Factor.
        :type gamma : float
        :param tau : for soft update of target parameters.
        :type tau : float
        :param lr : Learning Rate.
        :type lr : float
        """
        self.env = env
        self.env.seed(1024)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.Q_targets = 0.0
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.callbacks = callbacks

        # Model arch definations
        layer_sizes = [256, 256]
        batch_norm_options = [False, False]
        dropout_options = [0, 0]

        logger.info("Initialising DDQN Agent with params : {}".format(self.__dict__))

        # Make local & target model
        logger.info("Initialising Local DQNetwork")
        self.local_network = DQNetwork(self.state_size, self.action_size,
                                       layer_sizes=layer_sizes,
                                       batch_norm_options=batch_norm_options,
                                       dropout_options=dropout_options,
                                       learning_rate=lr,
                                       logger=logger)

        logger.info("Initialising Target DQNetwork")
        self.target_network = DQNetwork(self.state_size, self.action_size,
                                        layer_sizes=layer_sizes,
                                        batch_norm_options=batch_norm_options,
                                        dropout_options=dropout_options,
                                        learning_rate=lr,
                                        logger=logger)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

    def reset_episode(self):
        state = self.env.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        """Make agent to perform a step as for the given params

        Params
        ======
            action (int): action selected on a given state
            reward (float): reward received by taking action in the emulator
            next_state (array_like): the state reached after taking action in state
            state (array_like): current state
            done (int): 0 or 1 defining if the episode termination has beem reached
        """
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = np.reshape(state, [-1, self.state_size])
        action = self.local_network.model.predict(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        for itr in range(len(states)):
            state, action, reward, next_state, done = states[itr], actions[itr], rewards[itr], next_states[itr], dones[
                itr]
            state = np.reshape(state, [-1, self.state_size])
            next_state = np.reshape(next_state, [-1, self.state_size])

            self.Q_targets = self.local_network.model.predict(state)
            if done:
                self.Q_targets[0][action] = reward
            else:
                next_Q_target = self.target_network.model.predict(next_state)[0]
                self.Q_targets[0][action] = (reward + gamma * np.max(next_Q_target))

            self.local_network.model.fit(state, self.Q_targets, epochs=1, verbose=0, callbacks=self.callbacks)

    def update_target_model(self):
        """copy weights from model to target_model"""
        self.target_network.model.set_weights(self.local_network.model.get_weights())


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
