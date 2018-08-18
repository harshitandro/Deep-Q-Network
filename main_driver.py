import argparse
import os

import gym
import numpy as np

np.random.seed(1024)
from collections import deque
import matplotlib.pyplot as plt
from time import time
import keras as K

from agent.ddqn_agent import DDQNAgent as Agent

from config.loggers import result_logger, logger
import imageio

env_name = None
initial_timestamp = 0.0

def train_model(n_episodes=2000, eps_start=1.0, eps_end=0.001, eps_decay=0.9, target_reward=1000):
    """DDQN Model Training method

    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    logger.info("Starting model training for {} episodes.".format(n_episodes))
    consolidation_counter = 0
    for i_episode in range(1, n_episodes + 1):
        init_time = time()
        state = agent.reset_episode()
        score = 0
        done = False
        while not done:
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                agent.update_target_model()
                break
        time_taken = time() - init_time
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        logger.debug('Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tState: {}\tMean Q-Target: {:.4f}'
                     '\tEffective Epsilon: {:.3f}\tTime Taken: {:.2f} sec'.format(
            i_episode, np.mean(scores_window), score, state[0], np.mean(agent.Q_targets), eps, time_taken))
        if i_episode % 100 == 0:
            result_logger.info(
                'Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tState: {}\tMean Q-Target: {:.4f}\tTime Taken: {:.2f} sec '.format(
                    i_episode, np.mean(scores_window), score, state[0], np.mean(agent.Q_targets), time_taken))
            agent.local_network.model.save('save/{}_local_model_{}.h5'.format(env_name, initial_timestamp))
            agent.target_network.model.save('save/{}_target_model_{}.h5'.format(env_name, initial_timestamp))
        if np.mean(scores_window) >= target_reward:
            consolidation_counter += 1
            if consolidation_counter >= 5:
                result_logger.debug("Completed model training with avg reward {} over last {} episodes."
                                    " Training ran for total of {} epsiodes".format(
                    np.mean(scores_window), 100, i_episode))
                return scores
        else:
            consolidation_counter = 0
    result_logger.debug("Completed model training with avg reward {} over last {} episodes."
                        " Training ran for total of {} epsiodes".format(
        np.mean(scores_window), 100, n_episodes))
    return scores


def play_model(actor, env_render=False, return_render_img=False):
    state = env.reset()
    logger.debug("Start state : {}".format(state))
    score = 0
    done = False
    images = []
    while not done:
        if env_render:
            if return_render_img:
                images.append(env.render("rgb_array"))
            else:
                env.render()
        state = np.reshape(state, [-1, env.observation_space.shape[0]])
        action = actor.predict(state)
        next_state, reward, done, _ = env.step(np.argmax(action))
        state = next_state
        score += reward
        if done:
            return score, images
    return 0, images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DQNetwork based RL agent program to solve control tasks having discrete action space.')
    parser.add_argument(
        'mode', metavar='str', type=str, default="train",
        help='mode in which to run the model : train or test. Defaults to train')
    parser.add_argument(
        'env', metavar='str', type=str,
        help='OpenAI Gym env to be used')
    parser.add_argument(
        '--itr-count', metavar='int', type=int, default=500,
        help='Number of iterations to run the model. Defaults to 500 iterations')
    parser.add_argument(
        '-b', '--batch-size', metavar='int', type=int, default=32,
        help='Batch Size to train the model. Defaults to 32')
    parser.add_argument(
        '-lr', default=0.0001, metavar='float', type=float,
        help='Learning rate of the model')
    parser.add_argument(
        '--gamma', default=0.99, metavar='float', type=float,
        help='Discounting Factor')
    parser.add_argument(
        '--render', default=False, metavar='bool', type=bool,
        help='Whether to render the env while the model performs in test mode')
    parser.add_argument(
        '--model-path', metavar='str', type=str,
        help='Saved Model file to load in case of test mode')
    parser.add_argument(
        '--test-total-iterations', metavar='int', type=int,
        help='Number of consecutive iterations for which the test score is to be averaqed')
    parser.add_argument(
        '--avg-expected-reward', default=195.0, metavar='float', type=float,
        help='Average reward expected over given consecutive iterations to consider the env solved')

    args = parser.parse_args()
    initial_timestamp = time()

    env_name = args.env

    if args.mode == "test":
        initial_timestamp = args.model_path[:-3].split("_")[-1]

    result_logger.info("=" * 110)
    result_logger.info("OpenAI Gym Task : {} Solution program".format(env_name))
    result_logger.debug("Initial Timestamp : {}".format(initial_timestamp))
    result_logger.info("Starting new instance of the program in  {} mode...".format(args.mode))
    result_logger.info("Initialised with parameters")
    result_logger.debug("Parameters list : {}".format(args.__dict__))

    mode = args.mode
    env = gym.make(env_name)
    logger.info("OpenAI Gym Env. under effect : {}".format(env_name))
    logger.debug('State shape: {}'.format(env.observation_space.shape))
    logger.debug('Number of actions: {}'.format(env.action_space.n))


    if args.mode == "train":
        agent = Agent(env, buffer_size=100000, gamma=args.gamma, batch_size=args.batch_size, lr=args.lr, callbacks=[],
                      logger=logger)
        scores = train_model(n_episodes=args.itr_count, target_reward=args.avg_expected_reward, eps_decay=0.99)
        logger.info("Plotting reward per episode.")
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        logger.info("Saving the reward plot as image at results/reward_plot/reward_plot_{}_{}.jpeg".format(env_name,
                                                                                                           initial_timestamp))
        fig.savefig("results/reward_plot/reward_plot_{}_{}.jpeg".format(env_name, initial_timestamp))
    elif args.mode == "test":
        test_scores = []
        result_logger.info("Loading the saved model from '{}'".format(args.model_path))
        actor = K.models.load_model('{}'.format(args.model_path))
        result_logger.info("Creating a gif of how agent is playing in the env at :"
                           " results/agent_play/agentplay_{}_{}.gif".format(env_name, initial_timestamp))
        images = play_model(actor, True, True)[1]
        with imageio.get_writer("results/agent_play/agentplay_{}_{}.gif".format(env_name, initial_timestamp),
                                mode='I') as writer:
            for image in images:
                writer.append_data(image)
        result_logger.info("Now running model test for {} iterations with expected reward >= {}".format(
            args.test_total_iterations, args.avg_expected_reward))
        for itr in range(1, args.test_total_iterations + 1):
            score = play_model(actor, args.render)[0]
            test_scores.append(score[0])
            result_logger.info("Iteration: {} Score: {}".format(itr, score))
        avg_reward = np.mean(test_scores)
        result_logger.info("Total Avg. Score over {} consecutive iterations : {}".format(args.test_total_iterations,
                                                                                         avg_reward))
        if avg_reward >= args.avg_expected_reward:
            result_logger.info("Env. solved successfully.")
        else:
            result_logger.warning("Agent failed to solve the env.")

    os.rename("logs/info.log", "logs/{}_{}_{}.log".format(args.mode, env_name, initial_timestamp))
    os.rename("results/result.log", "results/result_{}_{}_{}.log".format(args.mode, env_name, initial_timestamp))
