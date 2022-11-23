import random
import numpy as np
import wandb
import gym

from utils.config import ENV_CONFIG


def FIFO_worker(default_config, env):
    env = env(default_config)
    #wandb.init(config=default_config)
    #config = wandb.config
    #env = gym.make('jss_flexible_env.envs:JssFlexibleEnv', env_config={'instance_path': config['instance_path']})
    env.seed(2022)
    random.seed(2022)
    np.random.seed(2022)
    done = False
    state = env.reset()
    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.jobs, 7))
        remaining_time = reshaped[:, 5]
        illegal_actions = np.invert(legal_actions)
        mask = illegal_actions * -1e8
        remaining_time += mask
        FIFO_action = np.argmax(remaining_time)
        print(FIFO_action)
        assert legal_actions[FIFO_action]
        state, reward, done, _ = env.step(FIFO_action)
        print(f"Partial Solution: {env.solution}")
        print(f"legal actions: {env.legal_actions}")
    env.reset()
    make_span = env.last_time_step
    print(make_span)
    #wandb.log({"nb_episodes": 1, "make_span": make_span})


if __name__ == "__main__":
    FIFO_worker(ENV_CONFIG)