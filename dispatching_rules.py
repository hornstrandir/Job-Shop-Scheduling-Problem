from algorithms.fifo import FIFO_worker
from utils.config import ENV_CONFIG
from jss_flexible_env.envs import JssFlexibleEnv


if __name__ == "__main__":
    FIFO_worker(ENV_CONFIG, JssFlexibleEnv)