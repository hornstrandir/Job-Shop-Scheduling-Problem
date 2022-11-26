import os
import random
import time
import multiprocessing as mp
import numpy as np
import ray
import wandb
from envs.jss_env import JssEnv
from envs.energy_flexible_jss_env import EnergyFlexibleJssEnv
from utils.config import ENV_CONFIG
from utils.config import MODIFIED_CONFIG_PPO
from utils.CustomCallbacks import CustomCallbacks
from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import register_env
import ray.tune.integration.wandb as wandb_tune


import wandb
from algorithms.fcnn_model import FCMaskedActionsModelTF

tf1, tf, tfv = try_import_tf()

def _handle_results(results):
    pass

if __name__ == "__main__":
    ray.init()
    wandb.init(config=MODIFIED_CONFIG_PPO)
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    register_env("JssEnv-v0", lambda config: EnergyFlexibleJssEnv(config))
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(MODIFIED_CONFIG_PPO)
    wandb.config.update(ppo_config)
    trainer = ppo.PPO(config=ppo_config)
    
    start_time = time.time()
    
    stop = {
        "time_total_s": 10 * 60,
    }
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        
        #print(result["custom_metrics"])
        #result = wandb_tune._clean_log(result)
        #log, config_update = _handle_result(result)
        wandb.log(result["custom_metrics"])
        # wandb.config.update(config_update, allow_val_change=True)
    # trainer.export_policy_model("/home/jupyter/JSS/JSS/models/")

    ray.shutdown()    