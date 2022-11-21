from jss_env.envs.jss_env import JssEnv
from algorithms.fcnn_model import FCMaskedActionsModelTF
from utils.env_config import env_config

import ray
from ray.rllib.algorithms import ppo

from ray.rllib.models import ModelCatalog

from ray.rllib.utils.framework import try_import_tf

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

import numpy as np
import random
import os


tf1, tf, tfv = try_import_tf()

MODEL_CONFIG = {
    "layer_size": 319,
    "layer_nb": 2,
}

CONFIG = {
    "env": "JssEnv-v0",  
    "env_config": env_config,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "model": {
        "custom_model": "fc_masked_model_tf",
        "fcnet_activation": "relu",
        "fcnet_hiddens": [MODEL_CONFIG['layer_size'] for k in range(MODEL_CONFIG['layer_nb'])],
        "vf_share_layers": False,
        
    },
    "num_workers": 2,  # parallelism
    "framework": "tf",
    "ignore_worker_failures": True,
    "log_level": "DEBUG"
}

if __name__ == "__main__":
    ray.init()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    register_env("JssEnv-v0", lambda config: JssEnv(config))
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(CONFIG)
    ppo_config['model'] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        'fcnet_hiddens': [MODEL_CONFIG['layer_size'] for k in range(MODEL_CONFIG['layer_nb'])],
        "vf_share_layers": False,

    }
    ppo_config["lr"] = 1e-3

    # Get PPOs common config and update it with the values from above



    trainer = ppo.PPO(config=ppo_config) #env="JssEnv-v0")

    for _ in range(40):
        result = trainer.train()
        print(pretty_print(result))
        if result["timesteps_total"] >= 100000:
            break    

    ray.shutdown()