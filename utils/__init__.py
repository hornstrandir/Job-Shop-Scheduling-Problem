import os

from dotenv import load_dotenv

load_dotenv()
if os.environ.get('WANDB_ENTITY') is None:
    raise ValueError('You need to provide environment variables as instructed in readme.')

