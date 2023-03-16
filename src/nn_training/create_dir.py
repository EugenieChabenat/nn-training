import os 
from nn_training import config, nn_modules 

directory = pathlib.Path(config['paths']['checkpoints'])


os.mkdir(directory)
