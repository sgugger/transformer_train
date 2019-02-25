from fastai.text import *
from fastai.callbacks import SaveModelCallback
#from memory_profiler import profile
import fastprogress

fastprogress.fastprogress.SAVE_PATH = 'log.txt'

path = Config().data_path()/'wikitext-2'
#path = "/raid/dldata/fastai/wikitext-2"
data = load_data(path, bs=32, bptt=150)
config = tfmerXL_lm_config.copy()
config['output_p'] = 0.1
config['embed_p'] = 0.1
config['ff_p'] = 0.1
config['resid_p'] = 0.1

learn = language_model_learner(data, TransformerXL, config=config, pretrained=False, clip=None, alpha=0, beta=2)
learn = learn.to_fp16(clip=0.1, dynamic=True)
learn.fit(1, 5e-4)
