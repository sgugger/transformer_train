from fastai.text import *
from fastai.callbacks import SaveModelCallback
#from memory_profiler import profile
import fastprogress

fastprogress.fastprogress.SAVE_PATH = 'log.txt'

path = Config().data_path()/'wikitext-2'
#path = "/raid/dldata/fastai/wikitext-2"

def istitle(line):
    return len(re.findall(r'^ = [^=]* = $', line)) != 0

def read_file(filename):
    articles = []
    with open(filename, encoding='utf8') as f:
        lines = f.readlines()
    current_article = ''
    for i,line in enumerate(lines):
        current_article += line
        if i < len(lines)-2 and lines[i+1] == ' \n' and istitle(lines[i+2]):
            current_article = current_article.replace('<unk>', UNK)
            articles.append(current_article)
            current_article = ''
    current_article = current_article.replace('<unk>', UNK)
    articles.append(current_article)
    return np.array(articles)

train = read_file(path/'train.txt')
valid = read_file(path/'valid.txt')

all_texts = np.concatenate([valid, train])
df = pd.DataFrame({'texts':all_texts})
df.head()

df['texts'] = df['texts'].apply(lambda x:[BOS] + x.split(' '))
processor = [NumericalizeProcessor(min_freq=0)]

data = (TextList.from_df(df, path, cols='texts', processor=processor)
                .split_by_idx(range(0,60))
                .label_for_lm()
                .databunch(bs=5, bptt=150))

config = tfmerXL_lm_config.copy()
config['output_p'] = 0.1
config['embed_p'] = 0.1
config['ff_p'] = 0.1
config['resid_p'] = 0.1

learn = language_model_learner(data, TransformerXL, config=config, pretrained=False, clip=None, alpha=0, beta=2)
learn = learn.to_fp16(clip=0.1, dynamic=True)
learn.fit(1, 5e-4)
