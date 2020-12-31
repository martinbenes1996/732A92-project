
import logging
import pandas as pd
import spacy

# === url ===
# by-source datasets
youtube_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-mendeley-9030361288/22895576/youtube_parsed_dataset.csv'
twitter_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-mendeley-9030361288/22895537/twitter_parsed_dataset.csv'
kaggle_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-mendeley-9030361288/22895477/kaggle_parsed_dataset.csv'
# sentiment datasets
aggression_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-mendeley-9030361288/22895468/aggression_parsed_dataset.csv'
attack_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-mendeley-9030361288/22895471/attack_parsed_dataset.csv'
toxicity_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-mendeley-9030361288/22895489/toxicity_parsed_dataset.csv'
twitter_racism_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-mendeley-9030361288/22895549/twitter_racism_parsed_dataset.csv'
twitter_sexism_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-mendeley-9030361288/22895561/twitter_sexism_parsed_dataset.csv'

def parse_figshare_dataset(url, sentiment = None, source = None, columns = {}):
    """FigShare dataset downloading and parsing."""
    logging.info("fetching %s [%s]" % (source, sentiment))
    # override columns
    columns = pd.Series({'Text': 'text', 'ed_label_1': 'score','oh_label': 'label', **columns})
    columns = columns[~columns.isna()].to_dict()
  
    # download
    x = pd.read_csv(url)\
        .rename(columns = columns) # rename columns
    # add sentiment columns
    if sentiment is not None:
        x['sentiment'] = x[columns['oh_label']].apply(
            lambda v: sentiment if bool(v) else False
        )
        columns['sentiment'] = 'sentiment'
    # add source
    x['source'] = source if source is not None else 'unknown'
    columns['source'] = 'source'
    # project only wanted columns
    return x[[c for c in columns.values()]]


def parse_figshare_twitter_dataset(*args, **kwargs):
    """FigShare Twitter dataset downloading and parsing."""
    cols = {'ed_label_1': None, 'Annotation': 'sentiment'}
    x = parse_figshare_dataset(*args, **kwargs, source = 'twitter', columns = cols)
    x['sentiment'] = x.sentiment.apply(lambda s: s if s != "none" else False)
    return x

def parse_figshare_youtube_kaggle_dataset(*args, **kwargs):
    """FigShare YouTube dataset downloading and parsing."""
    x = parse_figshare_dataset(*args, **kwargs,
                               sentiment = 'bully', columns = {'ed_label_1': None})
    return x

def bully_dataset():
    # parse datasets
    aggression_df = parse_figshare_dataset(aggression_url, 'aggression')
    attack_df     = parse_figshare_dataset(attack_url, 'attack')
    toxicity_df   = parse_figshare_dataset(toxicity_url, 'toxicity')
    # drop score column
    aggression_df = aggression_df.drop('score', axis = 1)
    attack_df     = attack_df.drop('score', axis = 1)
    toxicity_df   = toxicity_df.drop('score', axis = 1)
    
    # parse datasets
    twitter_sexism_df = parse_figshare_twitter_dataset(twitter_sexism_url)
    twitter_racism_df = parse_figshare_twitter_dataset(twitter_racism_url)
    twitter_df        = parse_figshare_twitter_dataset(twitter_url)
    
    # parse dataset - bully is a general type of negative sentiment
    youtube_df = parse_figshare_youtube_kaggle_dataset(youtube_url, source = 'youtube')
    kaggle_df = parse_figshare_youtube_kaggle_dataset(kaggle_url, source = 'kaggle')

    # concatenate the datasets
    df = pd.concat([aggression_df, attack_df, toxicity_df,
                    twitter_sexism_df, twitter_racism_df,
                    twitter_df, youtube_df, kaggle_df])
    logging.info("raw samples: %d" % (df.shape[0]))

    # drop duplicated comments and NA rows
    #df = df.drop_duplicates(subset = "text") # drops sentiment from many
    df = df.drop_duplicates(subset = "text")
    logging.info("deduplicated samples: %d" % (df.shape[0]))
    df = df.dropna()
    logging.info("nonempty samples: %d" % (df.shape[0]))
    # drop sentiment totally (for now)
    try: # if ran several times
        df = df.drop('sentiment', axis = 1)
        logging.info("multiclass sentiment to binary sentiment")    
    except: pass

    # load English tokenizer
    nlp = spacy.load("en_core_web_sm",
                     disable=["tagger","ner","textcat"])#"parser",,]) # to speed up

    # implement tokenizer
    def any_alnum(x):
        return any([c.isalnum() for c in x])
    i,i_last,N = 0,-1,df.shape[0]
    def preprocess_words(text):
        # logs
        #global i, i_last, N
        #i += 1
        #percent = int(i/N * 100)
        #if percent > i_last:
        #    print(percent, "%", end = "\r")
        
        return [tok.text for tok in nlp(text) if any_alnum(tok.text)]
    # tokenization
    logging.info("tokenizing")   
    df['text'] = df.text.apply(preprocess_words)
    
    # text to str
    logging.info("tokens to str")
    df['text'] = df.text.apply(str)
    # save
    logging.info("writing to output")
    df.to_csv('data/words.csv', index = False)
    # mount drive
    #from google.colab import drive
    #drive.mount('/drive', force_remount=True)
    # output to csv (to drive)
    #df.to_csv('/drive/My Drive/Colab Notebooks/data/words.csv', index = False)
    
if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    bully_dataset()