# Generating text using LSTM GAN in PyTorch

**Author**: Martin Benes

**Course:** 732A92 Text Mining

**Abstract:**

The aim of the project is to generate text using generative adversarial networks (GANs), where both components are recurrent/LSTM neural networks. The training data set is a collection of internet comments from YouTube, Twitter and Kaggle also containing a (binary) negative sentiment indicator. The model output is evaluated using the result of Markov chain trained on the same data. Absolute quality of the text is evaluated manually against a test set. Sentiment input integration into LSTM GAN is discussed.

## Usage

All following code works with *cwd* set to root of the project and initial lines

```{python}
import sys
sys.path.append('src')
```

When imported, the code tries to use Cuda GPU (if present), otherwise it
falls back to use CPU. Turn the logs on by

```{python}
import logging
logging.basicConfig(level = logging.INFO)
```

## Data

Data come from [figshare.com](https://figshare.com/articles/Cyberbullying_datasets/12423407). Download and parse the dataset with

```{python}
import fetch

bully_dataset(tokenize = False) # sentences
bully_dataset(tokenize = True) # words
```

## Training

```{python}
import gan
import train

# train NN with scalar incremental embedding
model = train.ScalarIncremental()
gan.save(model, 'models/nn/ScalarIncremental.model')
```

```{python}
# train NN with closest word2vec embedding
model = train.ClosestWord2Vec()
gan.save(model, 'models/nn/ClosestWord2Vec.model')
```

```{python}
# train NN with Bert embedding
model = train.Bert()
gan.save(model, 'models/nn/Bert.model')
```

The training can be parameterized in following way

```{python}
config.batch_size = 1000 # default 500
config.num_epochs = 5 # default 10

# if ran on colab, change to True to read from Google Drive
config.set_from_drive(False) # ran locally
```

## Generate text

Code will attempt to load the modules from the paths
above, if not found training is started.

```{python}
import evaluate

# generate text with Scalar Incremental embedding
result = evaluate.generate.ScalarIncremental(N = 10) # default 1
result.to_csv('output/scalar.txt', index = False)
```

```{python}
# generate text with closest word2vec embedding
result = evaluate.generate.ClosestWord2Vec(N = 10) # default 1
result.to_csv('output/word2vec.txt', index = False)
```

```{python}
# generate text with Bert embedding
result = evaluate.generate.Bert(N = 10) # default 1
result.to_csv('output/bert.txt', index = False)
```

Generate Markov Chain text with

```{python}
import dataset
import markov

# load train data
words = dataset._read_words()
# train markov chain and predict
result = markov.generate_sentences(words = words, N = 1000)
predictions.to_csv('output/markov.txt', index = False)
```

## Measure performance

Code will attempt to load the modules from the paths
above, if not found training is started.

```{python}
score1 = evaluate.confusion.ScalarIncremental()
score2 = evaluate.confusion.ClosestWord2Vec()
score3 = evaluate.confusion.Bert()
```

```{python}
# loss plot of model with scalar incremental embedding
model = gan.load('models/nn/ScalarIncremental.model')
evaluate.error_plot(model)
```

```{python}
# loss plot of model with closest word2vec embedding
model = gan.load('models/nn/ClosestWord2Vec.model')
evaluate.error_plot(model)
```

```{python}
# loss plot of model with Bert embedding
model = gan.load('models/nn/Bert.model')
config.num_epochs = 4
evaluate.error_plot(model)
```
