# simple_transformer PyTorch

Python implementation of [Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention#training-data-and-batching) using PyTorch to translate Korea into English.


#### Dataset
For this project, the Korean-English translation corpus from [AI Hub](https://aihub.or.kr/aidata/87/download) was utilized to train the Transformer. 

For Tokenizaton, I used Pytorch Tokenization using [spacy](https://spacy.io) for english and [soynlp](https://github.com/lovit/soynlp#vectorizer).