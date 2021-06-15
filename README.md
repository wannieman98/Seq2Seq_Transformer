# simple_transformer PyTorch

Python implementation of [Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention#training-data-and-batching) using PyTorch to translate Korea into English.


#### Dataset
For this project, the Korean-English translation corpus from [AI Hub](https://aihub.or.kr/aidata/87/download) was utilized to train the Transformer. 

For Tokenizaton, I used Pytorch Tokenization using [spacy](https://spacy.io) for english and [soynlp](https://github.com/lovit/soynlp#vectorizer). But the alternative is to use [tokenizers](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer) module to train BPEtokenizers from scratch.

###### Overview
    * # of Korean Sentences: 60,000
    * # of English Sentences: 60,000
    ```
    [KOR]: 11장에서는 예수님이 이번엔 나사로를 무덤에서 불러내어 죽은 자 가운데서 살리셨습니다.
    [ENG]: In Chapter 11 Jesus called Lazarus from the tomb and raised him from the dead.

    [KOR]: 6.5, 7, 8 사이즈가 몇 개나 더 재입고 될지 제게 알려주시면 감사하겠습니다.
    [ENG]: I would feel grateful to know how many stocks will be secured of size 6.5, 7, and 8.

    [KOR LEN]: 600000
    [ENG LEN]: 600000
    ```