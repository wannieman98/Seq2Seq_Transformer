# simple_transformer PyTorch

Python implementation of [Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention#training-data-and-batching) using PyTorch to translate Korea into English.


### Dataset
For this project, the Korean-English translation corpus from [AI Hub](https://aihub.or.kr/aidata/87/download) was utilized to train the Transformer. 

For Tokenizaton, I used Pytorch Tokenization using [spacy](https://spacy.io) for english and [soynlp](https://github.com/lovit/soynlp#vectorizer). But the alternative is to use [tokenizers](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer) module to train BPEtokenizers from scratch.

### Overview
    * # of Korean Sentences: 60,000
    * # of English Sentences: 60,000
    ```

    [KOR]: 11장에서는 예수님이 이번엔 나사로를 무덤에서 불러내어 죽은 자 가운데서 살리셨습니다.
    [ENG]: In Chapter 11 Jesus called Lazarus from the tomb and raised him from the dead.

    [KOR]: 6.5, 7, 8 사이즈가 몇 개나 더 재입고 될지 제게 알려주시면 감사하겠습니다.
    [ENG]: I would feel grateful to know how many stocks will be secured of size 6.5, 7, and 8.

    [KOR LEN]: 600000
    [ENG LEN]: 600000

    soynlp tokenizer:
    [KOR]: ['11', '장에서는', '예수님', '이', '이번', '엔', '나사로를', '무덤에서', '불러', '내어', '죽은', '자', '가운데', '서', '살리셨습니다.']
    spacy tokenizer:
    [ENG]: ['In', 'Chapter', '11', 'Jesus', 'called', 'Lazarus', 'from', 'the', 'tomb', 'and', 'raised', 'him', 'from', 'the', 'dead', '.']

    sentenceBPE tokenizers:
    [KOR]: ['11', '장', '에서는', '예수', '님이', '이번엔', '나', '사로', '를', '무', '에서', '불러', '내어', '죽은', '자', '가운데', '서', '살', '리', '셨습니다', '.']
    [ENG]: ['In', 'Cha', 'pter', '11', 'Jesus', 'called', 'La', 'z', 'ar', 'us', 'from', 'the', 'tomb', 'and', 'raised', 'him', 'from', 'the', 'dead', '.']    
    ```

### Requirements
* Such libraries are necessary to run the program.
    ```
    torch==1.9.0
    spacy==2.2.4
    soynlp==0.0.493
    tokenizers==0.10.3
    torchtesxt==0.10.0
    en-core-web-sm==2.1.0
    ```

### Usage
* By default, the trainer will use the korean-english dataset from [AI Hub](https://aihub.or.kr/aidata/87/download), in order to use your own dataset, please create a folder with datasets in it and run the script.

* You can also choose which tokenizers to use
    1. soynlp tokenizers for korean and spacy tokenizers for english(default)
    2. BPE Tokenizers from scratch for both kor and eng tokenizers

* As well as choose to load pre-trained transformer
    1. False(default)
    2. True
    ```
    default:
    python train.py

    custom:
    python train.py --f YOUR_FILE_PATH --t 2 --l True
    ```

### References

#TODO: Trying to turn the program into a runnable script which lets you train on your device with hyperparameters and data as variables.