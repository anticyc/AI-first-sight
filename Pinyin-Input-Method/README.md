# Pinyin Input Method

Author: [anticyc · GitHub](https://github.com/anticyc)

## Description

A simple pinyin input method for Chinese characters.
It supports translating a pinyin string with spacebars as separators to a string of Chinese characters based on the Viterbi Algorithm.<br> Multiple models are introduced, including the bi-gram model, tri-gram model, and the phrase-based model.

## Usage

```sh
pip install -r requirements.txt
python main.py <data/input.txt >data/output.txt
```

This enables you to build a bi-gram model and translate the input file to the output file.

```sh
# Data Processor
python src/dataProcessor.py --folder <CORPUS_FOLDER> --output <data/temp> --pinyin False
# --folder: the folder containing the corpus [optional]
# --output: the output folder [optional]
# --pinyin: whether to use the Pinyin model, under which case pinyin is the primary key [optional, default: False]
```

This enables you to preprocess the corpus so that it can be used for training the model.

```sh
python src/evaluate.py --model <model_name> --input <data/input.txt> --output <YOUR_OUTPUT_PATH> --answer <data/answer.txt>
# --model: the model to use, choices:[bigram, trigram, phrase, triphrase, pinyin]
# --input: the input file [optional]
# --output: the output file [optional]
# --answer: the answer file [optional]
```

This enables you to evaluate the performance of the model on the input file and compare it with the answer file.

### Guidance for testing

1. Test models other than pinyin: Use dataProcessor to process them into one-, two-, and three-tuple data sets without pinyin information, and then use evaluate.py to evaluate the model.
   First process the corpus using `dataProcessor.py` to generate dataset without pinyin information. Then, use `evaluate.py` to evaluate the model. **Choose your model from the following list: `[bigram, trigram, phrase, triphrase]`**
   
   ```sh
   python src/dataProcessor.py --pinyin False
   python src/evaluate.py --model < *model_name* > --input data/input.txt --output data/< *model_name* >_output.txt --answer data/answer.txt
   ```

2. Test the pinyin model: use dataProcessor to process it into one- and two-tuple data sets containing pinyin information, and then use evaluate.py to evaluate the model.
   First process the corpus using `dataProcessor.py` to generate dataset with pinyin information. Then, use `evaluate.py` to evaluate the model. **Choose your model from the following list: `[pinyin]`**
   
   ```sh
   python src/dataProcessor.py --pinyin True
   python src/evaluate.py --model pinyin --input data/input.txt --output data/pinyin_output.txt --answer data/answer.txt
   ```

## Structure

```
corpus/           Put your corpus here
data/
    answer.txt    Expected Answer
    input.txt
    output.txt
    temp/         Temp folder
拼⾳汉字表.txt
⼀⼆级汉字表.txt
src/              Source code
    bigram.py     bi-gram model
    phrase.py     phrase model
    trigram.py    tri-gram model
    tri_phrase.py tri-gram phrase model
    evaluate.py
    withpy.py     pinyin bi-gram model
    dataProcessor.py
main.py
README.md
requirements.txt
```

## Format of Results

> Current Model: xxx
> Loading data...
> Generating results...
> 训练时间:  5.118
> 运行时间:  4.257
> 句子正确率:  0.397
> 单字正确率:  0.840
> 
> 

Note that the sentence accuracy and single-character accuracy are calculated based on the comparison between the predicted sentence and the standard answer. The **time** shown in the results is measured in seconds.