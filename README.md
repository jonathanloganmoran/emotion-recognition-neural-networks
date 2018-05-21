# Emotion recognition with CNN

Contributing author: @jonathanloganmoran

Computational Cognitive Neuroscience (CSE 173) | Spring '18 | Prof. David Noelle

Forked model used in a Seminar Neural Networks course at TU Delft, published by @isseu

![Angry Test](https://raw.githubusercontent.com/jonathanloganmoran/emotion-recognition-neural-networks/master/paper/Screen%20Shot%202018-05-04%20at%207.01.20%20PM.png)

![Angry Test](https://github.com/jonathanloganmoran/emotion-recognition-neural-networks/blob/master/paper/Screen%20Shot%202018-05-04%20at%207.02.37%20PM.png)

 67% Accuracy

 ![Angry Test](https://raw.githubusercontent.com/isseu/emotion-recognition-neural-networks/master/paper/matrix_final.png)

## Dataset

We use the [FER-2013 Faces Database](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main), a set of 28,709 pictures of people displaying 7 emotional expressions (angry, disgusted, fearful, happy, sad, surprised and neutral).

You have to request for access to the dataset or you can get it on [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). Download `fer2013.tar.gz` and decompress `fer2013.csv` in the `./data` folder.

Install all the dependencies using `virtualenv`.

```bash
virtualenv -p python3 ./
source ./bin/activate
pip install -r requirements.txt
```

The data is in CSV and we need to transform it using the script `csv_to_numpy.py` that generates the image and label data in the `data` folder.

```bash
$ python3 csv_to_numpy.py
```

## Usage

```bash
# To train a model
$ python3 emotion_recognition.py train
# To use it live
$ python3 emotion_recognition.py poc
```

## Presentation/Project Overview

[Link](https://github.com/jonathanloganmoran/emotion-recognition-neural-networks/blob/master/Emotion%20Recognition%20in%20CNNs.pdf)

## Paper

[Link](https://github.com/isseu/emotion-recognition-neural-networks/blob/master/paper/Report_NN.pdf)
