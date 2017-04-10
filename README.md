# Fake News Challenge (formal name pending) 
### Fake News Detector for CS483
### [Fake News Challenge](http://www.fakenewschallenge.org)

# Goal
The goal of the Fake News Challenge is to explore how artificial intelligence technologies, particularly machine learning and natural language processing, might be leveraged to combat the fake news problem. We believe that these AI technologies hold promise for significantly automating parts of the procedure human fact checkers use today to determine if a story is real or a hoax.

# Overview
## Stance Detection
This project leverages stance detection which involves estimating the relative perspective (or stance) of two pieces of text relative to the topic, claim or issue.
### Input
A headline and a body text - either from the same news article or from two different articles.
### Output
Classify the stance of the body text relative to the claim made in the headline into one of four categories:
  1. Agrees: The body text agrees with the headline.
  2. Disagrees: The body text disagrees with the headline.
  3. Discusses: The body text discuss the same topic as the headline, but does not take a position
  4. Unrelated: The body text discusses a different topic than the headline

# Baseline
## Current FNC Baseline
https://github.com/FakeNewsChallenge/fnc-1-baseline

### Features
  1. lemmatizes and lowercases word from nltk
  2. removes stop words
  3. uses a list of hardcoded refuting features for headline (returns 1 if a refuting word is in the headline, otherwise 0) 
  4. uses a list of refuting words to calcuate the polarity by returning the sum of the number of times a refuting terms in the headline or body.
  5. counts how many times a token in the title appears in the body
  6. count how many times an n-gram of the title appears in the entire body and intro paragraph

# Reference
## Current FNC Baseline
https://github.com/FakeNewsChallenge/fnc-1-baseline

## Support Vector Machine
At the heart of this project's classification system is a support vector machine (SVM). A SVM is a supervised learning model used for classification and regression analysis.

The SVM handles multi-classification by a One-vs.-one scheme:
  1. http://www.svm-tutorial.com/2017/02/svms-overview-support-vector-machines/
  2. https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-one
