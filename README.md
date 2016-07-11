# IASNLP-2016
Author :Badr Mohammad Badr &amp; Gaurish Thakkar. This was the project done as part of IASNLP summer school

This project deals with extraction of synonyms from raw corpus. We have trained our model on word2vec and used it subsequently to train a ann to check if the pair is synonym or antonym.

To run the project 
1.run the file name model-trainer which creates the unsupervised model
2.run the nn-data-maker which creates the supervised data for the ann
3.run the nn-supervised to create the supervised model and find the accuracy of the classifier
4.run the ann-accuracy for testing the samples

To run 3 class classifier \n
1.run the nn-clean-network for training the ann and checking its accuracy
2.run the ann-accuracy for testing the samples
