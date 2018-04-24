This is a text classifier for classifying 8 of the cases :business competition,competitive response,industry analysis,market sizing,new business,organizational behavior,increase sales,mergers & acquisitions.


1.To run, simply type 
python start.py

2.You will be prompted to enter a filename, which is the same filename as that of the csv file in the folder, i.e: consulting.csv

3. You will be prompted to enter whether it is : 1 for multi-label or 2 for multi class
I have a 2 layer Bidirectional GRU Neural Network model trained for the multi-label classification
In case you want to train multi-label classifier, type 1

I have 2 models for multi-class classifier, a Support Vector Machine and a 2 layer Bidirectional GRU Neural Network model.

4.You will be prompted to enter 1 for svm or 2 for neural network.
For multi-label classification, please enter 2
For mult-class classification, you can entter 1 or 2

5.View the results

Dependencies: Python 3+, keras, sklean, pandas, csv, nltk packages
Please install these before using this program.

