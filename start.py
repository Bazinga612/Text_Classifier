import utilities

print('Please enter the filename')
file_name = input()

print('The classification is done for 8 Case-types: business competition,competitive response,industry analysis,market sizing,new business,organizational behavior,increase sales,mergers & acquisitions.There are 2 classifiers:svm and neural network for multi-class and neural network for multilabel')
print('Please enter 1 for multi-label or 2 for multi class')

multi_class = int(input())

print('Please enter 1 for svm or 2 for neural network')
svm = int(input())
utilities.invoke_classifier(file_name,multi_class,svm);





