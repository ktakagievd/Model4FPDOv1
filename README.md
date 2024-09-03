# Model4FPDOv1

How to use
(i) Run "main.py"
(ii) The result (selected gene list and others) will be saved in the folder, "./dataset/res/geneselect/" 

Input data:
Substitute the input data sets for the sample data in the folder, "./dataset/input/"
(a) Image data: (n, 64,64,3) dimension image patch data (color image with 64 size (64, 64, 3))
(b) Label data: n dimension label from 0~N (the number of original source image) for each image
(c) Gene expression data: N (number of labels)×(number of gene) expression data
(d) Gene name: Gene name for expression data (default: Genbank accession number)

Parameters:
Change the parameters in the file, "./seq/settings.py"

Version:
Python 3.7, Tensorflow 2.1 Numpy 1.18.5, panda 1.2.0, Scipy 1.5.0
