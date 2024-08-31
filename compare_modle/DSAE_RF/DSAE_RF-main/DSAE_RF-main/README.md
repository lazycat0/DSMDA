# DSAE_RF
**Predicting potential microbe-disease associations based on multi‑source features and deep learning**
# Experimental steps
Step 1, we calculate the four similarity networks of microbes and diseases based on the preprocessed data, and integrate the four similarity networks, respectively.
Step 2, based on the similarity network between microbes and diseases, construct sample features, that is, characteristics of disease-microbe pairs.
Step 3, select reliable negative samples through k-means clustering method, and the number of negative samples is approximately equal to the number of positive samples.
Step 4, select the effective features of the sample through the DSAE model.
Step 5, classify samples by RF model and predict potential microbe-disease associations.
# Run steps
Run DSAE_RF.py to train the model and obtain prediction scores for microbe-disease associations.
# Requirements
* DSAE_RF is implemented to work under Python 3.8.
* torch
* numpy
* pandas
* sklearn
* math
