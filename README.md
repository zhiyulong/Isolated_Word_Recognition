# Isolated Word Recognition 

Code: [main.py](https://github.com/zhiyulong/Isolated_Word_Recognition/blob/main/main.py)

- Generate GMM-HMM models for each training set.
- Test the GMM-HMM models with the testing data. 
- Visualization for data analysis
- Print the Training and Testing process while running the code.
- Produce plots showing the models performance and comparing the test result for different features.

## Proposed Method

To implement the Isolated Word Recognition system, using GMM-HMM models trained on Mel Frequency Cepstral Coefficients (MFCC) features extracted from speech signals. The system trains multiple GMM-HMM models on different sets of speech signals and tests them on a separate set of speech signals to classify spoken words. Using K-Means clustering to group the MFCC features into multiple Gaussian mixture models (GMM) for each state of the HMM. The resulting GMM-HMM models are then used to classify the spoken words based on the highest likelihood of the observed speech signal. Compared to other methods, such as deep neural network-based models, GMM-HMM models are computationally less expensive and require less training data. 


## Data

Path: ./train

- The training data contains 16 * 5 + 2 different words spoken by 16 speakers, with five audios per training set and training set #13 containing two more audios. 
    
- Training data sets #1 to #11 contain only double-word audios (n-state = 6), while #12 to #16 contain only single-word audios (n-state = 3).

Path: ./test

- Every 7 test data should belong to one certain train set (one speaker). There are 112 (7 * 16) test audio files in total.

