
# coding: utf-8
# Categorical Naive Bayes

# Import the libraries needed
# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import itertools

# Other libraries we will need
# import seaborn as sns   # for creating a heatmap of correlated variables
# import matplotlib.pyplot as plt  # for creating plots of the data



# Some utility functions:

def strip_white_space(df):
        '''
        Get rid of leading and trailing white space from all categorical variables in df.
        This function can also be used outside the Categorical Naive Bayes class
        '''
        
        # select out the categorical columns (right now Python calls them "object")
        catData = df.select_dtypes(['object'])


        # Get rid of leading and trailing white space in dataframe values
        newData = df.copy()
        
        newData[catData.columns] = catData.apply(lambda x: x.str.strip())
        return newData
    
def update_dtype(oldCol, dataType):
    '''
    This function changes the data type of the given column oldCol to the type specified in dataType.
    It is designed to work with the pandas apply method.
    '''
    
    newCol = oldCol.astype(dataType)
    return newCol
 
    
def convert_to_cat(df):
    '''
    Pandas treats all string variables as "objects".  Here, we convert such columns to the datatype "category"
    '''
    
    # select out the categorical columns (right now Python calls them "object")
    catData = df.select_dtypes(['object'])
        
    catData = catData.apply(func = update_dtype, dataType = "category")
    return catData



# Taken from:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()





# create the categorical naive bayes model
# first create the naive bayes object, and then the class that will perform the fit method

class CatNaiveBayesObj:
    
    def __init__(self, df, alpha = 1e-3):
        self.type = "Categorical NB Object"
        self.alpha = alpha
        self.data = strip_white_space(df)
        self.catData = convert_to_cat(self.data)
        self.numData = df.select_dtypes(['number']).assign(Income = self.catData.Income)
        self.corr = self.numData.corr()
        self.aggregateTotals = self.catData.groupby(["Income"]).count()
        self.aggregateCounts = self._summary_counts()
        self.priorProbs = np.log(self.catData.groupby(["Income"]).Income.count()) - np.log(self.catData.Income.count())
        
    
    

    def _summary_counts(self):
        '''
        For each level of each categorical variable in self.catData, return the total number of occurrences per
        target variable level
        '''
        
        df = self.catData
        countSummary = {k + "_Counts" : df.groupby(["Income", k])[k].count() for k in df.columns}
        
        return countSummary
        
    
    def _compute_conditionals(self, feature):
        '''
        For the given feature variable, compute the
        conditional probability p(X = feature | Y = cl) for each class cl in the target variable, 
        by counting the number of occurences of feature in the
        training set, when observed class was cl. Use the Laplacian alpha and return the log likelihoods as a
        dataframe, where the columns correspond to the different values of cl, and the rows 
        correspond to different values of the feature variable. 
        '''
    
        df = self.aggregateCounts
        alpha = self.alpha
        
        Numerator = pd.DataFrame({cl : df[feature + "_Counts"][cl, ] for cl in self.catData.Income.unique()}).fillna(value = alpha)


        multiplier = self.catData[feature].nunique()
        Denominator = self.aggregateTotals.loc[:, feature] + multiplier*alpha

    
        # We will look at the log likelihood
        result = np.log(Numerator/Denominator)
        return result
    
    
    def log_likelihoods(self):
        '''
        Compute the dictionary where for every key = categorical variable in self.catData, a df is returned 
        with the conditional log likelihoods of all observations in self.catData
        '''

        logLikelihoods = {k + "_Probs" :  self._compute_conditionals(k) for k in self.catData.columns[:-1]}
        
        return logLikelihoods
        
        
    def _sum_probs(self, obs):
        '''
        Use the censusProbabilities data to compute the likelihoods Y = ">50K" vs Y = "<=50K" of the observation obs
        '''
        
        log_lik = self.log_likelihoods()
        useObs = obs[:-1]
        
        useCols = useObs.notna()   # be careful to remove NA values from the calculation
        r = useObs[useCols]
        ncol = len(r)
        
        colNames = self.catData.columns[:-1]
        dictNames = colNames[useCols] + "_Probs"
        sum = 0
    
        for k in range(ncol) :            
            j = dictNames[k]
            sum += log_lik[j].loc[r[k],]
            
        return sum

    
    def predict(self, testDF, includePriors = True, result = "all", loglik = True):
        '''
        Return the Naive Bayes generated log-likelihoods, together with the predicted class, as a dataframe
        If includePriors is True (the default), the result includes the prior porbabilities for the 
        different levels of the target variable. One reason to exclude the priors is if the results from this
        prediciton will be combined with another naive bayes, for example on the numerical features.
        result determines whether the computed probabilities alone, the class alone, or both are returned
        loglik determines whether the probabilites are returned as log-likelihoods (default), or
        reconverted into probabilities first)
        '''  
        
        if testDF is None:
            useDF = self.catData
        else:
            useDF = strip_white_space(testDF)
            useDF = convert_to_cat(useDF)
         
        
        tempResult = useDF.apply(self._sum_probs, axis = 1)
        if includePriors:
            tempResult += self.priorProbs
        
        if not(loglik):
            tempResult = np.exp(tempResult)
        
        if result == "prob":
            nbResult = tempResult
        else:    
            nbResult = tempResult.assign(MaxClass = tempResult.idxmax(axis = 1))
            if result == "class":
                nbResult = nbResult.MaxClass
                
        return nbResult


    
    
class CategoricalNaiveBayes:  
    
    def __init__(self, alpha = 1e-3):
        self.version = "Categorical NB"
        self.alpha = alpha
        
        
    def fit(self, fitDF):
        '''
        Return the Naive Bayes generated log-likelihoods, together with the predicted class
        '''  
        
        fitSelf = CatNaiveBayesObj(fitDF, self.alpha)
            
        return fitSelf

    

if __name__ == "__main__":
    app.run(debug=True)
