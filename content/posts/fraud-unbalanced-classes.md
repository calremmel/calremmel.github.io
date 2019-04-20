title: Fraud Detection in Python: Part One
slug: fraud-detection-part-one
category: fraud-detection
date: 2019-04-20
modified: 2019-04-20

# Visualization and Unbalanced Classes

![png]({static}/images/pca_fraud_title.png)

Fraud detection is an important area of business where machine learning techniques have a particularly powerful use case. While fraud detection as a discipline predates the widespread popularity of machine learning, traditional techniques rely primarily on rules of thumb for flagging potentially fraudulent behavior. These rules can yield impressive results, but they cannot deal with interactions between different variables or improve over time the way a machine learning model can.

This is part one in a multi-part deep dive into the latest and greatest in fraud detection techniques using machine learning in the Python programming language. I'll be going through the very latest methods for predicting rare events like fraud, starting from the basics and proceeding through the cutting edge.

This series will be a set of living documents, and I will be updating them as much as I can as new ideas arise and as I learn more myself! Updates will be logged at the bottom of the page, so check there to see if there is anything new.

# Visualizing Fraud

Part of what makes it so difficult to detect fraud is that **fraud is rare.** Within any given dataset of transactions, there will usually be far more legitimate cases than fraudulent ones. This makes it difficult to build a profile for what might distinguish fraud from non-fraud, and it makes it very difficult to avoid flagging legitimate transactions as fraud inadvertently when you try.

To get a sense of this, it is useful to have a way to visualize the frequency of fraud versus non-fraud. That's where we'll start.

Throughout this series, we'll be working with **Synthetic Financial Datasets For Fraud Detection,** generously provided by the Digital Forensics Research Group from NTNU in Gjøvik, Norway. An explanation of the dataset, the data dictionary, and the data itself are all available on Kaggle. [Feel free to download it if you would like to follow along](https://www.kaggle.com/ntnu-testimon/paysim1).

First, let's import the libraries we'll need, and take a first glance at the dataset.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

%matplotlib inline
```


```python
df = pd.read_csv('../data/raw/PS_20174392719_1491204439457_log.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>step</th>
      <th>type</th>
      <th>amount</th>
      <th>nameOrig</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
      <th>isFlaggedFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>9839.64</td>
      <td>C1231006815</td>
      <td>170136.0</td>
      <td>160296.36</td>
      <td>M1979787155</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>1864.28</td>
      <td>C1666544295</td>
      <td>21249.0</td>
      <td>19384.72</td>
      <td>M2044282225</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>TRANSFER</td>
      <td>181.00</td>
      <td>C1305486145</td>
      <td>181.0</td>
      <td>0.00</td>
      <td>C553264065</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>CASH_OUT</td>
      <td>181.00</td>
      <td>C840083671</td>
      <td>181.0</td>
      <td>0.00</td>
      <td>C38997010</td>
      <td>21182.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>11668.14</td>
      <td>C2048537720</td>
      <td>41554.0</td>
      <td>29885.86</td>
      <td>M1230701703</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6362620 entries, 0 to 6362619
    Data columns (total 11 columns):
    step              int64
    type              object
    amount            float64
    nameOrig          object
    oldbalanceOrg     float64
    newbalanceOrig    float64
    nameDest          object
    oldbalanceDest    float64
    newbalanceDest    float64
    isFraud           int64
    isFlaggedFraud    int64
    dtypes: float64(5), int64(3), object(3)
    memory usage: 534.0+ MB
    None


This is a large dataset with well over six-million entries. Let's have a look at the data dictionary so we know what kind of information we have:

* **step** - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).
* **type** - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
* **amount** - amount of the transaction in local currency.
* **nameOrig** - customer who started the transaction
* **oldbalanceOrg** - initial balance before the transaction
* **newbalanceOrig** - new balance after the transaction
* **nameDest** - customer who is the recipient of the transaction
* **oldbalanceDest** - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).
* **newbalanceDest** - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).
* **isFraud** - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
* **isFlaggedFraud** - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

It appears that part of our task will be to improve as much on `isFlaggedFraud` as possible. This is ideal, as it represents a typical rules-based heuristic for identifying fraud, and this is exactly the kind of benchmark we want to show we can outperform. We'll have a look at exactly how well they did at a later point in the series.

Next question: How many and what percentage of these entries are fraud?


```python
counts = df['isFraud'].value_counts()
counts
```




    0    6354407
    1       8213
    Name: isFraud, dtype: int64




```python
counts / df.shape[0]
```




    0    0.998709
    1    0.001291
    Name: isFraud, dtype: float64



Herein lies the central problem: only a little over **.1%** of these entries are fraudulent.

In order to make this more visually apparent, we'll use the following process:

1. We'll take a large random sample from the dataset. I've learned from hard experience that plotting millions of points on your home laptop can take a good long while otherwise
2. We'll drop `nameOrig` and `nameDest` for now. These may be useful for modeling at a later point, but they will require some transformation, and for now we just want to get a sense of scale
3. We'll transform the `type` column into numeric dummy variables, so as to be able to include this information in our modeling
4. We'll reduce the dataset into two dimensions using Principal Component Analysis for plotting
5. We'll create a scatterplot of the reduced data, colored by fraudulent vs legitimate transactions


```python
# Take random sample of dataset
sample = df.sample(n=10000, random_state=42)
```


```python
# Check to make sure that the sample is representative in terms of class ratio
sample.isFraud.value_counts()
```




    0    9981
    1      19
    Name: isFraud, dtype: int64




```python
# Create a function to select and process features
def get_features(df):
    """Selects and prepares features for plotting and modeling
    
    Args:
        df (DataFrame): Unprocessed fraud data
    Returns:
        df (DataFrame): Processed fraud data
    """
    selected_cols = [
        'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'isFraud'
    ]
    
    df = df[selected_cols].copy()
    dummies = pd.get_dummies(df.type)
    df = pd.concat([df, dummies], axis=1).drop("type", axis=1)
    
    return df
```


```python
# Create a function to perform PCA on the features and return a DataFrame for plotting
def reduce_data(pca_df):
    """Returns features for plotting proportion of fraud
    
    Args:
        df (DataFrame): Synthetic fraud database
    Returns:
        plot_df (DataFrame): DataFrame with two principal components and target variable
    """
    pca_df = pca_df.copy()
    target = pca_df.pop("isFraud")
    scaler = StandardScaler()
    pca_df = scaler.fit_transform(pca_df)
    pca = PCA(n_components=2)
    components = pca.fit_transform(pca_df)

    comp_df = pd.DataFrame(components, columns=["X", "y"])
    target = target.reset_index(drop=True)
    plot_df = pd.concat([comp_df, target], axis=1)
    
    return plot_df
```


```python
plot_df.columns
```




    Index(['X', 'y', 'isFraud'], dtype='object')




```python
# Create a function for plotting
def fraud_plot(plot_df, maj_alpha=0.5, min_alpha=1, save=None):
    """Plots reduced data
    
    Args:
        plot_df (DataFrame): Reduced data
        maj_alpha (float): Transparency setting for majority class
        min_alpha (float): Transparency setting for minority class
        save (str): Filename for saving plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.scatterplot(x="X", y="y", alpha=maj_alpha, data=plot_df[plot_df.isFraud == 0], label="Legitimate")
    sns.scatterplot(x="X", y="y", alpha=min_alpha, data=plot_df[plot_df.isFraud == 1], ax=ax, label="Fraud")
    plt.title("Legitimate vs Fraudulent Purchases")
    plt.tight_layout()
    if save != None:
        plt.savefig(save)
    plt.show()
    
    pass
```


```python
# Produce the plot!
processed_data = get_features(sample)
plot_df = reduce_data(processed_data)
fraud_plot(plot_df)
```

    /home/calre/anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:645: DataConversionWarning: Data with input dtype uint8, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /home/calre/anaconda3/lib/python3.7/site-packages/sklearn/base.py:464: DataConversionWarning: Data with input dtype uint8, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, **fit_params).transform(X)



![png]({static}/images/fraud_pca_1.png)


There we go! The fraudulent entries are fairly well clustered, but they are very similar to many legitimate entries, and are completely dwarfed in number.

Having built these functions, we should now be able to visualize the changes we make to the distribution of our dataset as we move ahead into resampling.

# Resampling Techniques

As we've seen, one of the central challenges in fraud detection is that cases of fraud are incredibly rare in comparison to legitimate transactions. This makes training a machine learning model challenging, because it means that if 1% of all transactions are fraudulent (this is still unrealistically high), the model would be able to achieve 99% accuracy simply by guessing that every single transaction is legitimate. Not what we want! This situation where there is much more of one class than another is called an **imbalanced class problem.**

The way we deal with imbalanced classes is through **resampling,** which covers an array of techniques for making the ratio of classes more equitable. These techniques may be broadly separated into three categories:

1. **Random Under-Sampling (RUS):**
    * Take a random sample of the majority class and train the model on the sample combined with the entirety of the minority class
2. **Random Over-Sampling (ROS):**
    * Randomly sample from the minority class *with replacement* until the sizes of both classes match
3. **Synthetic Minority Over-sampling Technique (SMOTE):**
    * Creating new synthetic data for the minority class using K-Nearest Neighbors to create new minority cases that are representative but not exact duplicates

How to choose? **RUS** may be appropriate when there is a large enough population of the minority class that a representative sample of the majority class will make the class sizes equitable. RUS may be a good option in our case, as given our majority class size of 6354407 and minority class size of 8213, a sample size of 9589 for the majority class would give us a representative sample with a confidence of 95% and a margin of error of 1%.

For over-sampling techniques, given that our cases of fraud are similar enough to each other that KNN might produce good representative data, **SMOTE** could also produce good results. If our fraud cases were more spread out, creating synthetic data using KNN could produce wildly unrepresentative noise, throwing the model off.

If there is a situation where **ROS** would be preferable to both RUS and SMOTE, I'm not sure what that is! Training the model on duplicate data is a huge drawback, and could lead to serious overfitting on very specific data points.

In our case, as both **RUS** and **SMOTE** are solid options, we will try them both.

## Random Under-Sampling

Both of our resampling strategies are implemented in the `imblearn` library. The documentation may be found [by clicking here](https://imbalanced-learn.readthedocs.io/en/stable/index.html). I'll explain the parameters we'll be using as we go along.


```python
from imblearn.under_sampling import RandomUnderSampler
```

`RandomUnderSampler` does exactly what it says on the tin. The important parameter we need to tweak is called `sampling_strategy`. If you don't touch this setting, `RandomUnderSampler` will simply make the sizes of the larger classes match the size of the smallest class through random sampling without replacement. However, we want to make sure that we have a representative sample for the majority class, and using [this calculator](https://www.qualtrics.com/blog/calculating-sample-size/) while asking for a 95% Confidence Level with a 1% Margin of Error suggests a slightly larger sample size of 9589.

Fortunately `sampling_strategy` can optionally take a dictionary, where the keys are class names, and the values are sample sizes. This allows us to set the exact sample size we want for the majority class.


```python
# Instantiate RandomUnderSampler
RUS = RandomUnderSampler(sampling_strategy={0: 9589}, random_state=42)
```


```python
# Create a function for use with any resampling method
def resample(df, method):
    """Resamples df using method with .fit_resample()
    
    Args:
        df (DataFrame): Fraud data
        method (object): Resampler with .fit_resample() method
    Retuns:
        resampled_df (DataFrame): Resampled DataFrame
    """
    processed_df = get_features(df)
    target = processed_df.pop('isFraud')

    processed_x, processed_y = method.fit_resample(processed_df, target)

    cols = list(processed_df.columns) + ["isFraud"]

    pdf_x = pd.DataFrame(processed_x, columns=processed_df.columns)
    pdf_y = pd.DataFrame(processed_y, columns=['isFraud'])
    resampled_df = pd.concat([pdf_x, pdf_y], axis=1)
    
    return resampled_df
```


```python
# Apply RandomUnderSampler to data
rus_resampled = resample(df, RUS)
print(rus_resampled.shape)
print(rus_resampled.isFraud.value_counts())
```

    (17802, 11)
    0    9589
    1    8213
    Name: isFraud, dtype: int64



```python
fraud_plot(reduce_data(rus_resampled), min_alpha=0.5)
```


![png]({static}/images/fraud_pca_2.png)


The result of our work is a nicely balanced dataset, using all real data and a representative random sample of the majority class.

Let's wrap up by applying SMOTE, which should be a snap given that we've already prepared the major groundwork using the functions we've written to this point.

## SMOTE


```python
from imblearn.over_sampling import SMOTE
```


```python
# Instantiate SMOTE
SM = SMOTE(random_state=42)
```


```python
# Use our helpful resampling function
sm_resampled = resample(df, SM)
print(sm_resampled.shape)
print(sm_resampled.isFraud.value_counts())
```

    (12708814, 11)
    1    6354407
    0    6354407
    Name: isFraud, dtype: int64



```python
# Take random sample for easier plotting
sm_sample = sm_resampled.sample(n=10000, random_state=42)
# Produce the plot!
fraud_plot(reduce_data(sm_sample), min_alpha=0.5)
```


![png]({static}/images/fraud_pca_3.png)


Again, we have a well balanced dataset. This dataset is much larger than the one produced by RUS, but contains a lot of synthetic data points. Because our original fraudulent data was so closely clustered, this hopefully shouldn't compromise the quality of our predictions. Notice that our plot is visually quite similar to the one produced by RUS.

# Next Steps

That's enough for our first foray into the fundamentals of fraud detection in Python! To wrap up, here are the major takeaways in condensed form:

1. Fraud is damaging, but it is also **very rare.** This is the central challenge for modeling fraud
2. A dataset where one class of interest is wildly outnumbered by other classes is said to be **imbalanced**
3. It is useful to have visual strategies to get a sense for the scope of class imbalance in a dataset. We've learned to use **PCA and scatterplots** in order to visualize these relationships
4. In order to prepare an imbalanced dataset for modeling, we need to apply **resampling strategies** to make the class balance more equitable. We learned about **Random Under-Sampling (RUS)** of the majority class, **Random Over-Sampling (ROS)** of the minority class, and **Synthetic Minority Over-sampling Technique (SMOTE)** for generating new representative minority class data

In the next part of our deep dive, we'll cover common machine learning techniques for modeling fraud, and how to use them in pipelines with Scikit-Learn. Thanks for reading, and stay tuned!
