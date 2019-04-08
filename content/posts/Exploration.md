title: Heatmap Color Labels in Seaborn
slug: heatmap-color-labels-in-seaborn
category: dataviz
date: 2019-04-05
modified: 2019-04-05

# Multiple Layers of Color Labels in Seaborn Heatmaps

I'm currently working with biological test data, which by its nature tends to have a large number of features. This presents all sorts of challenges, not least of which is the difficulty in interpreting correlation heatmaps when there are so many rows and columns that the labels become impossible to read!

One solution to this problem is to group the features into categories, assign each category a color, and annotate the rows and columns of a heatmap. For a toy example of this using a more manageable non-biological dataset, consider the following:

![Heatmap One]({static}/images/heatmap_one.png)

This is a nice way to interpret the correlation heatmap of a large dataset, as the column and row colors allow you to identify useful clusters by sight. What if, however, each feature has not just one useful attribute for grouping, but two? For those working in life sciences, you might take the example of wanting to be able to know both reagent and antigen by sight.

Fortunately, seaborn makes this easy as well. Let's work through an example using the Residential Building Data Set from the UCI Machine Learning Library.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```


```python
df = pd.read_excel('data/raw/Residential-Building-Data-Set.xlsx')
```


```python
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
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PROJECT DATES (PERSIAN CALENDAR)</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>PROJECT PHYSICAL AND FINANCIAL VARIABLES</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>...</th>
      <th>Unnamed: 99</th>
      <th>Unnamed: 100</th>
      <th>Unnamed: 101</th>
      <th>Unnamed: 102</th>
      <th>Unnamed: 103</th>
      <th>Unnamed: 104</th>
      <th>Unnamed: 105</th>
      <th>Unnamed: 106</th>
      <th>OUTPUTS</th>
      <th>Unnamed: 108</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>START YEAR</td>
      <td>START QUARTER</td>
      <td>COMPLETION YEAR</td>
      <td>COMPLETION QUARTER</td>
      <td>V-1</td>
      <td>V-2</td>
      <td>V-3</td>
      <td>V-4</td>
      <td>V-5</td>
      <td>V-6</td>
      <td>...</td>
      <td>V-22</td>
      <td>V-23</td>
      <td>V-24</td>
      <td>V-25</td>
      <td>V-26</td>
      <td>V-27</td>
      <td>V-28</td>
      <td>V-29</td>
      <td>V-9</td>
      <td>V-10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>81</td>
      <td>1</td>
      <td>85</td>
      <td>1</td>
      <td>1</td>
      <td>3150</td>
      <td>920</td>
      <td>598.5</td>
      <td>190</td>
      <td>1010.84</td>
      <td>...</td>
      <td>815.5</td>
      <td>1755</td>
      <td>8002</td>
      <td>60.74</td>
      <td>54.26</td>
      <td>2978.26</td>
      <td>41407</td>
      <td>601988</td>
      <td>2200</td>
      <td>410</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84</td>
      <td>1</td>
      <td>89</td>
      <td>4</td>
      <td>1</td>
      <td>7600</td>
      <td>1140</td>
      <td>3040</td>
      <td>400</td>
      <td>963.81</td>
      <td>...</td>
      <td>1316.3</td>
      <td>8364.78</td>
      <td>8393</td>
      <td>90.95</td>
      <td>89.79</td>
      <td>11379.4</td>
      <td>44835</td>
      <td>929027</td>
      <td>5000</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>78</td>
      <td>1</td>
      <td>81</td>
      <td>4</td>
      <td>1</td>
      <td>4800</td>
      <td>840</td>
      <td>480</td>
      <td>100</td>
      <td>689.84</td>
      <td>...</td>
      <td>765.8</td>
      <td>1755</td>
      <td>4930</td>
      <td>38.7</td>
      <td>32.04</td>
      <td>1653.06</td>
      <td>37933</td>
      <td>377829</td>
      <td>1200</td>
      <td>170</td>
    </tr>
    <tr>
      <th>4</th>
      <td>72</td>
      <td>2</td>
      <td>73</td>
      <td>2</td>
      <td>1</td>
      <td>685</td>
      <td>202</td>
      <td>13.7</td>
      <td>20</td>
      <td>459.54</td>
      <td>...</td>
      <td>152.25</td>
      <td>1442.31</td>
      <td>1456</td>
      <td>9.73</td>
      <td>8.34</td>
      <td>686.16</td>
      <td>8194</td>
      <td>122032</td>
      <td>165</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 109 columns</p>
</div>



This dataset contains measurements relating to real estate construction projects in Iran. Broadly speaking, these measurements can be grouped into **physical and financial (P&F)** measurements which were recorded once, and **economic (E)** measurements which were recorded at five time points throughout the contruction project. For more information on the features, you may check out the data dictionary here:

First, we'll clean the data so that it contains only the P&F measurements and the E measurements at the final timepoint, converted into the appropriate data type.


```python
# Use the first row as the columns
df.columns = df.iloc[0,:]
# Select only the P&F features, and the E features for one timepoint
df = pd.concat([df.iloc[:,4:12], df.iloc[:, -21:]], axis=1)
# Reorder the columns so that they are in ascending numerical order
col_order = ["V-" + str(i) for i in range(1,30)]
df = df[col_order]
# Drop the extra row of column names and reset the index numbering
df = df.drop(0).reset_index(drop=True)
# Convert the DataFrame to numeric
df = df.apply(pd.to_numeric, axis=1)
```


```python
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
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V-1</th>
      <th>V-2</th>
      <th>V-3</th>
      <th>V-4</th>
      <th>V-5</th>
      <th>V-6</th>
      <th>V-7</th>
      <th>V-8</th>
      <th>V-9</th>
      <th>V-10</th>
      <th>...</th>
      <th>V-20</th>
      <th>V-21</th>
      <th>V-22</th>
      <th>V-23</th>
      <th>V-24</th>
      <th>V-25</th>
      <th>V-26</th>
      <th>V-27</th>
      <th>V-28</th>
      <th>V-29</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>3150.0</td>
      <td>920.0</td>
      <td>598.5</td>
      <td>190.0</td>
      <td>1010.84</td>
      <td>16.0</td>
      <td>1200.0</td>
      <td>2200.0</td>
      <td>410.0</td>
      <td>...</td>
      <td>15.0</td>
      <td>733.800000</td>
      <td>815.50</td>
      <td>1755.00</td>
      <td>8002.0</td>
      <td>60.74</td>
      <td>54.26</td>
      <td>2978.26</td>
      <td>41407.0</td>
      <td>601988.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>7600.0</td>
      <td>1140.0</td>
      <td>3040.0</td>
      <td>400.0</td>
      <td>963.81</td>
      <td>23.0</td>
      <td>2900.0</td>
      <td>5000.0</td>
      <td>1000.0</td>
      <td>...</td>
      <td>15.0</td>
      <td>1143.800000</td>
      <td>1316.30</td>
      <td>8364.78</td>
      <td>8393.0</td>
      <td>90.95</td>
      <td>89.79</td>
      <td>11379.37</td>
      <td>44835.0</td>
      <td>929027.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>4800.0</td>
      <td>840.0</td>
      <td>480.0</td>
      <td>100.0</td>
      <td>689.84</td>
      <td>15.0</td>
      <td>630.0</td>
      <td>1200.0</td>
      <td>170.0</td>
      <td>...</td>
      <td>15.0</td>
      <td>589.500000</td>
      <td>765.80</td>
      <td>1755.00</td>
      <td>4930.0</td>
      <td>38.70</td>
      <td>32.04</td>
      <td>1653.06</td>
      <td>37933.0</td>
      <td>377828.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>685.0</td>
      <td>202.0</td>
      <td>13.7</td>
      <td>20.0</td>
      <td>459.54</td>
      <td>4.0</td>
      <td>140.0</td>
      <td>165.0</td>
      <td>30.0</td>
      <td>...</td>
      <td>12.0</td>
      <td>197.679557</td>
      <td>152.25</td>
      <td>1442.31</td>
      <td>1456.0</td>
      <td>9.73</td>
      <td>8.34</td>
      <td>686.16</td>
      <td>8194.0</td>
      <td>122031.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>3000.0</td>
      <td>800.0</td>
      <td>1230.0</td>
      <td>410.0</td>
      <td>631.91</td>
      <td>13.0</td>
      <td>5000.0</td>
      <td>5500.0</td>
      <td>700.0</td>
      <td>...</td>
      <td>14.0</td>
      <td>2220.600000</td>
      <td>2244.10</td>
      <td>9231.76</td>
      <td>9286.0</td>
      <td>136.60</td>
      <td>140.20</td>
      <td>9821.00</td>
      <td>48260.0</td>
      <td>1734973.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



Excellent. We now have useable data for generating clustered correlation heatmaps. If we do that now, we get the following result:


```python
sns.clustermap(df.corr(), cmap='bwr')
```




    <seaborn.matrix.ClusterGrid at 0x7fd12bb00278>




![png](images/output_9_1.png)


There are some strong patterns here, but the labels aren't very useful. It would be nice to see if these are grouped by our categories of features, P&F and E.

In order to accomplish this, we first take our list of columns and split them into their respective groups.


```python
physical_financial = col_order[:10]
economic = col_order[10:]
```

Next, we will need some colors. Seaborn makes this easy through the `color_palette()` function.


```python
palette = sns.color_palette()
palette
```




    [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
     (1.0, 0.4980392156862745, 0.054901960784313725),
     (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
     (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
     (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
     (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
     (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
     (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
     (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]



To see what these colors look like, seaborn offers the useful `palplot()` function.


```python
sns.palplot(palette)
```


![png]({static}/images/output_15_0.png)


Very nice! In order to assign these colors to categories, seaborn will want a Series with the colors as values, and the associated features as index labels. Let's create that.


```python
# Create dictionary with features as keys and colors as values
color_dict = {}
for col in df.columns:
    if col in physical_financial:
        color_dict[col] = palette[0]
    else:
        color_dict[col] = palette[1]
# Convert the dictionary into a Series
color_rows = pd.Series(color_dict)
color_rows.head()
```




    V-1    (0.12156862745098039, 0.4666666666666667, 0.70...
    V-2    (0.12156862745098039, 0.4666666666666667, 0.70...
    V-3    (0.12156862745098039, 0.4666666666666667, 0.70...
    V-4    (0.12156862745098039, 0.4666666666666667, 0.70...
    V-5    (0.12156862745098039, 0.4666666666666667, 0.70...
    dtype: object



In order to assign this color mapping to the clustered heatmap, we simply assign it to the `row_colors` and `col_colors` optional arguments.


```python
sns.clustermap(df.corr(), cmap='bwr', row_colors=[color_rows], col_colors=[color_rows])
```




    <seaborn.matrix.ClusterGrid at 0x7fd12b254080>




![png]({static}/images/output_19_1.png)


Very nice. Now, let's add a second layer. We might also want to know at sight what kind measurement each feature contains. Let's have a look at the data dictionary in order to determine this.


```python
# Load the data dictionary in the second page of the Excel file
desc = pd.read_excel('data/raw/Residential-Building-Data-Set.xlsx', sheet_name=1)
desc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable Group</th>
      <th>Variable ID</th>
      <th>Descriptions</th>
      <th>Unit</th>
      <th>Time Lag Number p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PROJECT PHYSICAL AND FINANCIAL VARIABLES</td>
      <td>V-1</td>
      <td>Project locality defined in terms of zip codes</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>V-2</td>
      <td>Total floor area of the building</td>
      <td>m2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>V-3</td>
      <td>Lot area</td>
      <td>m2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>V-4</td>
      <td>Total preliminary estimated construction cost ...</td>
      <td>10000000 IRRm</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>V-5</td>
      <td>Preliminary estimated construction cost based ...</td>
      <td>10000 IRRm</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert the Unit column to string type so that we can get unique values
desc.Unit = desc.Unit.astype(str)
# Get unique unit types
units = desc.Unit.unique()
```


```python
units
```




    array(['nan', 'm2 ', '10000000 IRRm ', '10000 IRRm ',
           'As a number of time resolution e ', '10000 IRRm', 'm2',
           '10000000 IRRm', '%', '10000 IRRm /m2', 'IRRm'], dtype=object)



It seems as though we measurements in currency (IRRm), in area (m2), and some other miscellaneous types of measures. Let's make a new mapping using the same pattern.


```python
unitmap = {}

for unit in units:
    if "IRRm" in unit:
        unitmap[unit] = palette[2]
    elif "m2" in unit:
        unitmap[unit] = palette[3]
    else:
        unitmap[unit] = palette[4]
```


```python
desc['Color'] = desc.Unit.map(unitmap)
desc[['Variable ID  ', 'Unit', 'Color']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable ID</th>
      <th>Unit</th>
      <th>Color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V-1</td>
      <td>nan</td>
      <td>(0.5803921568627451, 0.403921568627451, 0.7411...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V-2</td>
      <td>m2</td>
      <td>(0.8392156862745098, 0.15294117647058825, 0.15...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V-3</td>
      <td>m2</td>
      <td>(0.8392156862745098, 0.15294117647058825, 0.15...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V-4</td>
      <td>10000000 IRRm</td>
      <td>(0.17254901960784313, 0.6274509803921569, 0.17...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V-5</td>
      <td>10000 IRRm</td>
      <td>(0.17254901960784313, 0.6274509803921569, 0.17...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Delete extraneous row at end of dictionary
desc.drop(29, inplace=True)
```


```python
# Get only features and colors for mapping
color_rows_two = desc[['Variable ID  ', 'Color']]
# Set features as index
color_rows_two = color_rows_two.set_index('Variable ID  ')
# Delete the index name for cleanliness
del color_rows_two.index.name
# Use iloc to convert DataFrame into Series
color_rows_two = color_rows_two.iloc[:,0]
color_rows_two.head()
```




    V-1    (0.5803921568627451, 0.403921568627451, 0.7411...
    V-2    (0.8392156862745098, 0.15294117647058825, 0.15...
    V-3    (0.8392156862745098, 0.15294117647058825, 0.15...
    V-4    (0.17254901960784313, 0.6274509803921569, 0.17...
    V-5    (0.17254901960784313, 0.6274509803921569, 0.17...
    Name: Color, dtype: object



Having completed all this, we simple pass both Series into `row_colors` and `col_colors` as a list. It is that simple.


```python
sns.clustermap(df.corr(), cmap='bwr', row_colors=[color_rows, color_rows_two], col_colors=[color_rows, color_rows_two])
```




    <seaborn.matrix.ClusterGrid at 0x7fd12a929588>




![png]({static}/images/output_30_1.png)


Voila! Two layers of row and column colors, for easy interpretation of feature clusters by groups.

In my next post, I'll cover how to make custom legends for these color labels using matplotlib. Thanks for reading, and stay tuned!
