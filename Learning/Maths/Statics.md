# Statics

Branch of mathematics that involves collecting, analyzing, interpreting and presenting data.

### Two types of statics:
<b>Descriptive Statics</b> : It's a branch of statics to analysis, summary, and presentation of findings related to a data set derived from a sample or entire population. Descriptive statistics comprises three main categories – Frequency Distribution, Measures of Central Tendency, and Measures of Variability. 

<b>Inferential Statics</b>: Used for prediction for upcoming future using currently available data. Inferential statics is all about prediction about population.

<b>Terminology</b>

`Population` : Entire group of individuals or objects that we studying for analysis.

`Sample`: Subset of population, smaller group of individuals or objects that we select from the population to study.

<b>Example</b>
____

Population Example                                   | Sample Example                                  
-----------------------------------------------------|------------------------------------------------
All cricket fans that watch match(include OTT, TV).  | All cricket fans who are present in the stadium
All students of college(include passout students, online class students as well)| All students who visit lectures.                                    

* Note: Things to be careful about while creating sample
    - sample size
    - biased(random) sample
    - Representation

 
### Types of data

* Categorical or Qualitative Data (Example: Gender, Brands)
    - Nominal Data : No order in categories | ex: male, female
    - Ordinal Data : Order in categorical data | ex: bad, average, good

* Numerical or Quantitative Data (Example: age, height)
    - Discrete Data : Only take integer value. | ex: age, rank
    - Continuous Data : Such data which can take any value | ex: weight, height


### Measure of Central Tendency
___
Statistical measure that represents a typical central value for a dataset provide summary of data by identifying a single value that is most representative of the dataset as a whole.

Ex: We have a list of passengers in a bus with their age now we want to calculate centrality of this data. This measuring called central tendency.

For this measure we need some parameters like mean, median and mode.

- Mean(Average) : Sum of all values in the dataset divided by the number of values.

    <i> Formula : </i> `sum of all values` / `total number of values` 


    Ex: data = [3, 4, 1, 2, 5] Calculate the mean ?

    calculating_mean = 3 + 4 + 1 + 2 + 5 / 5 => 3 (mean = 3)


    Note: Mean is different for population mean and sample mean

    <i>Flaw in mean is called <b>outlier</b>, this means if one of the value is so large in data the mean can't be correct then. So in this kind of case don't use mean as a parameter use median</i>

<br /><br />
- Median : Middle value in the dataset when the data is arranged in order.

    - In case of odd number of data

        data = [3, 4, 1, 2, 5]

        median = 1,2, <b>3</b>, 4, 5

        median = 3

    - In case of even number of data

        data = [3, 4, 1, 2, 5, 6]

        median = 1, 2, <b> 3, 4, </b> 5, 6

        median = (3 + 4) / 2 = 3.5

        median = 3.5

<br />

- Mode : The value that appears most frequently in the dataset.

    data = [1, 2, 1, 3, 1, 5]

    frequency of 1 : 3,  2 : 1, 3 : 1, 5 : 1

    mode of data is : 1 | because 1 is most frequent

    <i>Note : Mostly used in categorical data, discrete data</i>

<br />

- Weighted Mean: Sum of the products of each value and it's weight divided by the sum of the weights. It is used to calculate a mean when the values in the dataset have different importance or frequency.

    Example : 3 models for rent prediction 
    
    Model1 | Importance | Result
  |---------|-----------|----------|
    Model 1 |  0.2      |  10L  
    Model 2 |  0.3      |  15L
    Model 3 |  0.5      |  12L

Weighted mean = (0.2 * 10L + 0.3 * 15L + 0.5 * 12L) / 0.2 + 0.3 + 0.5


<br />

- Trimmed Mean : Calculated by removing a certain percentage of the smallest and largest values from the dataset and then taking the mean of the remaining values. The percentage of values removed is called the trimming values.

    rent_data = []