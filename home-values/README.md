# Estimating Home Prices


Estimating home values is an important aspect of the Windfall net worth model.  Property often represents a significant portion of an individual’s net worth and, due to changing home values, can be a significant driver of wealth growth. 

Your task is to create a model using the attached dataset to attempt to predict home values in Denver.

The dataset contains information on approximately 15,000 single-family homes in Denver, CO.  Each row contains the following information about the property that can be used to estimate the home’s value:

* id
* address 
* city
* state
* zipcode
* latitude
* longitude
* bedrooms
* bathrooms
* rooms
* square footage
* lot size (ft^2)
* year built
* last sale date
* last sale amount
* prior sale date
* prior sale amount

In addition, we have included the Zillow “Zestimate” (http://www.zillow.com/zestimate) of each home’s current value in the `estimated_value` column of the attached file.  You should use these estimates to train your model.

We have reserved a set of single family home records from Denver that we will use to assess the quality of your model.  The reserve dataset is in the same format as the training set.

Please implement a model using either python or R.  When you are finished, send us back:

1. Your code
2. Instructions for running your model on our reserve dataset
3. A document explaning how you arrived at your answer (see below)

Here are some points you may want to cover when writing your explanation:
* What was your methodology in assessing the data?
* How did you decide which features were important and which ones were of no value?
* Did you consider any strategies other than the one you used for your final model?  What were they?
* How did you train your model?
* How did you evaluate the accuracy of your model?

Please feel free to include in your document any graphs or charts that you created during the process of either
data exploration of model evaluation. 

