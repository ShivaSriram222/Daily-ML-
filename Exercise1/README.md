# Data sanity checks

Checks on Data Sanity (Titanic Baseline Model)

#### Exercise Overview

- Fit a baseline logistic regression model using the Titanic data set to estimate the survival of passengers.

- Put in controlled breaks in the data to look at sensitivity of models:

- Shuffling labels - approximate corrupted target data.

- Dropped Age feature - feature simulation by missing.

- mimimaximum scaling of Fare - mimic preprocessing error.

#### Objective

- Know the impact of data quality and preprocessing on model performance.

- Write debugging ML pipelines.

- Intuition to feature importance, labelling quality and numerical stability.

#### Experiment Results
Model	Change	Accuracy	Observations
Baseline	None	~80%	Normal model training
Model 1	Shuffle labels	50 percent (5)	Catastrophic drop - labels are critical.
Model 2 Lap drop age - has feature importance - moderate drop.
Model 3	Scale Fare x1000 = -75-78% = Slight instability - preprocessing is important.
Recombined	All 3 changes	~50%	Multiple issues compound - catastrophic failure.

#### Key Learnings

- Labels are important - tracing labels kills predictive strength completely.

- The value of features is significant - the removal of Age illustrates the role of important features.

- Preprocessing may cause distortions in models - extreme scaling is changing model coefficients.

- Compounding problems - taking several "breaks" will result in disastrous performance.

- debug models in a systematic way: check labels - features - preprocessing - assumptions.

#### Next Steps / Reflections

- Always check target data and feature quality prior to training.

- Use powerful preprocessing pipelines to avoid scaling errors.

- Observations of documents and experiments - helpful in debugging and reproducibility.



This information is prepared to be posted as either:

README.md in the Exercise 1  folder, or

a file in your root repo README on Exercise 1  - Data Sanity Checks.
