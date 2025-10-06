# FS4MT : Features selection for multi-table data
# Which variable is the most informative in a multi-table schema ?

## Context

The methodology presented here applies to a multi-table classification problem, where the input data is flattened by an [auto-feature engineering](https://khiops.org/learn/autofeature_engineering/) step. 
More particularly, the *feature selection for multi-table data* tool is used to filter *columns in secondary tables* in within a *multi-table star schema*, prior to supervised classification.

Current tools only take into account variables from the main table, so we propose a method for exploring data from secondary tables in order to extract parameters relevant to classification. By selecting only information-bearing parameters, the classification task is lightened, resulting in reduced processing time. The parameters considered here are native secondary variables (variables present in secondary tables) and construction primitives (mathematical rules).

This system initially focused on creating a measure of the importance of variables and primitives in relation to the target variable. The final objective is to reduce the space of primitives and variables to select only the relevant elements.

**Multi-table star data**

A *multi-table star schema* is a structure where information is organised in several data tables : a main table containing at least an identifier and a target, and one or more secondary tables. The target is the category to which the individual belongs.

<img src="fig_MT.jpg" width="300px"><br>

**Supervised classification and AUC**

Supervised classification is a machine learning approach for categorizing objects into classes.
Modeling quality can be measured by the AUC (area under the ROC curve, between 0 and 1, 0.5 being random modeling and 1 perfect modeling).


## Prerequisites

We use Khiops for classification : [khiops](https://www.khiops.org).

The tool requires the installation of the python packages : [khiops](https://www.khiops.org/setup/), tqdm and tabulate.

In addition to the data tables, the user must provide a dictionary in *kdic* format as input. This file contains the data description : [dictionary](https://https://khiops.org/tutorials/kdic_intro/#what-is-a-khiops-dictionary).
 
## Installation

pip install git+https://github.com/KhiopsLab/feature-selection-for-multi-table-data.git


## General approach

<img src="fig_MT_flattening.jpg" width="700px"><br>
Flattening variables in secondary tables leads to the creation of a large number of aggregates.

<img src="fig_MT_flattening_with_selection.jpg" width="700px"><br>
Selecting variables before creating aggregates reduces search space and could improve the classifier.

The importance measurement method used is a univariate one: each variable is evaluated independently of the others. For each variable, aggregates are constructed using only that variable. The importance used is the maximum level of all aggregates, provided by Khiops.


## Noise in multi-table data

In flat table, noise provides no information. But what about multi-table data ?

==> **The noise in secondary tables can provide information and degrades performance.**

**Why ?** Because the number of instances in secondary tables carries information.

**Solution :** A discretization according to *Count* intervals is performed to limit the effect of noise on the secondary tables.

For further information, see the following article : [Sélection of secondary features from multi-table data for classification](https://editions-rnti.fr/?inprocid=1002995)


## Example on the Accidents dataset

We consider the *Accidents* dataset with 2 secondary tables, *Users* and *Vehicles* directly linked to the main table.

<img src="fig_accidents.jpg" width="400px"><br>

We have added numerical and categorical noise variables to each of the 2 tables. Applying Count discretization, the *Users* table is discretized into 3 intervals (1 user, 2 users, 3 or more users) and the *Vehicles* table into 2 intervals (1 vehicle, 2 or more vehicles).

For the *Users* table, the information contained in the noise variables (N_5 to C_9) is strongly attenuated thanks to discretization.

<img src="fig_Users.jpg" width="500px"><br>

For the *Vehicles* table, the information contained in the noise variables (N_10 to C_14) is cancelled out by discretization.

<img src="fig_Vehicles.jpg" width="500px"><br>


## FS4MT Foundation

The model training process is a fundamental component of machine learning workflows, involving key steps such as feature selection, data transformation, model training, and validation to develop accurate and robust predictive models. Multi-table data represents a significant portion of the data sets available within organizations. Analyzing these data sets, particularly through the model training process, provides essential insights for businesses, such as fraud detection, service improvement, or customer relationship management. Exploiting these multi-table data sets involves a transformation step known as "flattening" (i.e. propositionalization), which consists of converting the multiple tables into a single table. For example, in a customer database, this process might aggregate all transaction records related to each customer into one row, including features such as the total number of transactions, average transaction amount, and last transaction date. This transformation creates aggregates from the original variables and primitive construction rules to simplify the data for analysis: such as the primitive "Count" (number of transactions) or "Mean" (mean transaction amount). These primitives serve as fundamental building blocks for generating new features (aggregates) that capture essential information. Khiops (www.khiops.org) automate the propositionalization process. 

While this approach offers rich informational content, it also introduces additional challenges due to the relational nature of the data and the presence of many redundant or non-informative variables. For instance, a non-informative or noisy variable can propagate through the aggregation process, resulting in aggregates that are also non-informative. Moreover, the information contained in the one-to-many (1-N) relationship between tables, such as a customer table (A) and a transaction table (B), can provide valuable insights independently of the variables in table B. This relational information can be captured through aggregates, but if not carefully managed, it may lead to the inclusion of irrelevant or redundant features that do not improve, or even harm, classification model performance. 

FS4MT allows to evaluate the contribution of secondary variables importance measures for supervised classification with multi-table data. FS4MT will quantify the impact of noise and measure the utility of secondary variable filtering to improve classifier performance in a multi-table data context.




