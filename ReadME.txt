You are provided with an anonymized dataset containing a large number of numeric variables. The "TARGET" column is the variable to predict. It equals one for unsatisfied customers and 0 for satisfied customers.

The task is to predict the probability that each customer in the test set is an unsatisfied customer.

File descriptions

train.csv - the training set including the target
test.csv - the test set without the target
sample_submission.csv - a sample submission file in the correct format

To run this code, follow below steps:

1) clone or download this code.
2) Import this project into Scala IDE.
3) Copy data present in Data folder to some path in hadoop box and mention the same in code.
4) Right click on project and "Run As" --> "Maven Install"
5) After jar is created, using WinSCP copy the jar file to cloudera or other Hadoop distribution box.
6) Navigate to jar path then, use below command in shell.


spark-submit --master yarn-client --driver-memory 512m --executor-memory 512m --class mllib.logisticRegression.logisticRegression Mllib-LogisticRegression-0.0.1-SNAPSHOT-jar-with-dependencies.jar