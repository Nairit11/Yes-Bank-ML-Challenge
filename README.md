# Yes-Bank-ML-Challenge

Steps Followed:

1) Drop non-important columns(serial_no, date, month_of_year) based on intuition.

2) Map string values of binary columns to 0 for "no" and 1 for "yes"

3) One-hot encode categorical features(poutcome_of_campaign, phone_type,job_description, marital_status,education_details).

4) Separate target variable(outcome).

5) Pre-process the test dataset, using above steps.

6) Apply Classification models
	- Logistic Classification
	- Random Forest Classifier
	- Linear SVM

7) Perform feature selection based on feature importances as returned by above models after training with whole dataset. Then training with only important features
