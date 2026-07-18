from sklearn.model_selection import train_test_split

#Define target column
target = "loan_status"

#Create features and target
x = df.drop(columns=[target])
