from sklearn import tree
from sklearn.linear_model import LogisticRegression
import pandas as pd

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

survival_df = train_df['Survived']
new_train_df = train_df.loc[:,['Pclass','Sex','Parch']]
new_test_df = test_df.loc[:,['Pclass','Sex','Parch']]

new_train_df['Sex'].replace('male', 1, inplace=True)
new_train_df['Sex'].replace('female', 0, inplace=True)

new_test_df['Sex'].replace('male', 1, inplace=True)
new_test_df['Sex'].replace('female', 0, inplace=True)

# new_train_df.fillna(0, inplace=True)
# new_test_df.fillna(0, inplace=True)




clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(new_train_df, survival_df)

clf_lr = LogisticRegression()
clf_lr.fit(new_train_df, survival_df)


predictions_dt = clf_dt.predict(new_test_df)
predictions_lr = clf_lr.predict(new_test_df)

result_dt = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':predictions_dt}).set_index('PassengerId')
result_lr = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':predictions_lr}).set_index('PassengerId')

result_dt.to_csv('decision_tree_2.csv')
result_lr.to_csv('log_regression_2.csv')
