import pandas as pd
import time
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_data(train_file,test_file,sample_files):
    print("Loading data")
    train_df=pd.read_csv(train_file)
    test_df=pd.read_csv(test_file,index_col=0)
    sample_df=pd.read_csv(sample_files)
    return train_df,test_df,sample_df

#load in the data
train_df,test_df,sample_df=load_data("train.csv","test.csv","sample_submission.csv")
x_train=train_df[['transliteration']]
y_train=train_df[['translation']]
x_test=test_df[['transliteration']]

encoder=OneHotEncoder(handle_unknown='ignore')

x_train_encoded=encoder.fit_transform(x_train)
label_encoder=LabelEncoder()
y_train_encoded=label_encoder.fit_transform(train_df['translation'])

#generate the submission file
dummyModel=DummyClassifier(strategy='most_frequent')
startTime=time.time()
cv_scores=cross_val_score(
    dummyModel,
    x_train_encoded,
    y_train_encoded,
    cv=2,
    scoring='accuracy',
    n_jobs=-1
)

elapsed_time=time.time()-startTime
print("Mean cv-score",-cv_scores.mean())
print("Time taken",elapsed_time)

dummyModel.fit(x_train_encoded,y_train_encoded)
x_test_encoded=encoder.transform(test_df[['transliteration']])
test_prediction=dummyModel.predict(x_test)
test_prediction=label_encoder.inverse_transform(test_prediction)
submission=pd.DataFrame({
    'id':test_df.index,
    'translation':test_prediction
})
print(submission)
filename = "dummy_submission.csv"
submission.to_csv(filename, index=False)