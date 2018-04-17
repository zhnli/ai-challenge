import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import datasets

opt = "train" # train/predict
data_src = "sample" # sample/challenge/scoring
alg = "random_forest" # random_forest/logistic_regression
encoder = "one_hot_encoder" # one_hot_encoder/label_encoder

if data_src == "sample":
    feature_prefix = "sample_set"
elif data_src == "challenge":
    feature_prefix = "challenge_data"
else:
    feature_prefix = "scoring_set"


def preprocess():
    print("Preprocessing data sets")
    
    #iris = datasets.load_iris()
    #df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    if data_src == "sample":
        df = pd.read_csv('../data_set_v2/sampleData.tsv', sep='\t')
    elif data_src == "challenge":
        df = pd.read_csv('../data_set_v2/challengeData.tsv', sep='\t')
    else:
        df = pd.read_csv('../data_set_v2/scoring_set.tsv', sep='\t')
    
    # sklearn provides the iris species as integer values since this is required for classification
    # here we're just adding a column with the species names to the dataframe for visualisation
    #df['species'] = np.array([iris.target_names[i] for i in iris.target])
    #sns.pairplot(df, hue='species')
    
    #if data_src == "sample":
    #    ff = open("feature_names_sample.txt", "r")
    #else:
    #    ff = open("feature_names_challenge.txt", "r")
    
    ff = open("feature_names.txt", "r")
    feature_names = []
    for line in ff:
        if line[0] == '#':
            continue
        if line[0] == '@':
            continue
        feature_names.append('.'.join([feature_prefix, line.strip()]))
    
    ff = open("id_feature_names.txt", "r")
    drop_features = []
    for line in ff:
        drop_features.append('.'.join([feature_prefix, line.strip()]))
    
    fnames = [ x for x in feature_names if x not in drop_features ]
    
    print(df[fnames].shape)
    print(pd.get_dummies(df[fnames]).shape)
    print(df[fnames].shape)
    
    target_name = '.'.join([feature_prefix, 'renewed_yorn'])
    
    #else:
    #    target_name = 'challenge_data.renewed_yorn'
    
    # Drop rows without target value
    df = df.dropna(subset=[target_name])
    
    # Drop columns that all rows have the same values
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1)
    
    #df = df.reset_index()
    df = df.fillna(0)

    return (df, fnames, target_name)

def encode(encoder, df, fnames):
    df_encoded = None
    if encoder == 'one_hot_encoder':
        df_encoded = pd.get_dummies(df[fnames])
    elif encoder == 'label_encoder':
        pass
    else:
        print("Error: Encoder not recognized")
        exit(1)

    return df_encoded

#def train(df, fnames, target_name, save_model=False):
def train(X, y, save_model=False):
    print("Splitting data sets")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        #pd.get_dummies(df[fnames]),
        X,
        #df[target_name],
        y,
        #iris.target,
        test_size=0.5,
        #stratify=df[target_name],
        stratify=y,
        #stratify=iris.target,
        random_state=123456)

    print("X_train: {}".format(X_train.shape))
    print("X_test: {}".format(X_test.shape))
    
    print("Training model")
    if alg == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
    elif alg == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    else:
        print("Algorithm not recognized")
        exit(1)
    model.fit(X_train, y_train)

    if save_model:
        from sklearn.externals import joblib
        model_file = ''
        if data_src == "sample":
            model_file = 'rf_model_sample.pkl'
        else:
            model_file = 'rf_model_challenge.pkl'

        joblib.dump(model, model_file) 

    return (model, X_test, y_test)

def predict_and_assess(model, X_test, y_test):
    print("Predicting...")
    from sklearn.metrics import accuracy_score
    predicted = model.predict(X_test)
    
    if data_src != "scoring":
        accuracy = accuracy_score(y_test, predicted)
        #print(predicted)
        #print(y_test)
        #print(f'Out-of-bag score estimate: {model.oob_score_:.3}')
        print(f'Mean accuracy score: {accuracy:.3}')
    
    #from sklearn.preprocessing import OneHotEncoder
    #enc = OneHotEncoder()
    #newarray = enc.fit_transform(predicted).toarray()
    #print(newarray)
    
    #newresult = []
    #for a in predicted:
    #    if a == 0:
    #        newresult.append([0.8,0.1,0.1])
    #    elif a == 1:
    #        newresult.append([0.1,0.8,0.1])
    #    elif a == 2:
    #        newresult.append([0.1,0.1,0.8])
    #print(newresult)
    #
    #from sklearn.metrics import log_loss
    #logloss = log_loss(y_test, newresult)
    #print(f'Log Loss: {logloss:.3}')

def predict_and_save():
    print("Loading model")
    from sklearn.externals import joblib
    rf = joblib.load('rf_model.pkl')

    if data_src == "sample" or data_src == "challenge":
        print("Splitting data sets")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            pd.get_dummies(df[fnames]),
            df[target_name],
            #iris.target,
            test_size=0.9,
            stratify=df[target_name],
            #stratify=iris.target,
            random_state=123456)
        print("X_train: {}".format(X_train.shape))
        print("X_test: {}".format(X_test.shape))
    else:
        #print("Splitting data sets")
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(
        #    pd.get_dummies(df[fnames]),
        #    df[target_name],
        #    #iris.target,
        #    test_size=1.0,
        #    stratify=df[target_name],
        #    #stratify=iris.target,
        #    random_state=123456)
        #print("X_train: {}".format(X_train.shape))
        #print("X_test: {}".format(X_test.shape))
    
        print("Encoding features")
        y_test = None
        X_test = pd.get_dummies(df[fnames])
        print("X_test: {}".format(X_test.shape))

    print("Predicting...")
    from sklearn.metrics import accuracy_score
    predicted = model.predict(X_test)

    if data_src == "scoring":
        res = df[feature_prefix + 'innovation_challenge_key'] + predicted
        res.to_csv("predict_resoults.tsv", sep='\t')

def main():
    if opt == "train" and data_src == "scoring":
        print("Cannot train using scoring data set!")
        exist(1)

    df, fnames, target_name = preprocess()

    X = encode(encoder, df, fnames)
    y = df[target_name]

    if opt == "train":
        model, X_test, y_test = train(X, y)
        predict_and_assess(model, X_test, y_test)

    elif opt == "predict":
        predict_and_save()

if __name__ == "__main__":
    main()

