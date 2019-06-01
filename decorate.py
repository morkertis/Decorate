import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from keras import layers
from keras import models
from sklearn.utils import shuffle

#separate x, y from dataframe
def x_y(T, start=False):
    if start:
        x = T.iloc[:, 1:].values
        y = T.iloc[:, 0].values
        xdf = T.iloc[:, 1:]
    else:
        x = T.iloc[:, :-1].values
        y = T.iloc[:, -1].values
        xdf = T.iloc[:, :-1]
    return x, y, xdf

#predict probability of all classifiers
def C_predictProba_avg(C, data):
    preds = []
    for clf in C.values():
        #print(data)
        preds.append(clf.predict_proba(np.array(data)))
    avgpred = np.array(preds).mean(axis=0)
    return avgpred

#find the class from the probability
def C_classes_from_proba(avgpred, clas):
    classes = []
    for li in avgpred:
        new_class = clas[np.argmax(li)]
        classes.append(new_class)
    return classes


# return the invers class (not the probability)
def inverseClasses(Y_proba, classes):
    newY = []
    for li in Y_proba:
        new_class = classes[np.argmin(li)]
        newY.append(new_class)
    return np.array(newY)


# def X_for_clf(Xdf):
#     catcol = list(Xdf.columns[Xdf.dtypes != float])
#     Xnp = Xdf
#     return np.array(Xnp)

#return the normal distribution of array
def numeric(arr, R_size):
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    return np.random.normal(mean, std, int(R_size * len(arr))).T

#counts the number of categorial features and create probability
def nominal(arr, R_size):
    y = np.bincount(arr)
    unique_nominal = np.nonzero(y)[0]
    dist = y / y.sum()
    np.nonzero(y)
    dist = dist[np.nonzero(dist)]
    return np.random.choice(unique_nominal, int(R_size * len(arr)), p=dist).T


def generate_data_decorate(R_size, Xdf):
    newdf = pd.DataFrame()

    for i, col in enumerate(Xdf.columns):

        if Xdf[col].dtype == 'int32' or Xdf[col].dtype == 'int64':
            arr_nom = nominal(np.array(Xdf[col]), R_size)
            newdf[col] = arr_nom
        elif Xdf[col].dtype == 'float_':
            arr_num = numeric(np.array(Xdf[col]), R_size)
            newdf[col] = arr_num
    return newdf.values

#union new data and old data
def T_UNION(X, Y, R_data, newY):
    Xconcat = np.concatenate((X, R_data), axis=0)
    Yconcat = np.concatenate((Y, newY), axis=0)
    return Xconcat, Yconcat

#train classifier
def clf_train(clf, X_TunionR, Y_TunionR):
    newClf = clone(clf)
    X_TunionR, Y_TunionR = shuffle(X_TunionR, Y_TunionR, random_state=0)
    newClf.fit(X_TunionR, Y_TunionR)
    return newClf

## main function fot decorate algorithm
def decorateAlg(clf, T, C_size, I_max, R_size, GAN_OR_DECORATE=True, data=None):
    i = 1
    trials = 1
    C = {}
    X, Y, Xdf = x_y(T)
    #Xclf = X_for_clf(Xdf)
    Xclf = X
    C[0] = clone(clf)
    C[0].fit(Xclf, Y)
    classes = C[0].classes_
    error = 1 - accuracy_score(Y, C[0].predict(Xclf))
    while i < C_size and trials < I_max:
        if GAN_OR_DECORATE:
            R_data = generate_data_decorate(R_size, Xdf)
        else:
            R_data = data
        Y_proba = C_predictProba_avg(C, R_data)
        newY = inverseClasses(Y_proba, classes)
        X_TunionR, Y_TunionR = T_UNION(Xclf, Y, R_data, newY)  # union data
        C_tag = clf_train(clf, X_TunionR, Y_TunionR)
        C[i] = C_tag
        y_predict = C_classes_from_proba(C_predictProba_avg(C, Xclf), classes)
        error_tag = 1 - accuracy_score(Y, y_predict)
        if error_tag <= error:
            i += 1
            error = error_tag
        else:
            del C[i]
        trials += 1
    return C

#predict for all classifiers
def predict_C(C, X):
    if C:
        classes = C[0].classes_
    return C_classes_from_proba(C_predictProba_avg(C, X), classes)


#GAN models
def critic_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count * 4, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count * 2, activation='relu')(x)  # 2
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count * 1, activation='relu')(x)  # 1
    # x = layers.Dense(base_n_count*4, activation='relu')(x) # extra
    # x = layers.Dense(base_n_count*4, activation='relu')(x) # extra
    # x = layers.Dense(1, activation='sigmoid')(x)
    x = layers.Dense(1)(x)
    return x


def generator_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count, activation='relu')(x)
    x = layers.Dense(base_n_count * 2, activation='relu')(x)
    x = layers.Dense(base_n_count * 4, activation='relu')(x)
    x = layers.Dense(data_dim)(x)
    return x


def discriminator_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count * 4, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count * 2, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    # x = layers.Dense(1)(x)
    return x


def define_models_GAN(rand_dim, data_dim, base_n_count, type=None):
    generator_input_tensor = layers.Input(shape=(rand_dim,))
    generated_image_tensor = generator_network(generator_input_tensor, data_dim, base_n_count)

    generated_or_real_image_tensor = layers.Input(shape=(data_dim,))

    if type == 'Wasserstein':
        discriminator_output = critic_network(generated_or_real_image_tensor, data_dim, base_n_count)
    else:
        discriminator_output = discriminator_network(generated_or_real_image_tensor, data_dim, base_n_count)

    generator_model = models.Model(inputs=[generator_input_tensor], outputs=[generated_image_tensor], name='generator')
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor],
                                       outputs=[discriminator_output],
                                       name='discriminator')

    combined_output = discriminator_model(generator_model(generator_input_tensor))
    combined_model = models.Model(inputs=[generator_input_tensor], outputs=[combined_output], name='combined')

    return generator_model, discriminator_model, combined_model

#end GAN models

# GAN function for generates data
def generate_data_gan(n_samples, Xdf, path_model, rand_dim=32):
    generator_model, discriminator_model, combined_model = define_models_GAN(rand_dim, data_dim=len(Xdf.columns),
                                                                             base_n_count=128)
    generator_model.load_weights(path_model)
    z = np.random.normal(size=(n_samples, rand_dim))
    g_z = generator_model.predict(z)

    newdf = pd.DataFrame()
    for i, col in enumerate(Xdf.columns):

        if Xdf[col].dtype == 'int32' or Xdf[col].dtype == 'int64':
            newdf[col] = np.rint(np.abs(g_z[:, i])).astype(int)
        elif Xdf[col].dtype == 'float_':
            newdf[col] = np.abs(g_z[:, i])
    return newdf


def main():
    #GAN DATASET
    paths1 = ['winequality-white' , 'nursery', 'mammographic_masses', 'mushroom', 'cars_dummies']
    #DECORATE DATASETS
    paths2 = ['winequality-white' , 'nursery', 'mammographic_masses', 'mushroom', 'cars_dummies']
    paths3 = ['diabetic_data_fix', 'diseased_trees_imbalance', 'HR_comma_sep - salary', 'income_salary', 'credit_card_clients']
    path = 'data/{}.csv'
    n_samples = 10
    import time

    accuracy_all = []
    recall_all = []
    for p in range(0, 3):
        for pos in range(0, 5):
            print("p= ", p, " pos= ", pos)
            start_time = time.time()

            if p == 0: #GAN
                df = pd.read_csv(path.format(paths1[pos]))
                Xdf = df.iloc[:, :-1]
                path_model = 'cache/' + paths1[pos] + '/GAN_generator_model_weights_step_500.h5'
                data2 = generate_data_gan(n_samples, Xdf.copy(), path_model, rand_dim=32)
            else: #DECORATE
                data2 = None
                if p == 1:
                    df = pd.read_csv(path.format(paths2[pos]))
                else:
                    df = pd.read_csv(path.format(paths3[pos]))
                Xdf = df.iloc[:, :-1]
            C_size = 15
            I_max = 50
            R_size = 1
            accuracy = []
            recall = []
            kfold = KFold(n_splits=10, shuffle=True, random_state=0)
            X, y, Xdf = x_y(df.copy())
            clf = DecisionTreeClassifier(max_depth=2)
            for train_index, test_index in kfold.split(X, y):
                dftraincv, dftestcv = df.iloc[train_index], df.iloc[test_index]
                if p == 0: #GAN
                    clfs = decorateAlg(clf, dftraincv, C_size=C_size, I_max=I_max, R_size=R_size, GAN_OR_DECORATE=False,
                                       data=data2)
                else: #DECORATE
                    clfs = decorateAlg(clf, dftraincv, C_size=C_size, I_max=I_max, R_size=R_size,GAN_OR_DECORATE=True)
                pred = predict_C(clfs, dftestcv.iloc[:, :-1].values)
                accuracy.append(metrics.accuracy_score(dftestcv.iloc[:, -1].values, pred))
                recall.append(metrics.recall_score(dftestcv.iloc[:, -1].values, pred, average='macro'))
            run_time = time.time() - start_time
            print("(p,pos)=", p, ",", pos, " --- " + str(run_time) + "  seconds ---")
            accuracy_all.append(np.mean(accuracy))
            recall_all.append(np.mean(recall))
    print("accuracy:")
    print(accuracy_all)
    print("recall:")
    print(recall_all)

if __name__ == '__main__':
    main()

