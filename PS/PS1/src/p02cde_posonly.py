import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    ## Step0 : Initialize model and check the size of datas
    clf = LogisticRegression()
    train_x, train_t = util.load_dataset(train_path,label_col='t',add_intercept=True)
        # print("train_x.shape : ",train_x.shape) #(1250,3)
        # print("train_t.shape : ",train_t.shape) #(1250,)
    
    ## Step1 : Train
    clf.fit(train_x, train_t)
    
    ## Step2 : Test (predict)
    test_x, _ = util.load_dataset(test_path,add_intercept=True)
    result_pred = clf.predict(test_x)
    
    ## Step3 : save outputs
    np.savetxt(pred_path_c,result_pred,'%d')
    
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    print("Start training and prediction - Train on y-labels")
    
    ## Step0 : initialize model and check the size of datas
    clf = LogisticRegression()
    train_x, train_y = util.load_dataset(train_path,label_col='y',add_intercept=True)
    # print("train_x_pos.shape : ", train_x_pos.shape)
    # print("train_y_pos.shape : ", train_y_pos.shape)
    ## Step1 : Train
    
    # pick out train examples where y-labels are 1
    # selected_train_x_pos = train_x[train_y == 1]
    # selected_train_y_pos = train_y[train_y == 1]
    
    # print("selected_train_x_pos.shape : ", selected_train_x_pos.shape)
    # print("selected_train_y_pos.shape : ", selected_train_y_pos.shape)
    
    clf.fit(train_x,train_y)
    
    
    ## Step2 : Test (predict)
    result_pred = clf.predict(test_x)
    
    ## Step3 : save outputs
    np.savetxt(pred_path_d,result_pred,'%d')
    
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
