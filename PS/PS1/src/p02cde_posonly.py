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
    train_x, train_y = util.load_dataset(train_path,label_col='y',add_intercept=True)
    train_x, train_t = util.load_dataset(train_path,label_col='t',add_intercept=True)
    valid_x, valid_y = util.load_dataset(valid_path,label_col='y',add_intercept=True)
    _, valid_t = util.load_dataset(valid_path,label_col='t',add_intercept=True)
    test_x, test_y = util.load_dataset(test_path,add_intercept=True)
    _, test_t = util.load_dataset(test_path,label_col='t',add_intercept=True)
    #################################################################################################
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    print("Part (c) - Train and test on true labels")
    ## Step0 : Initialize model and check the size of datas
    clf = LogisticRegression()
        # print("train_x.shape : ",train_x.shape) #(1250,3)
        # print("train_t.shape : ",train_t.shape) #(1250,)
    
    ## Step1 : Train
    clf.fit(train_x, train_t)
    print("Theta is : ", clf.theta)
    print("The accuracy on training set is : ", np.mean(train_t == (clf.predict(train_x) >= 0.5) ))
    util.plot(train_x,train_t,clf.theta,'output/p02c_true_label(train).png')
    ## Step1-2 : check accuracy for validation set
    print("The accuracy on validation set is : ", np.mean(valid_t == (clf.predict(valid_x) >= 0.5)))
    util.plot(valid_x,valid_t,clf.theta,'output/p02c_true_label(valid).png')
    ## Step2 : Test (predict)
    result_pred_c = clf.predict(test_x)
    print("The accuracy on test set is : ", np.mean(test_t == (result_pred_c >= 0.5)))
    
    ## Step3 : save outputs
    np.savetxt(pred_path_c,result_pred_c >= 0.5,'%d')
    
    ## Step4 : plot decision boundary
    util.plot(test_x,test_t,clf.theta,'output/p02c_true_label(test).png')
    #################################################################################################
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    print("Part (d) - Train on y-labels and test on true labels")
    
    ## Step0 : initialize model and check the size of datas
    clf = LogisticRegression()
    ## Step1 : initialize model and check accuracy
    
    clf.fit(train_x,train_y)
    print("Theta is : ",clf.theta)
    print("The accuracy on training set is : ", np.mean(train_y == (clf.predict(train_x) >= 0.5) ))
    util.plot(train_x,train_t,clf.theta,'output/p02d_correction1(train).png')
    
    ## Step1-2 : Check accruacy for validation set
    print("The accuracy on validation set is : ", np.mean(valid_y == (clf.predict(valid_x) >= 0.5 )))
    util.plot(valid_x,valid_t,clf.theta,'output/p02d_correction1(valid).png')
    
    ## Step2 : Test (predict)
    result_pred_d = clf.predict(test_x)
    
    ## Step3 : save outputs
    np.savetxt(pred_path_d,result_pred_d >= 0.5,'%d')
    
    ## Step4 : plot decision boundary
    print("The accuracy on test set is : ", np.mean(test_t == (result_pred_d >= 0.5)))
    util.plot(test_x,test_t,clf.theta,'output/p02d_correction1(test).png')
    #################################################################################################
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    print("Part (e) - Apply correction factor using validation set and test on true labels")
    selected_valid_x = valid_x[valid_y == 1]
    
    ## clf is trained on y-label
    h_x = clf.predict(selected_valid_x)
    alpha = np.average(h_x)
    print("alpha : ", alpha)
    print("The accuracy on test set after correction is : ", np.mean(test_t == (result_pred_d/alpha > 0.5)))
    np.savetxt(pred_path_e, result_pred_d/alpha > 0.5, '%d')
    
    # util.plot(test_x,test_t,clf.theta,'output/p02e_correction_alpha(test).png',correction=( 1- (1/clf.theta[0]) * np.log(2/alpha - 1) ))
    # A : add correction
    util.plot(test_x,test_t, clf.theta,'output/p02e_correction_alpha(test).png',correction= (1 + np.log(2/alpha - 1)/clf.theta[0]))
    # B : adjust theta itself
        # theta_adjusted = clf.theta + [np.log(2 / alpha - 1) , 0 , 0]
        # util.plot(test_x,test_t, theta_adjusted,'output/p02e_correction_alpha_adjusted(test).png')
    #################################################################################################
    # *** END CODER HERE
