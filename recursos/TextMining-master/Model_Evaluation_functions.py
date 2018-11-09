

import matplotlib.pyplot as plt
import seaborn as  sns
from sklearn.metrics import roc_curve, auc, confusion_matrix , roc_auc_score,accuracy_score , recall_score, f1_score,precision_score
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import copy



def plot_score_progession(listed_tuples,
                          eval_func,
                          pc_list = np.arange(0.1,0.9,0.05),
                          color_list = plt.get_cmap("tab10").colors,
                          ax=None,
                          title=None,
                          fig_size=(6,4),
                          return_results=False,
                          plot_pc=True,

                         ):
    results = {}
    if not ax:
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=fig_size)

    for i,(name, y_real,y_pred_proba) in enumerate(listed_tuples):
        scores = [eval_func(y_real,y_pred_proba>=pc) for pc in pc_list]
        argmax = np.argmax(scores)

        ax.plot(pc_list,scores, label="%s (%.2f,%.2f)"%(name,pc_list[argmax],scores[argmax]),c=color_list[i])
        if plot_pc:
            ax.axvline(pc_list[argmax], linestyle="--", alpha=0.7, c=color_list[i])

        results[name] = {"max_score": scores[argmax], "best_pc":pc_list[argmax]}




    if title:
        ax.set_title(title)

    ax.set_xlabel("Class Probability Thresholds")
    ax.legend()

    if return_results:
        return results


def plot_roc_auc(listed_tuples,
                  color_list = plt.get_cmap("tab10").colors,
                  ax=None,
                  title=None,
                  fig_size=(6,4),
                  return_results=False,
                 ):
    results = {}
    if not ax:
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=fig_size)

    for i,(name, y_real,y_pred_proba) in enumerate(listed_tuples):
        fpr, tpr, thresholds = roc_curve(y_real, y_pred_proba)
        roc_auc = roc_auc_score(y_real,y_pred_proba)
        label = "%s AUC: %.2f"%(name,roc_auc )

        ax.plot(fpr, tpr, label=label , c= color_list[i] )
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
        ax.legend(loc="lower right")

        results[name] = {"roc_auc": roc_auc}




    if title:
        ax.set_title(title)

    ax.legend()

    if return_results:
        return results

def plot_prob_density(listed_tuples,
                      color_list = plt.get_cmap("tab10").colors,
                      ax=None,
                      title=None,
                      fig_size=(6,4),
                      pc=None,
                     ):

    if not ax:
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=fig_size)

    for i,(name, y_real,y_pred_proba) in enumerate(listed_tuples):
        for cat_name in np.unique(y_real):
            Series(y_pred_proba[np.where(y_real == cat_name)]).plot(kind= "kde", ax= ax, label="%s %s"%(name,cat_name))


        #DataFrame({"real_values":y_real,"True_prob":y_pred_proba}).groupby("real_values")["True_prob"].plot(kind= "kde")

    if pc:
        ax.axvline(pc,linestyle="--", c="k", label="punto de corte: %.2f"%(pc))

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Class Probability density function")
    ax.legend()
    ax.set_xlim(0,1)
    ax.set_xlabel("Probabilities")
def plot_confusion_matrix(y_real,y_pred,
                          normalize=False,
                          cmap= "Blues",
                          ax=None,
                          title=None,
                          fig_size=(6,4),
                          ):

    if not ax:
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=fig_size)


    cfn_matrix = confusion_matrix(y_real, y_pred)

    v_max= cfn_matrix.sum()
    fmt = "d"

    if normalize:
        cfn_matrix = cfn_matrix.astype(float) / cfn_matrix.sum(axis=1)[:, np.newaxis]
        v_max=1
        fmt = ".2f"

    sns.heatmap(cfn_matrix, annot=True, ax=ax , fmt=fmt, vmin=0, vmax=v_max,
                cmap=cmap,  linewidths=.2, linecolor="black", square=True)
    ax.set_xticklabels([False,True])
    ax.set_yticklabels([False,True])
    _=ax.set_ylabel("True Values")
    _=ax.set_xlabel("Predicted Values")


    if title:
        ax.set_title(title)
    else:
        ax.set_title("Confusion Matrix")


def fit_clf (mod_dict, X_train,y_train ,eval_set=None,early_stoping= None, extra_clf_params={}):
    features_pipe = mod_dict.get("dataset_pipeline")
    if features_pipe: #si hay pipeline , trasnformo X_train
        features_pipe.set_params(**mod_dict.get("dataset_pipeline_params"))
        X_train_prep = features_pipe.fit_transform(X_train)
    else:
        X_train_prep = X_train

    clf= copy.deepcopy(mod_dict.get("clf"))
    clf.set_params(**mod_dict.get("clf_params"))

    if eval_set: # si hay validation set, lo transformo si hay pipelie
        X_valid,y_valid = eval_set
        if features_pipe:

            X_valid_prep = features_pipe.transform(X_valid)
        else:
            X_valid_prep=X_valid

        extra_clf_params["eval_set"]=(X_valid_prep, y_valid)
        extra_clf_params["early_stopping_rounds"]=early_stoping

    clf.fit(X_train_prep, y_train,**extra_clf_params)
    return {"clf_fitted": clf, "feat_pipe_fitted": features_pipe}
