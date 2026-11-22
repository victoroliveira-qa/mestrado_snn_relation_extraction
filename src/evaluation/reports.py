import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score


def save_metrics(ytrue, ypred, label_names, output, prefix):
    label_ids=np.arange(len(label_names))
    payload={"accuracy":float(accuracy_score(ytrue,ypred)),"macro_f1":float(f1_score(ytrue,ypred,labels=label_ids,average="macro",zero_division=0)),"classification_report":classification_report(ytrue,ypred,labels=label_ids,target_names=list(label_names),output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(ytrue,ypred,labels=label_ids).tolist()}
    (output/f"{prefix}_metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload


def save_training_curves(history, output, logs_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df=pd.DataFrame(history.history);df.to_csv((logs_dir or output)/"training_history.csv",index=False)
    for metric in ("loss","accuracy","auc"):
        if metric not in df.columns and f"val_{metric}" not in df.columns:
            continue  # Avoid blank plots and Matplotlib legend warnings.
        fig, ax = plt.subplots(figsize=(7, 4))
        if metric in df.columns:
            ax.plot(np.arange(1, len(df) + 1), df[metric], label="train")
        if f"val_{metric}" in df.columns:
            ax.plot(np.arange(1, len(df) + 1), df[f"val_{metric}"], label="validation")
        ax.set(xlabel="Epoch", ylabel=metric, title=f"Training {metric}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)


def save_confusion(ytrue,ypred,names,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm=confusion_matrix(ytrue,ypred,labels=np.arange(len(names)))
    fig,ax=plt.subplots(figsize=(max(7,len(names)),max(6,len(names)*.8)))
    im=ax.imshow(cm,cmap="Blues");fig.colorbar(im,ax=ax)
    ax.set_xticks(range(len(names)),names,rotation=45,ha="right")
    ax.set_yticks(range(len(names)),names)
    ax.set_xlabel("Predicted");ax.set_ylabel("Actual")
    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha="center",va="center",fontsize=8)
    fig.tight_layout();fig.savefig(output/"confusion_matrix.png",dpi=160);plt.close(fig)
