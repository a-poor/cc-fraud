import time, pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split

def train_test_val_split(X,y,test_pct=0.3,val_pct=0.33):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_pct)
    X_val, X_test, y_val, y_test = train_test_split(X_train,y_train,test_size=val_pct)
    return X_train, X_val, X_test, y_train, y_val, y_test

def show_confusion(y_true,y_pred,log_transform=False):
    c = confusion_matrix(y_true,y_pred)
    if log_transform: 
        c = np.log(c + (1/len(y_true)))
    mat = pd.DataFrame(
        c,
        columns=["True -","True +"],
        index=["Pred -","Pred +"]
    )
    sns.heatmap(mat)
    
    
def strftime():
    return time.strftime("%Y.%m.%d.%H.%M.%S")

def makefn(name,score):
    return f"models/{name}_{strftime()}_{score:.6f}.pkl"

def save_model(model,name,score,cutoff=0):
    if score < cutoff: return False
    with open(makefn(name,score),"wb") as f:
        pickle.dump(model,f)
    return True
