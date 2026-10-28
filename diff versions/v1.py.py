#Predictive Maintenance from Sensor Data

#Phase A dev 
print("Phase A Dev Started")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,roc_auc_score,average_precision_score,confusion_matrix, ConfusionMatrixDisplay,precision_recall_curve,roc_curve,mean_squared_error,mean_absolute_error
from xgboost import XGBClassifier,XGBRFRegressor
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as mp
import seaborn as sns
import shap
from ucimlrepo import fetch_ucirepo
print("-> Phase 0 Done")

#dataset 
ds=fetch_ucirepo(id =601)
x=ds.data.features
y=ds.data.targets
print("Dataset:\n",x.head())
print("X shape:",x.shape)
print("y shape:",y.shape)
print("Feature Columns:",x.columns.tolist())
print("Target Columns:",y.columns.tolist())
print("Type Values:",x["Type"].unique())
print("Target Distribution:",y["Machine failure"].value_counts())
print("Failure rate:",y["Machine failure"].mean())
print("-> Phase 1 Done")

#feature manupilation
x_en=pd.get_dummies(x,columns=["Type"],drop_first=True)
x_en["power"]=x["Torque"]*x["Rotational speed"]*(2*np.pi/60)
x_en["temp_diff"]=x["Process temperature"]-x["Air temperature"]
x_en["wear_torque"]=x["Tool wear"]*x["Torque"]
print("Final feature coloumns:",x_en.columns.to_list())
print("Null Check:\n",x_en.isnull().sum())
print("Derived Feature Stats:\n",x_en[["power","temp_diff","wear_torque"]].describe())
print("-> Phase 2 Done")

#Independent Feature
x=x_en
#Dependent Feature
y_target=y["Machine failure"]

#Train_Test_Split
x_train,x_test,y_train,y_test=train_test_split(
    x,y_target,test_size=0.25,stratify=y_target,random_state=42
)
print("Train Data:",x_train.shape,"\nTrain failure rate:",y_train.mean())
print("Test Data",x_test.shape,"\nTest failure rate:",y_test.mean())
print("-> Phase 3 Done")

#Standardlizin' data
ss=StandardScaler()
x_train_sd=ss.fit_transform(x_train)
x_test_sd=ss.transform(x_test)

#Model Callin'
lgr=LogisticRegression(class_weight="balanced",max_iter=1000,random_state=42)
lgr.fit(x_train_sd,y_train)

#Prediction
pred=lgr.predict(x_test_sd)
prob=lgr.predict_proba(x_test_sd)[:,1]
print("Classification Report:\n",classification_report(y_test,pred,target_names=["Normal","Failure"]))
print("Accuracy:",roc_auc_score(y_test,prob)*100,"\nPrecision:",average_precision_score(y_test,prob)*100)
print("-> Phase 4 Done")

#Boosting
# Calutatin' imbalance wt 
spos_wt=((y_train==0).sum()/(y_train==1).sum())
print("Scaler (+ve) Wt:",spos_wt)
#XG-Boost trainnin'
xgb=XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=spos_wt,
    eval_metric="aucpr",
    random_state=42
)
xgb.fit(x_train,y_train)
#Predition Nd Evalution
xgb_ypred=xgb.predict(x_test)
xgb_yprob=xgb.predict_proba(x_test)[:,1]
print("Classification Report:\n",classification_report(y_test,xgb_ypred,target_names=["Normal","Failure"]))
print("Accuracy:",roc_auc_score(y_test,xgb_yprob)*100,"\nPrecision:",average_precision_score(y_test,xgb_yprob)*100)
print("-> Phase 5 Done")

#Handling Imbalance
#-> No imbalance
rf=RandomForestClassifier(n_estimators=300,random_state=42)
rf.fit(x_train,y_train)
plain_prob=rf.predict_proba(x_test)[:,1]
plain_pred=rf.predict(x_test)
print("0 Handling")
print("Classification :\n",classification_report(y_test,plain_pred,target_names=["Normal","Failure"]))
print("Acurracy:",average_precision_score(y_test,plain_prob)*100)
#->Class wt
rfw=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42)
rfw.fit(x_train,y_train)
probw=rfw.predict_proba(x_test)[:,1]
predw=rfw.predict(x_test)
print("Class wt: balanced")
print("Classification:\n",classification_report(y_test,predw,target_names=["Normal","Failure"]))
print("Acuraacy :",average_precision_score(y_test,probw)*100)
#->Smote Resampling
smote=SMOTE(random_state=42)
smx_train,smy_train=smote.fit_resample(x_train,y_train)
print('Before smote:',y_train.value_counts().to_dict())
print("After Smote:",smy_train.value_counts().to_dict())
rf_smote=RandomForestClassifier(n_estimators=300,random_state=42)
rf_smote.fit(smx_train,smy_train)
sm_prob=rf_smote.predict_proba(x_test)[:,1]
sm_pred=rf_smote.predict(x_test)
print("Smote")
print("Classification :\n",classification_report(y_test,sm_pred,target_names=["Normal","failure"]))
print("Accuracy:",average_precision_score(y_test,sm_prob)*100)
print("-> Phase 6 Done")

#Evaluation
#-> Confusion Matirx
cm=confusion_matrix(y_test,xgb_ypred)
disp=ConfusionMatrixDisplay(cm,display_labels=["Normal","Failure"])
disp.plot(cmap="Blues")
mp.title("Consfusion Matrix- XGBoost (Threshold=.5)")
mp.show()
#-> Precision-Recall Curve
p,rcl,ths=precision_recall_curve(y_test,xgb_yprob)
mp.plot(rcl,p,color="darkorange")
mp.xlabel("Recall")
mp.ylabel("Precision")
mp.title("Precision-Recall Curve - XGBoost")
mp.show()
#-> Cost-based Threshold Sweep
cfn=10
cfp=1
bth,bc=0.5,np.inf
for t in np.arange(0.05,0.95,0.01):
    pat=(xgb_yprob>=t).astype(int)
    fn = ((pat == 0)&(y_test==1)).sum()
    fp=((pat==1)&(y_test==0)).sum()
    c=fn*cfn+fp*cfp
    if c <bc:
        bc,bth=c,t
dfn=((xgb_ypred==0)&(y_test==1)).sum()
dfp=((xgb_ypred==1)&(y_test==0)).sum()
dc=dfn*cfn+dfp*cfp
print("Cost-optimal Threshold:",bth,"\n Total cost:",bc)
print("Deafult .5 Threshold cost:",dc)
print("-> Phase 7 Done")

#Interpretability
#explainer
ex=shap.TreeExplainer(xgb)
sv=ex.shap_values(x_test)
#Global Feature
shap.summary_plot(sv,x_test,plot_type="bar",show=False)
mp.title("Global Feature Importance - mean |Shap value|")
mp.tight_layout()
mp.show()
#Full Shap Summary
shap.summary_plot(sv,x_test,show=False)
mp.title("Shap Summary")
mp.tight_layout()
mp.show()
print("-> Phase 8 Done")
print("Phase A Done ")

#Phase B
print("Phase B Dev Started")

