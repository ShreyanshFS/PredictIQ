#Predictive Maintenance from Sensor Data

#Phase A dev 
print(" Model Started\n \nPhase A Started")
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report,roc_auc_score,average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay,precision_recall_curve,
                             roc_curve,mean_squared_error,mean_absolute_error)
from xgboost import XGBClassifier,XGBRegressor
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as mp
import seaborn as sns
import shap
#--Config--
Seed=42
dp=os.path.join(os.path.dirname(os.path.abspath(__file__)),"Data","factory_sim.csv")
Step_sec=10
N_hrs=1
wndw=30
cost_fn=10
cost_fp=1
Show_plot=True
def show():
    mp.show()if Show_plot else mp.close()
print("Data file Found:",os.path.exists(dp))
print("Horizon :",N_hrs,"\nH | Window:",wndw,"\nReadings | cost FN:FP =",F"{cost_fn}:{cost_fp}")
print("-> Phase 0 Done")
print("")

#Phase 1: data + labels
rw=pd.read_csv(dp).sort_values(["machineID","t_sec"]).reset_index(drop=True)
rw=rw.drop(columns=["_health_hidden"])
print("Shape :",rw.shape,"\nColumns:",rw.columns.tolist(),
      "\nMachines:",rw.machineID.nunique())
print("Time Steps(S):",rw.groupby("machineID")["t_sec"].diff().dropna().unique().tolist())
print("Missing Values:",int(rw.isna().sum().sum()))
#time to failure 
fail_t=rw["t_sec"].where(rw["fail_event"])
rw["next_fail_t"]=fail_t.groupby(rw["machineID"]).bfill()
rw["ttf_sec"]=rw["next_fail_t"]-rw["t_sec"]
n0=len(rw)
rw=rw.dropna(subset=["ttf_sec"]).reset_index(drop=True)
print("Dropped Censored rows:",n0-len(rw),"| kept:",len(rw))
rw["fail_within_N"]=(rw["ttf_sec"]<=N_hrs*3600).astype(int)
print("[+VE] Rate:",rw["fail_within_N"].mean())
print("time to failure(h):\n",(rw["ttf_sec"]/3600).describe().round(2))
print("-> Phase 1 Done")
print("")

#Phase 2: Feature Engineering + Targets
dc=["machineID","t_sec","fail_event","next_fail_t","ttf_sec","fail_within_N"]
x=rw.drop(columns=dc)
x["power_per_rpm"]=x["power"]/x["rotation"]
x["temp_x_vib"]=x["temp"]*x["vib"]
x["maint_load"]=x["hours_since_maint"]*x["power"]
y_target=rw["fail_within_N"]
y_ttf=rw["ttf_sec"]
grp=rw["machineID"]
print("Final feature coloumns:",x.columns.tolist(),"\nNull check:",int(x.isnull().sum().sum()))
print("Derived feature stats:\n",x[["power_per_rpm","temp_x_vib","maint_load"]].describe().round(3))
print("Shapes:",x.shape,y_target.shape,y_ttf.shape,grp.shape)
print("-> Phase 2 Done")
print("")

#Phase 3: Split by MACHINE
gss=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=Seed)
tr_idx,te_idx=next(gss.split(x,y_target,groups=grp))
x_train,x_test=x.iloc[tr_idx],x.iloc[te_idx]
y_train,y_test=y_target.iloc[tr_idx],y_target.iloc[te_idx]
ttf_train,ttf_test=y_ttf.iloc[tr_idx],y_ttf.iloc[te_idx]
grp_train,grp_test=grp.iloc[tr_idx],grp.iloc[te_idx]
print("Train:",x_train.shape,"\nMachines :",sorted(grp_train.unique().tolist()))
print("Test :",x_test.shape,"\nMachine :",sorted(grp_test.unique().tolist()))
print("Machine Overlap:",set(grp_train)&set(grp_test))
print("Train [+ve] rate:",round(y_train.mean(),4),"\nTest [+ve] rate:",round(y_test.mean(),2))
print("Train TTF mean:",round(ttf_train.mean()/3600,2),"\nTest ttf  mean:",round(ttf_test.mean()/3600,2))
print("-> Phase 3 Done")
print("")

#Phase 4: Baseline Logistic Regression
#->Standardlizin' data
ss=StandardScaler()
x_train_sd=ss.fit_transform(x_train)
x_test_sd=ss.transform(x_test)
#-> Model Callin'
lgr=LogisticRegression(class_weight="balanced",max_iter=1000,random_state=Seed)
lgr.fit(x_train_sd,y_train)
#->Prediction
pred=lgr.predict(x_test_sd)
prob=lgr.predict_proba(x_test_sd)[:,1]
print("CLassification Report:\n",classification_report(y_test,pred,target_names=["Normal","Failure"]))
print("ROC-Accuracy:",round(roc_auc_score(y_test,prob)*100,1),"\nPr-Accuracy:",round(average_precision_score(y_test,prob)*100,1),
      "\nChance-lvl Pr-Auc:",round(y_test.mean()*100,1))
print("-> Phase 4 Done")
print("")

#Phase 5: Boosting
# Calutatin' imbalance wt 
spos_wt=((y_train==0).sum()/(y_train==1).sum())
print("Scaler (+ve) Wt:",round(spos_wt,2))
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
print("Accuracy:",round(roc_auc_score(y_test,xgb_yprob)*100,1),"\nPrecision:",round(average_precision_score(y_test,xgb_yprob)*100,1))
print("Log Reg:",round(average_precision_score(y_test,prob)*100,1),"\n Chance:",round(y_test.mean()*100,1))
print("-> Phase 5 Done")
print("")

#Phase 6: Handling Imbalance
rf_kw=dict(n_estimators=100,random_state=Seed,n_jobs=-1)
#-> No imbalance
rf=RandomForestClassifier(**rf_kw).fit(x_train,y_train)
plain_prob=rf.predict_proba(x_test)[:,1]
plain_pred=rf.predict(x_test)
print("0 Handling")
print("Classification :\n",classification_report(y_test,plain_pred,target_names=["Normal","Failure"]))
print("Acurracy:",average_precision_score(y_test,plain_prob)*100)
#->Class wt
rfw=RandomForestClassifier(class_weight="balanced",**rf_kw).fit(x_train,y_train)
probw=rfw.predict_proba(x_test)[:,1]
predw=rfw.predict(x_test)
print("Class wt: balanced")
print("Classification:\n",classification_report(y_test,predw,target_names=["Normal","Failure"]))
print("Acuraacy :",average_precision_score(y_test,probw)*100)
#->Smote Resampling
smote=SMOTE(random_state=Seed)
smx_train,smy_train=smote.fit_resample(x_train,y_train)
print('Before smote:',y_train.value_counts().to_dict(),"\nAfter Smote:",smy_train.value_counts().to_dict())
rf_smote=RandomForestClassifier(**rf_kw).fit(smx_train,smy_train)
sm_prob=rf_smote.predict_proba(x_test)[:,1]
sm_pred=rf_smote.predict(x_test)
print("Smote")
print("Classification :\n",classification_report(y_test,sm_pred,target_names=["Normal","failure"]))
print("Accuracy:",average_precision_score(y_test,sm_prob)*100)
print("-> Phase 6 Done")
print("")

#Phase 7: Evaluation
#-> Confusion Matirx
cm=confusion_matrix(y_test,xgb_ypred)
disp=ConfusionMatrixDisplay(cm,display_labels=["Normal","Failure"])
disp.plot(cmap="Blues")
mp.title("Consfusion Matrix- XGBoost (Threshold=.5)")
mp.show()
#-> Precision-Recall Curve
p,rcl,ths=precision_recall_curve(y_test,xgb_yprob)
mp.plot(rcl,p,color="darkorange")
mp.xlabel("Recall"); mp.ylabel("Precision")
mp.title("Precision-Recall Curve - XGBoost")
mp.show()
#-> Cost-based Threshold Sweep
#cfn=10;cfp=1
bth,bc=0.5,np.inf
for t in np.arange(0.05,0.95,0.01):
    pat=(xgb_yprob>=t).astype(int)
    fn = ((pat == 0)&(y_test==1)).sum()
    fp=((pat==1)&(y_test==0)).sum()
    c=fn*cost_fn+fp*cost_fp
    if c <bc:
        bc,bth=c,t
dfn=((xgb_ypred==0)&(y_test==1)).sum()
dfp=((xgb_ypred==1)&(y_test==0)).sum()
dc=dfn*cost_fn+dfp*cost_fp
print("Cost-optimal Threshold:",bth,"\n Total cost:",bc,"\nDeafult .5 Threshold cost:",dc)
print("-> Phase 7 Done")
print("")

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

print("")
print("Phase A Done ")

print("")
print ("------><-------")

#Phase B Dev
print("")
print("Phase B Started")
print("")

#Phase 9:Per-Machine rolllin' feature
rc=["temp","vib","pressure","rotation","power"]
for c in rc:
    g=rw.groupby("machineID")[c]
    x[f"{c}_rmean"]=g.transform(lambda s:s.rolling(wndw,min_periods=1).mean())
    x[f"{c}_rstd"]=g.transform(lambda s:s.rolling(wndw,min_periods=1).std()).fillna(0)
    x[f"{c}_trend"]=g.transform(lambda s:s.diff(wndw)).fillna(0)
x_train=x.iloc[tr_idx]
x_test=x.iloc[te_idx]
print("New Feature counts:",x.shape[1],"\nNull Check :",int(x.isnull().sum().sum()),
      "\nTrain/Test Shapes :",x_train.shape,x_test.shape)
print("-> Phase 9 Done")
print("")

##Phase 10: Ridge baseline (regression)
r=Ridge(alpha=1.0,random_state=Seed)
r.fit(x_train,ttf_train)
r_pred=r.predict(x_test)
r_pred=np.clip(r_pred,0,None)
rmse=mean_squared_error(ttf_test,r_pred)**0.5
mae=mean_absolute_error(ttf_test,r_pred)
print("Ridge RMSE:",round(rmse/3600,2),"\nRidge MAE:",round(mae/3600,2))
print("TTF Test Mean:",round(ttf_test.mean()/3600,2),"\n STD:",round(ttf_test.std()/3600,2))
print("-> Phase 10 Done")
print("")

#Phase 11: XGBoost — RUL regressor + "fails within N" classifier
xgb_reg=XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,random_state=Seed)
xgb_reg.fit(x_train,ttf_train)
reg_pred=xgb_reg.predict(x_test)
reg_pred=np.clip(reg_pred,0,None)
rmse_b=mean_squared_error(ttf_test,reg_pred)**0.5
mae_b=mean_absolute_error(ttf_test,reg_pred)
print("XGB Regressor Rmse:",round(rmse_b/3600,2),"\nMAE :",round(mae_b/3600,2))
print("Ridge Rmse:",round(rmse/3600,2),"MAE :",round(mae/3600,2))
xgb_b=XGBClassifier(
    n_estimators=300,max_depth=4,learning_rate=0.05,
    scale_pos_weight=spos_wt,eval_metric="aucpr",random_state=Seed   # eval_metric, not eval_metrics
)
xgb_b.fit(x_train,y_train)
xgb_b_prob=xgb_b.predict_proba(x_test)[:,1]
print("XGB Classifier (Phase B, with rollin' feature) PR-AUC:",round(average_precision_score(y_test,xgb_b_prob)*100,1))
print("XGB Classifier (Phase A, Snapshot only) Pr-Auc:",round(average_precision_score(y_test,xgb_yprob)*100,1))
print("-> Phase 11 Done")
print("")

#Phase 12:Test Evalution 
print("="*50)
print("Phase B Test Results (Holdout Machine:",sorted(grp_test.unique().tolist()),")")
print("="*50)
print(f"{'Model':<20}{'RMsE':>10}{'MAE':>10}",f"\n{'Ridge (baselibne)':<20}{rmse/3600:>10.2f}{mae/3600:>10.2f}")      
print(f"{'XGBOOST':<20}{rmse_b/3600:>10.2f}{mae_b/3600:>10.2f}",f"\n\nImprovement over Ridge:{(1-rmse_b/rmse)*100:.1f}% lower RMSE")
print("-> Phase 12 Done")
print("")

#Phase 13: Uncertainty range (quantile regression) + alert
q_lo=XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,objective="reg:quantileerror",quantile_alpha=0.1,random_state=Seed)
q_hi=XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,objective="reg:quantileerror",quantile_alpha=0.9,random_state=Seed)
q_lo.fit(x_train,ttf_train);q_hi.fit(x_train,ttf_train)
lo_pred=np.clip(q_lo.predict(x_test),0,None);hi_pred=np.clip(q_hi.predict(x_test),0,None)
coverage= ((ttf_test>=lo_pred)&(ttf_test<=hi_pred)).mean()
print("10-90% internal coverag: ",round(coverage*100,1),"% (target ~80%)")
#alert
alert_h=N_hrs*3600;alerted=lo_pred<=alert_h;actual_soon=ttf_test<=alert_h
caught=(alerted&actual_soon).sum();total_soon=actual_soon.sum();False_alarms=(alerted&~actual_soon).sum()
print(f"Alerts firin' when ttf=<{N_hrs}: caught {caught}/{total_soon} | false alarms: {False_alarms}")
print("-> Phase 13 Done")
print("")

#Phase 14: SHAP on the RUL regressor
ex_b=shap.TreeExplainer(xgb_reg);sv_b=ex_b.shap_values(x_test)
shap.summary_plot(sv_b,x_test,plot_type="bar",show=False)
mp.title("Global Feature Importance - RUL Regressor");mp.tight_layout
mp.show()
shap.summary_plot(sv_b,x_test,show=False)
mp.title("Shap Summary - RUL Regressor");mp.tight_layout()
mp.show()
print("-> Phase 14 Done")
print("")

#Phase 15: predict_rul() wrapper + hrs:min:sec output
def sec2hrs(s):
    s=max(int(s),0)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%6:02d}"
def predict(row_df):
    """row df:single row df with same coloums as x (features only)"""
    ttf_pred=np.clip(xgb_reg.predict(row_df),0,None)[0]
    lo=np.clip(q_lo.predict(row_df),0,None)[0]
    hi=np.clip(q_hi.predict(row_df),0,None)[0]
    fail_prob=float(xgb_b.predict_proba(row_df)[:,1][0])
    
    fail_soon="Yes" if fail_prob>=bth else "No"
    return{
        "fails_within_N": fail_soon,
        "Probability": round(fail_prob*100,1),
        "time_left": sec2hrs(ttf_pred),
        "Range": f"{sec2hrs(lo)}-{sec2hrs(hi)}"
    }
#demo on 3 randam test rows
sample=x_test.sample(3,random_state=Seed)
for idx in sample.index:
    r=predict(x_test.loc[[idx]])
    print(f"Fails within {N_hrs}h? {r['fails_within_N']} ({r['Probability']}%)."
          f"Time Left ~{r['time_left']},Range {r['Range']}.")

print("\n--- TAKEWAYS ---")
print(f"XGBoost regressor beats Ridge by 67.5% lower RMSE (0.32h vs 1.00h),")
print(f"showing rolling trend/std features capture non-linear degradation Ridge cannot.")
print(f"Classifier PR-AUC near ceiling (99.9) even snapshot-only (99.3) - simulated")
print(f"degradation is strong and smooth, so real-world data would likely be harder.")
print(f"10-90% interval coverage 79.4% confirms well-calibrated uncertainty.")
print(f"Pessimistic-bound alert catches 99.5% of failures at the cost of 612 false alarms,")
print(f"consistent with cost_fn:cost_fp=10:1 favoring recall over precision.")
print("-> Phase 15 Done")
raise SystemExit