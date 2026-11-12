import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay
from DataloaderHeart import HeartDataset
import numpy as np
train_dataset = HeartDataset("heart.csv", split="train")
val_dataset   = HeartDataset("heart.csv", split="validate")
test_dataset  = HeartDataset("heart.csv", split="test")

X_train, y_train = train_dataset.features, train_dataset.labels
X_val,   y_val   = val_dataset.features,   val_dataset.labels
X_test,  y_test  = test_dataset.features,  test_dataset.labels

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"AUC:      {roc_auc_score(y_test, y_pred_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
np.save("xgb_probs.npy",  y_pred_prob)
np.save("xgb_labels.npy", y_test)
RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title("XGBoost ROC Curve")
plt.show()