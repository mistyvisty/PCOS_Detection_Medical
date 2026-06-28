# 🩺 PCOS Detection & Risk Prediction

**Preeti Bhardwaj** | [github.com/mistyvisty](https://github.com/mistyvisty)

ML classification model to detect Polycystic Ovary Syndrome (PCOS) risk from clinical indicators, built with production-grade pipeline practices and full explainability.

---

## 🎯 Problem

PCOS affects 1 in 10 women of reproductive age and is frequently underdiagnosed. Early detection from routine clinical data (hormone levels, BMI, cycle data) can significantly improve outcomes. This model predicts PCOS risk as a binary classification task.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Mean CV AUC (5-fold) | **0.97** |
| Model | Random Forest |
| Validation | Stratified 5-Fold Cross-Validation |

---

## ⚙️ Pipeline

Raw clinical data → EDA + outlier analysis (with medical justification) → Feature engineering → SMOTE inside `ImbPipeline` (no leakage across folds) → Random Forest → Threshold tuning → SHAP explainability

**No-leakage design:** SMOTE is applied inside the pipeline per fold, not before splitting — a common mistake this project explicitly avoids.

---

## 🔍 Key Features (via SHAP)

- Follicle count (left & right ovary)
- LH/FSH hormone ratio
- BMI
- Cycle regularity
- Skin darkening / hair growth indicators

Outliers in hormone values were retained with medical justification — elevated LH:FSH ratio is a clinically documented PCOS marker, not noise.

---

## 🛠️ Tech Stack

`Python` `scikit-learn` `XGBoost` `Random Forest` `SMOTE` `imbalanced-learn` `ImbPipeline` `SHAP` `matplotlib` `seaborn`

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
jupyter notebook PCOS_Detection_Medical.ipynb
```

---

## 🔗 Related Projects

- [PCOS Neurodivergence RAG](https://github.com/mistyvisty/pcos-neurodivergence-rag) — RAG chatbot answering PCOS + neurodivergence queries
- [Advanced RAG with Pinecone](https://github.com/mistyvisty/pcos-neurodivergence-advanced-rag-pinecone) — Hybrid search upgrade with BM25 + re-ranking
- [Fine-tuned Phi-3 on PCOS data](https://github.com/mistyvisty/pcos-finetune-phi3-lora) — QLoRA fine-tuning on PCOS QA dataset

---



