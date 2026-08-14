# Capstone Project Final Report  

## Project Title : Recruitment Acceptance Prediction

**Author:** Kim Kok (kimkok@live.com)

**Notebook:** [`Capstone_Final_Project_Recruitment_Acceptance_Prediction.ipynb`](./Capstone_Final_Project_Recruitment_Acceptance_Prediction.ipynb)

**Colab Link:** https://colab.research.google.com/github/kimkok-UCBerkeleyHaas/Capstone_Project-24-1_Final_Report/blob/main/Capstone_Final_Project_Recruitment_Acceptance_Prediction.ipynb

## Executive Summary

This project predicts whether a candidate will accept or decline a job offer, using historical recruitment data (`hr_data.csv`, ~8,995 records). Through EDA, baseline modeling (Logistic Regression, Random Forest), and an enhancement pass (hyperparameter tuning, SMOTE, gradient boosting, a stacking ensemble), the best-performing model — a **Stacking Ensemble** of Logistic Regression, a tuned Random Forest, and HistGradientBoosting — lifts test-set **Accuracy from 62.4% (baseline Logistic Regression) to 81.9%**, **Recall from 59.9% to 96.7%**, and reaches an **ROC-AUC of 0.760**. Interestingly, the single untuned baseline Random Forest actually edges out every other model on ROC-AUC alone (0.762), a useful reminder that "more complex" doesn't automatically mean "better on every metric" — see [Model Performance](#model-performance) below for the full picture and why the Stacking Ensemble is still the recommended production model.

The analysis shows that **notice period**, the **gap between offered and expected CTC hike**, **candidate source**, **line of business (LOB)**, and **location** are the most meaningful drivers of acceptance. A deeper causal analysis (Propensity Score Matching) also uncovered and corrected two issues in the original EDA-stage narrative: a data-leakage artifact in the "relocation" feature, and an overstated joining-bonus effect that turns out to be statistically null. Both corrections are described below and in the notebook.

## Rationale

Recruitment is a resource-intensive process. When candidates decline offers after extensive interviewing, it represents wasted recruiter time, a longer time-to-hire, and lost opportunity cost. Understanding what drives acceptance decisions lets a recruitment team:

- **Allocate resources efficiently** — focus recruiter follow-up on candidates with a higher predicted decline risk.
- **Improve offer strategy** — know which offer components (CTC, notice period, joining bonus) actually move the needle.
- **Reduce time-to-hire** — streamline the process for candidates who are likely to accept regardless.
- **Increase acceptance rates** — proactively address the factors most associated with decline.

## Research Question

**What factors most significantly influence a candidate's decision to accept or decline a job offer, and can we predict decline risk early enough to intervene?**

Specifically:
1. Which recruitment-lifecycle metrics (notice period, time to accept) correlate with acceptance?
2. How do candidate profile characteristics (experience, source, location) affect the decision?
3. What financial factors (expected vs. offered CTC hike, joining bonus) drive acceptance?
4. Can we build an accurate predictive model to identify candidates likely to decline?

## Data Sources

### 1. HR Data (`hr_data.csv`) — primary dataset
- **Size:** 8,995 records (8,308 unique candidates after de-duplication), 18 columns.
- **Description:** Recruitment-lifecycle records with the real target of this project.
- **Key features:** DOJ Extended, Duration to accept offer, Notice period, Offered band, % hike expected/offered, CTC % difference, Joining Bonus, Candidate relocation status, Gender, Candidate Source, Experience (Rex in Yrs), LOB, Location, Age, and **Status** (target: Joined / Not Joined).
- **Source:** `https://raw.githubusercontent.com/kimkok-UCBerkeleyHaas/Assignments/refs/heads/main/datasets/hr_data.csv`

### 2. Recruitment Data (`recruitment_data.csv`) — secondary dataset, context only
- **Size:** 1,500 records, 11 columns.
- **Description:** A *different* dataset with a *different* target (`HiringDecision`, i.e. whether the **recruiter** extends an offer, not whether the **candidate** accepts one). It is loaded and explored briefly for context in the notebook but is **not** used to train the acceptance-prediction models.
- **Source:** `https://raw.githubusercontent.com/kimkok-UCBerkeleyHaas/Assignments/refs/heads/main/datasets/recruitment_data.csv`

## Methodology

### Data Cleaning & Feature Engineering
- No missing values in either dataset.
- Binary Yes/No fields mapped to 1/0 (`DOJ_Extended`, `Relocate_Required`, `Joining_Bonus`, `Gender_Binary`).
- Categorical fields (LOB, Candidate Source, Offered Band, Location) label-encoded for the tree models.
- Engineered features: `Salary_Satisfaction` (offered hike ≥ expected hike), `Expected_Offered_Ratio`, `Experience_Category`, `Location_Category`, `Age_Category`, `CTC_Diff_Flag` (Positive/Negative/Zero); later extended with `Exp_x_Hike` (interaction), `CTC_Diff_Squared` (non-linear), and `Notice_Period_Group` (binned) for the enhanced models.

### Exploratory Data Analysis
- Univariate distributions for the target and all continuous/categorical features.
- Bivariate analysis: correlation matrix, boxplots of numeric features by outcome, and acceptance-rate breakdowns by every categorical feature.
- All EDA figures use real, computed values pulled directly from the cleaned dataframe (no illustrative placeholder numbers).

### Modeling
- **Baseline models:** Logistic Regression (scaled features, `class_weight='balanced'`) and Random Forest (`class_weight='balanced'`), 80/20 stratified train/test split.
- **Feature selection:** Recursive Feature Elimination (RFE) with a Logistic Regression estimator, to identify the top predictors independent of a tree model's importance ranking.
- **Model enhancement:**
  - `GridSearchCV` (5-fold `StratifiedKFold`, scoring = ROC-AUC) to tune the Random Forest.
  - `SMOTE` to rebalance the training set (imbalanced 5,850:1,346 → balanced 5,850:5,850).
  - `HistGradientBoostingClassifier`, alone and combined with SMOTE.
  - A `StackingClassifier` combining Logistic Regression, the tuned Random Forest, and HistGradientBoosting, with a Logistic Regression meta-learner.
- **Deeper analysis:** segment-level performance breakdown (by LOB, experience band, candidate source), a temporal-drift check (using candidate reference order as a proxy for time, since no real date field exists), and a causal analysis using Propensity Score Matching to test the joining-bonus effect while controlling for confounders.

### Evaluation Metric Rationale
The target is imbalanced (~81% Joined / 19% Not Joined), so **accuracy alone is misleading** — a model that always predicts "Joined" would already score ~81%. **ROC-AUC** is used as the primary ranking metric because it is insensitive to class balance and measures how well a model separates the two classes across all thresholds. **Recall and F1 on the minority ("Declined") class** are tracked alongside it, since for this business problem a **missed decliner (false negative)** — a candidate the model wrongly reassures the recruiter about — is more costly than a false alarm.

## Results

### Key EDA Findings (real, computed values)

| Cut | Acceptance rate |
|---|---|
| Overall | 81.3% Accepted / 18.7% Declined |
| CTC-difference sign: Zero / Positive / Negative | 85.2% / 82.6% / 77.4% |
| Top locations: Others / Mumbai / Cochin / Noida / Ahmedabad | 100.0% / 89.3% / 87.5% / 86.6% / 83.3% |
| Top LOBs: MMS / INFRA / ETS / Healthcare / CSMP | 100.0% / 87.8% / 83.1% / 82.3% / 81.5% |
| Joining Bonus: No / Yes | 81.3% / 80.6% |
| Candidate Source: Employee Referral / Direct / Agency | 88.0% / 82.0% / 75.8% |

Correlation with the target (`Status_Binary`, Joined = 1):

| Feature | Correlation with acceptance |
|---|---|
| Notice period | **-0.192** (longer notice period → lower acceptance) |
| Duration to accept offer | -0.065 |
| Age | 0.046 |
| Rex in Yrs (experience) | -0.038 |
| Percent difference CTC | 0.033 |
| Percent hike offered in CTC | 0.028 |
| Percent hike expected in CTC | 0.000 |

*(Age vs. Experience: 0.568 · Expected vs. Offered hike: 0.669 — both sanity-check as expected.)*

> ⚠️ **Data-quality correction:** an early feature-importance pass ranked `Relocate_Required` as the single most predictive feature. The causal-analysis section later found this field is **100% co-occurrent with `Status = Joined`** — it records *actual* post-hoc relocation, which by definition can only be observed for candidates who already joined. This is a data-leakage artifact, not a genuine predictor, and is excluded from the findings below.

### Model Performance

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.7326 | 0.6242 | 0.9078 | 0.5988 | 0.7216 |
| Random Forest (Baseline) | **0.7620** | 0.7421 | 0.8930 | 0.7758 | 0.8303 |
| Tuned Random Forest (GridSearchCV) | 0.7563 | 0.7160 | 0.8987 | 0.7334 | 0.8077 |
| Gradient Boosting (HistGB) | 0.7505 | 0.7126 | 0.8903 | 0.7375 | 0.8067 |
| Gradient Boosting + SMOTE | 0.7442 | 0.8093 | 0.8585 | 0.9166 | 0.8866 |
| **Stacking Ensemble (LR + RF + HGB)** | 0.7596 | **0.8193** | 0.8363 | **0.9672** | **0.8970** |

**Interpretation:**
- By **ROC-AUC alone**, the untuned baseline Random Forest (0.7620) is actually the single best-ranking model — narrowly ahead of the Stacking Ensemble (0.7596). This is a legitimate finding on this particular 80/20 split, not an error: simple, well-regularized models can be very competitive on pure ranking quality.
- By **Accuracy, Recall, and F1**, the **Stacking Ensemble** is unambiguously the best model, and is the one recommended for production, since minimizing missed decliners (Recall) matters more operationally here than a marginal ROC-AUC edge.
- **SMOTE was the single biggest driver of improvement** for Recall/F1 — both SMOTE-based models (Gradient Boosting + SMOTE, Stacking) dramatically outperform their non-SMOTE counterparts on catching candidates likely to accept, and the Stacking Ensemble in particular pushes Recall to 96.7%.
- Hyperparameter tuning alone (Tuned Random Forest) gave only a small, stability-focused improvement over the untuned baseline — the untuned model was already reasonably well-specified for this data.
- **Caveat:** even after SMOTE, the *minority* "Not Joined" (Declined) class remains harder to predict than "Joined" (see the per-model confusion matrices in the notebook) — precision/recall on that specific class stays modest, so the model should be used as a decline-risk flag paired with recruiter judgment, not a fully automated decision-maker.

### Feature Importance (Random Forest, top 10, excluding the leaky `Relocate_Required` field)

1. **Duration to accept offer** (0.129)
2. **Notice period** (0.110)
3. **Percent hike expected in CTC** (0.075)
4. **Percent hike offered in CTC** (0.074)
5. **Percent difference CTC** (0.074)
6. **LOB (encoded)** (0.066)
7. **Rex in Yrs (experience)** (0.062)
8. **Age** (0.057)
9. **Location (encoded)** (0.044)

RFE (Recursive Feature Elimination, Logistic Regression estimator) independently confirms **Notice period**, relocation-related fields (flagged above as leaky), **Offered band**, **Candidate Source (Agency)**, and several **LOB**/**Location** categories as the top drivers — consistent with the tree-based importance ranking.

### Segmentation, Temporal, and Causal Findings

- **Segmentation:** The Stacking Ensemble is most accurate for CSMP/INFRA candidates (~87–88%) and weakest for BFSI/ERS (~77–79%) and Agency-sourced candidates (76.4%) — segments that also have the lowest actual acceptance rates and thus the most "Declined" cases for the model to get right. Recommendation: weight the model's decline predictions more heavily for BFSI, ERS, and Agency-sourced candidates, and more skeptically for INFRA/ETS (where it almost never predicts a decline).
- **Temporal (proxy):** neither dataset has a real date/timestamp field. Using candidate-reference order as a rough proxy for chronology shows only a weak overall drift (correlation ≈ 0.30) across ten sequential cohorts, with one anomalous late cohort (much shorter notice periods, higher acceptance) that looks like a distinct hiring batch rather than a genuine time trend. Recommendation: capture real application/offer timestamps going forward.
- **Causal analysis (Propensity Score Matching):** confirms the `Relocate_Required` leakage finding above, and tests whether **joining bonuses** causally affect acceptance (controlling for age, experience, notice period, CTC gap, LOB, location, offered band, source, and gender). Both the naive gap and the PSM-adjusted estimate are ≈ **-0.8 percentage points** — i.e., **no meaningful causal or correlational effect** of joining bonuses on acceptance. Recommendation: prioritize CTC alignment and notice-period negotiation over bonus offers as levers to improve acceptance.

## Actionable Recommendations for HR (non-technical summary)

1. **Close the salary gap where possible.** Offers with a hike below what the candidate expected see a meaningfully lower acceptance rate (77.4% vs. 82.6–85.2%). Where budget allows, aligning the offered hike with the expected hike is the single most controllable lever.
2. **Watch the notice period.** Longer notice periods are associated with lower acceptance — candidates with more time before their prior obligations end have more opportunity to receive or accept a competing offer.
3. **Segment recruiter attention.** BFSI, ERS, and Agency-sourced pipelines have both lower actual acceptance rates and are harder for the model to get right — these deserve more hands-on recruiter follow-up than INFRA/ETS or Employee-Referral pipelines.
4. **Don't rely on joining bonuses as a lever.** The causal analysis found no meaningful effect on acceptance — this budget is likely better spent elsewhere.
5. **Fix the data:** the "relocation requirement" field currently only captures relocation *after the fact* (so it's meaningless as a leading indicator); if relocation truly matters, it needs to be captured as a pre-offer requirement instead.
6. **Deploy the Stacking Ensemble as a decline-risk flag**, not a fully automated filter — pair its predictions with a manual review step for borderline cases, given the minority "Declined" class remains the hardest to predict precisely.

## Next Steps

- **Model refresh & monitoring:** periodic retraining (e.g. quarterly) and holdout validation as new recruitment cycles complete.
- **Real timestamps:** add application/offer date fields to replace the current cohort-order proxy with genuine seasonality/trend analysis.
- **Segment-specific models:** consider a dedicated or rebalanced model for Agency-sourced and BFSI/ERS candidates, where the current model has the most room to improve.
- **Decision support tooling:** wrap the Stacking Ensemble in a simple scoring dashboard for recruiters, surfacing a decline-risk flag at the moment an offer is extended.
- **A/B validation:** test whether acting on the model's flags (e.g., targeted CTC renegotiation for high-risk candidates) measurably improves acceptance rates in practice.

## Outline of Project

- [`Capstone_Final_Project_Recruitment_Acceptance_Prediction.ipynb`](./Capstone_Final_Project_Recruitment_Acceptance_Prediction.ipynb) — the full, executed notebook: business understanding, EDA, baseline modeling, RFE, hyperparameter tuning, SMOTE, gradient boosting, stacking ensemble, segmentation/temporal/causal analysis, and final findings.
- `README.md` — write-up of the final project report.

### How to run
1. Open the notebook in Jupyter or Google Colab.
2. Run all cells top to bottom — it pulls both CSVs directly from GitHub (or a local copy if present), so no manual upload is required.
3. Requires `scikit-learn` and `imbalanced-learn` (for `SMOTE`); both are `pip install`-able in a standard Colab runtime.
