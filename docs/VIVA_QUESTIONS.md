# CSE570 Viva Preparation — Questions and concise answers

1. **What is the project?**  
   A binary classifier that predicts whether a customer subscribes to a bank term deposit.

2. **Why is this classification?**  
   The target has two categories: yes and no.

3. **What is the dataset source?**  
   UCI Machine Learning Repository, Bank Marketing dataset ID 222.

4. **How many rows and features?**  
   The bank-full release has 45,211 records, 16 predictors and one target column.

5. **What is the target variable?**  
   `y`, indicating term-deposit subscription.

6. **Why exclude duration?**  
   It is only known after the call, so it leaks post-contact information into a pre-call prediction task.

7. **What does class imbalance mean here?**  
   There are substantially more no outcomes than yes outcomes.

8. **Why is accuracy insufficient?**  
   A majority-class model can have high accuracy while failing to identify subscribers.

9. **What is precision?**  
   TP/(TP+FP): reliability of positive predictions.

10. **What is recall?**  
    TP/(TP+FN): proportion of actual subscribers identified.

11. **What is F1?**  
    Harmonic mean of precision and recall.

12. **What is ROC-AUC?**  
    Probability-style ranking measure of how well positives are ranked above negatives across thresholds.

13. **Why PR-AUC?**  
    It focuses on positive-class precision and recall and is informative under imbalance.

14. **What is a baseline?**  
    A simple reference model, here a majority-class DummyClassifier.

15. **Why Logistic Regression?**  
    It is a relatively interpretable linear classifier and provides probabilities.

16. **Why Decision Tree?**  
    It models nonlinear splits and is relatively easy to explain, though prone to overfitting.

17. **Why Random Forest?**  
    It averages many trees to reduce variance and model nonlinear interactions.

18. **What is boosting?**  
    Sequentially builds weak learners to correct errors made by earlier learners.

19. **Why use a Pipeline?**  
    It ensures preprocessing is fitted only on each training fold and prevents leakage.

20. **Why one-hot encode?**  
    Most classifiers require numeric input; one-hot encoding represents categories without imposing false order.

21. **Why scale numeric variables?**  
    It helps scale-sensitive models such as Logistic Regression.

22. **Why median imputation?**  
    The median is less sensitive to extreme numeric values than the mean.

23. **What is stratified splitting?**  
    It preserves approximately the same target proportions in train and test partitions.

24. **What is cross-validation?**  
    Repeated train/validation folds to estimate generalization more robustly than one split.

25. **What is GridSearchCV?**  
    Exhaustive evaluation of specified hyperparameter combinations using cross-validation.

26. **What is a p-value?**  
    Under the null hypothesis and assumptions, the probability of observing a result at least as extreme as the one obtained.

27. **Does statistical significance imply causation?**  
    No. It indicates evidence of association/difference, not causal effect.

28. **What does Cramér's V measure?**  
    Strength of association between categorical variables.

29. **Why Mann–Whitney U?**  
    It compares rank distributions between two independent groups without requiring normally distributed data.

30. **What is permutation importance?**  
    It measures the drop in model score when a feature is randomly shuffled.

31. **What is a false positive?**  
    Predicting subscription when the customer does not subscribe.

32. **What is a false negative?**  
    Missing a customer who would subscribe.

33. **What is target leakage?**  
    Using information unavailable at prediction time or information derived from the target.

34. **What are limitations?**  
    Historical, institution-specific data; imbalance; possible drift; unknown categories; no causal inference; no external validation.

35. **How could the project improve?**  
    Validate on newer data, calibrate probabilities, tune the decision threshold on validation data, monitor drift and evaluate business utility.

Viva instruction: Run the script yourself, open the outputs, and be able to explain your actual selected model and actual confusion matrix. Do not memorize made-up metrics.
