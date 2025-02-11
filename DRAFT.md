# Explainable AI in Python: A Comprehensive Tutorial

This tutorial provides a deep dive into Explainable AI (XAI) using Python. We will cover foundational concepts, model agnostic techniques, local explanation methods, and advanced topics in explainability. By combining theoretical insights with practical code examples and visualizations, you will learn how to interpret, diagnose, and trust your machine learning models.

## Table of Contents

1. [Chapter 1: Foundations of Explainable AI](#chapter-1-foundations-of-explainable-ai)
2. [Chapter 2: Model Agnostic Explainability](#chapter-2-model-agnostic-explainability)
3. [Chapter 3: Local Explainability](#chapter-3-local-explainability)
4. [Chapter 4: Further Concepts in Explainable AI](#chapter-4-further-concepts-in-explainable-ai)

---

## Chapter 1: Foundations of Explainable AI

### Overview
In this chapter, we introduce Explainable AI and discuss its importance in building transparent and trustworthy machine learning systems. We start with interpretable models like decision trees and linear/logistic regression, which offer inherent insights into their decision-making processes.

### Theory and Significance
- **Explainability**: Provides insights into how and why models make predictions.
- **Interpretable Models**: Some models (e.g., decision trees) are inherently understandable, as they expose their decision rules.
- **Linear Models**: The coefficients in linear and logistic regression directly indicate feature influence.
- **Neural Networks**: Although inherently complex, we can still derive explanations (e.g., via permutation importance).

### Core Code and Syntax

#### Decision Trees for Classification
```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# Initialize and train a decision tree classifier
model = DecisionTreeClassifier(random_state=42, max_depth=2)
model.fit(X_train, y_train)

# Export and print the decision tree rules
rules = export_text(model, feature_names=list(X_train.columns))
print(rules)

# Predict and compute accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

#### Neural Network (MLP Classifier)
```python
from sklearn.neural_network import MLPClassifier

# Initialize and train an MLP classifier
model = MLPClassifier(hidden_layer_sizes=(36, 12), random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

#### Linear and Logistic Regression with Visualization
```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt

# Standardize the training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Linear Regression Example
model = LinearRegression()
model.fit(X_train_scaled, y_train)
coefficients = model.coef_
feature_names = X_train.columns

# Plot coefficients for Linear Regression
plt.bar(feature_names, coefficients)
plt.title('Linear Regression Coefficients')
plt.show()

# Logistic Regression Example
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
coefficients = model.coef_[0]

# Plot coefficients for Logistic Regression
plt.bar(feature_names, coefficients)
plt.title('Logistic Regression Coefficients')
plt.show()
```

### Example Code
A sample demonstration of printing decision tree rules:
```python
# Print the exported decision tree rules to understand the splits
print(rules)
```
*This output shows the logical rules the tree uses to make predictions.*

### Visualization
Bar plots created with `matplotlib` allow you to visually assess the magnitude and direction of model coefficients in linear and logistic regression models.

---

## Chapter 2: Model Agnostic Explainability

### Overview
Model agnostic explainability techniques are designed to work with any type of machine learning model. This chapter covers methods such as permutation importance and SHAP (SHAPley Additive Explanations) that provide insights into feature contributions without needing to inspect the model internals.

### Theory and Significance
- **Permutation Importance**: Measures the decrease in model performance when a feature’s values are shuffled.  
- **SHAP Values**: Based on game theory, SHAP values quantify the contribution of each feature by comparing the prediction with and without the feature.
- **Universality**: These methods work regardless of whether the underlying model is a tree, neural network, or any other type.

### Core Code and Syntax

#### Permutation Importance Example
```python
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

model = MLPClassifier(hidden_layer_sizes=(10,), random_state=1)
model.fit(X, y)

# Compute permutation importance
result = permutation_importance(model, X, y, n_repeats=10, random_state=1, scoring="accuracy")

# Plot the permutation importances
plt.bar(X.columns, result.importances_mean)
plt.xticks(rotation=45)
plt.title('Permutation Importance')
plt.show()
```

#### SHAP Tree Explainer
```python
import shap

# Create a SHAP Tree Explainer (works best for tree-based models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Compute mean absolute SHAP values
mean_abs_shap = np.abs(shap_values.mean(axis=0))

plt.bar(X.columns, mean_abs_shap)
plt.title('Mean Absolute SHAP Values for RandomForest')
plt.xticks(rotation=45)
plt.show()
```

#### SHAP Kernel Explainer (Model-Agnostic)
```python
import shap

# Create a SHAP Kernel Explainer for any model
explainer = shap.KernelExplainer(
    model.predict,
    shap.kmeans(X, 10)
)

# Compute SHAP values
shap_values = explainer.shap_values(X)
mean_abs_shap = np.abs(shap_values.mean(axis=0))

plt.bar(X.columns, mean_abs_shap)
plt.title('Mean Absolute SHAP Values for MLPRegressor')
plt.xticks(rotation=45)
plt.show()
```

### Example Code
Using SHAP for a classification model (focusing on the positive class):
```python
# For a RandomForestClassifier, compute and visualize SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# For classification, extract values corresponding to the positive class (e.g., index 1)
mean_abs_shap = np.abs(shap_values[:,:,1].mean(axis=0))

plt.bar(X.columns, mean_abs_shap)
plt.title('Mean Absolute SHAP Values for RandomForest')
plt.xticks(rotation=45)
plt.show()
```

### Visualization
Visual outputs include bar plots for both permutation importance and SHAP values. These plots help you understand which features are most influential regardless of the model used.

---

## Chapter 3: Local Explainability

### Overview
Local explainability methods help understand why a model made a specific prediction for an individual data instance. This chapter focuses on techniques such as LIME and SHAP waterfall plots that provide fine-grained explanations.

### Theory and Significance
- **Local Explanations**: Provide instance-level insights rather than global model behavior.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Works by perturbing an instance and observing the impact on the prediction to build an interpretable local surrogate model.
- **SHAP Waterfall Plots**: Show how each feature drives a single prediction away from a baseline value.

### Local Explainability with LIME

LIME (Local Interpretable Model-agnostic Explanations) helps explain the predictions of complex machine learning models on a per-instance basis, making it a powerful tool for local interpretability.

#### Key Points:
- **Types of LIME Explainers:**
  - **LIME Tabular Explainer:** Used for structured/tabular data.
  - **LIME Text Explainer:** Used for textual data.
  - **LIME Image Explainer:** Used for image data.
- **How LIME Works:**
  - **Perturbation Generation:** Creates slight variations around a specific instance to observe changes in model output.
  - **Local Model Building:** Constructs a simplified model to explain the influence of individual features on the prediction.

#### Using LIME Tabular Explainer:

1. **Import the Explainer:**
   ```python
   from lime.lime_tabular import LimeTabularExplainer
   ```

2. **Initialize the Explainer:**
   ```python
   # Assuming 'X' is your dataset
   explainer = LimeTabularExplainer(
       X.values,
       feature_names=X.columns,
       mode='regression'  # Use 'classification' for classification problems
   )
   ```

3. **Generate an Explanation:**

   - *For Regression:*
     ```python
     sample_data_point = X.iloc[2, :]
     exp = explainer.explain_instance(
         sample_data_point.values,
         model.predict  # Replace with your regression model's prediction function
     )
     exp.as_pyplot_figure()
     plt.show()
     ```

   - *For Classification:*
     ```python
     sample_data_point = X.iloc[2, :]
     exp = explainer.explain_instance(
         sample_data_point.values,
         model.predict_proba  # Replace with your classification model's probability prediction function
     )
     exp.as_pyplot_figure()
     plt.show()
     ```

#### SHAP Waterfall Plot for Local Explanations
```python
import shap

# Assume we use a previously defined explainer (e.g., from SHAP Tree or Kernel Explainer)
# Compute SHAP values for a single instance (e.g., first instance in X)
test_instance = X.iloc[0, :]
shap_values = explainer.shap_values(X)

# Plot a waterfall plot for a specific class (e.g., index 1 for classification)
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[:,1],
        base_values=explainer.expected_value[1],
        data=test_instance,
        feature_names=X.columns
    )
)
```

#### LIME for Text Data
```python
from lime.lime_text import LimeTextExplainer

text_instance = "Amazing battery life and the camera quality is perfect! I highly recommend this smartphone."

# Create a LIME text explainer
explainer = LimeTextExplainer()

# Define a dummy model prediction function (to be replaced with actual model predictions)
def model_predict(instance):
    # ... your model prediction code returning class probabilities ...
    return class_probabilities

# Generate and display the explanation
exp = explainer.explain_instance(
    text_instance,
    model_predict
)
exp.as_pyplot_figure()
plt.show()
```

#### LIME for Image Data
```python
from lime import lime_image
import numpy as np

np.random.seed(10)
# Assume 'image' is your input image array and model_predict is defined for image classification

# Create a LIME image explainer
explainer = lime_image.LimeImageExplainer()

# Generate the explanation
explanation = explainer.explain_instance(image, model_predict, hide_color=0, num_samples=50)

# Retrieve and display the explanation image and mask
temp, _ = explanation.get_image_and_mask(explanation.top_labels[0], hide_rest=True)
plt.imshow(temp)
plt.title('LIME Explanation')
plt.axis('off')
plt.show()
```

### Example Code
A complete example for local explainability using LIME for a tabular classification problem:
```python
from lime.lime_tabular import LimeTabularExplainer

sample_data_point = X.iloc[2, :]

# Create the explainer for classification
explainer = LimeTabularExplainer(
    X.values,
    feature_names=X.columns,
    mode="classification"
)

# Generate the explanation using predict_proba
exp = explainer.explain_instance(
    sample_data_point.values,
    model.predict_proba
)

# Display the explanation plot
exp.as_pyplot_figure()
plt.show()
```

### Visualization
Local explanation methods produce intuitive visual outputs:
- **LIME Explanation Plots:** Highlight which parts of the input (features, words, or image regions) drive the model’s prediction.
- **SHAP Waterfall Plots:** Visually break down the prediction, showing the contribution of each feature relative to a baseline.

---

## Chapter 4: Further Concepts in Explainable AI

### Overview
This chapter delves into advanced topics in explainability, such as consistency, faithfulness, unsupervised model explanations, and explainability in large language models (LLMs). These topics address the robustness and reliability of explanations.

### Theory and Significance
- **Consistency:** Evaluates the stability of explanations when a model is trained on different subsets of data. High consistency indicates robust and reliable explanations.
- **Faithfulness:** Measures if the identified important features truly drive the model’s predictions. Faithful explanations help build trust in the model’s reasoning.
- **Unsupervised Models:** Explainability methods can be extended to clustering and generative models, offering insights into feature contributions that affect model performance (e.g., via silhouette scores or adjusted Rand Index).
- **LLM Explainability:** Techniques such as chain-of-thought prompting and self-consistency provide insights into the reasoning behind large language model outputs.

### Core Code and Syntax

#### Assessing Consistency with SHAP Values
```python
from sklearn.metrics.pairwise import cosine_similarity

# Assume model1 and model2 are trained on different subsets of data
explainer1 = shap.TreeExplainer(model1)
explainer2 = shap.TreeExplainer(model2)

shap_values1 = explainer1.shap_values(X1)
shap_values2 = explainer2.shap_values(X2)

# Compute average feature importance from SHAP values
feature_importance1 = np.mean(np.abs(shap_values1), axis=0)
feature_importance2 = np.mean(np.abs(shap_values2), axis=0)

# Calculate cosine similarity to assess consistency
consistency = cosine_similarity([feature_importance1], [feature_importance2])
print("Consistency between SHAP values:", consistency)
```

#### Evaluating Faithfulness
```python
# Compare predictions before and after perturbing an important feature
X_instance = X_test.iloc[0]
original_prediction = model.predict_proba(X_instance)[0, 1]
print(f"Original prediction: {original_prediction}")

# Perturb a key feature (e.g., 'GRE Score')
X_instance_perturbed = X_instance.copy()
X_instance_perturbed['GRE Score'] = 310  # New value
new_prediction = model.predict_proba(X_instance_perturbed)[0, 1]
print(f"Prediction after perturbing 'GRE Score': {new_prediction}")

faithfulness_score = np.abs(original_prediction - new_prediction)
print(f"Local Faithfulness Score: {faithfulness_score}")
```

#### Explainability for Unsupervised Models (Clustering)
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2).fit(X)
original_score = silhouette_score(X, kmeans.labels_)

# Evaluate the impact of each feature on clustering quality
for i in range(X.shape[1]):
    X_reduced = np.delete(X, i, axis=1)
    kmeans.fit(X_reduced)
    new_score = silhouette_score(X_reduced, kmeans.labels_)
    impact = original_score - new_score
    print(f'Feature {X.columns[i]}: Impact = {impact}')
```

#### Feature Importance for Cluster Assignments
```python
from sklearn.metrics import adjusted_rand_score

kmeans = KMeans(n_clusters=2).fit(X)
original_clusters = kmeans.predict(X)

# Evaluate feature importance using Adjusted Rand Index (ARI)
for i in range(X.shape[1]):
    X_reduced = np.delete(X, i, axis=1)
    reduced_clusters = kmeans.fit_predict(X_reduced)
    importance = 1 - adjusted_rand_score(original_clusters, reduced_clusters)
    print(f'{X.columns[i]}: {importance}')
```

#### Explainability for LLMs
```python
# Chain-of-Thought (CoT) Prompting Example for LLMs
prompt = """A shop starts with 20 apples. It sells 5 apples and then receives 8 more.
How many apples does the shop have now? Show your reasoning step-by-step."""
response = get_response(prompt)
print(response)

# Self-Consistency Example for Sentiment Analysis
prompt = """Classify the following review as positive or negative.
You should reply with either "positive" or "negative", nothing else.
Review: 'The customer service was great, but the product itself did not meet my expectations.'"""
responses = []  # Collect responses over multiple samples
for i in range(5):  # Simulate multiple sampling
    sentiment = get_response(prompt)
    responses.append(sentiment.lower())

confidence = {
    'positive': responses.count('positive') / len(responses),
    'negative': responses.count('negative') / len(responses)
}

print("LLM Confidence:", confidence)
```

### Example Code
An example to assess consistency using cosine similarity:
```python
# Compute consistency between SHAP-based feature importances from two models/datasets
consistency = cosine_similarity([feature_importance1], [feature_importance2])
print("Consistency between SHAP values:", consistency)
```

### Visualization
Visualizations in this chapter include:
- **Bar and Line Plots:** To display the impact of feature perturbations on clustering and prediction.
- **Printed Metrics:** Such as cosine similarity scores for consistency and faithfulness scores for local explanations.
- **LLM Output Examples:** Printed responses for chain-of-thought and self-consistency evaluations.

---

# Conclusion

This tutorial provided a comprehensive walkthrough of Explainable AI in Python—from building interpretable models to applying advanced techniques for local and model-agnostic explanations. By interleaving theory with practical code examples and visualizations, you now have a robust framework for applying XAI methods to various models and datasets.