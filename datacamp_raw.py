## --------------------------------------------- ##
## Explainable AI in Python (DataCamp)
## --------------------------------------------- ##

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree export_text

model = DecisionTreeClassifier(random_state=42, max_depth=2)
model.fit(X_train, y_train)

rules = export_text(model, feature_names = list(X_train.columns))
print(rules)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

## --------------------------------------------- ##

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(36, 12), random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

## --------------------------------------------- ##


from sklearn.preprocessing import MinMaxScalar
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt

# Standardize the training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression()

# Fit the model
model.fit(X_train_scaled, y_train)

# Derive coefficients
coefficients = model.coef_
feature_names = X_train.columns

# Plot coefficients
plt.bar(feature_names, coefficients)
plt.show()

# Logistic Regression for Classification
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Derive coefficients
coefficients = model.coef_[0]
feature_names = X_train.columns

# Plot coefficients
plt.bar(feature_names, coefficients)
plt.show()


"""
Decision Tree
* Fundamental block for tree-based model, for regression and classification
* Inherently explainable

Random forest, many decision trees. Consider the feature importance. 
Measures reduction of uncertainty in predictions across the decision trees. 
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

tree_model = DecisionTreeClassifier()
forest_model = RandomForestClassifier()

tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

# Feature Importance (reduction in impurity)
print(tree_model.feature_importances_)
print(forest_model.feature_importances_)

# Feature importance plots
plt.barh(X_train.columns,
	tree_model.feature_importances_)
plt.title('Feature Importance - Decision Tree')
plt.show()

## --------------------------------------------- ##

from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

model = MLPClassifier(hidden_layer_sizes=(10), random_state=1)
model.fit(X, y)

# Compute the permutation importance
result = permutation_importance(model,
	X, y,
	n_repeats = 10,
	random_state = 1,
	scoring = "accuracy")

# Plot feature importances
plt.bar(X.columns, result.importances_mean)
plt.xticks(rotation=45)
plt.show()

## --------------------------------------------- ##


from sklearn.inspection import permutation_importance

# Extract and store model coefficients
coefficients = model.coef_[0]

# Compute permutation importance on the test set
perm_importance = permutation_importance(model,
	X, y,
	n_repeats = 20,
	random_state = 1,
	scoring = "accuracy")

# Compute the average permutation importance
avg_perm_importance = perm_importance.importances_mean

def plot_importances(coefficients,  perm_importances):
    features = X.columns  

    x = np.arange(len(features))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax1 = plt.subplots()

    # Plotting coefficients on the primary y-axis
    rects1 = ax1.bar(x - width/2, np.abs(coefficients), width, label='Coefficients', color='b')
    ax1.set_ylabel('Coefficient Magnitude', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=45)
    
    # Creating a secondary y-axis for permutation importances
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, perm_importances, width, label='Permutation Importance', color='g')
    ax2.set_ylabel('Permutation Importance', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Adding title and adjusting layout
    ax1.set_title('Logistic Regression Coefficients vs. Permutation Importance')
    fig.tight_layout()

    # Adding legends outside the plot
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Placing the legend to the right outside of the plot
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(2, 1), borderaxespad=0.)

    # Adjusting the layout to avoid overlap
    plt.subplots_adjust(right=0.8)  # Allows space for the legend on the right

    plt.show()

plot_importances(coefficients, avg_perm_importance)

## --------------------------------------------- ##

## SHAP: MODEL AGNOSTIC EXPLAINABILITY

'''
SHAP -> SHAPley Additive Explanations
Model-Agnostic Technique, leveraging SHAPley values from Game Theory
Quantify feature contributions to predictions

> General Explainers: Can be applied to any model
> Type-Specific Explainers: Optimized for specific model types

Tree Explainers for Tree-Based Models
'''

import shap

# Create a SHAP Tree Explainer (model is a regressor model)
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values.mean(axis=0))

plt.bar(X.columns, mean_abs_shap)
plt.title('Mean Absolute SHAP Values for RandomForest')
plt.xticks(rotation=45)
plt.show()

## --------------------------------------------- ##

import shap

# Create a SHAP Tree Explainer (model is a RandomForestClassifier)
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values (sleect values of the positive class)
mean_abs_shap = np.abs(shap_values[:,:,1].mean(axis=0))

plt.bar(X.columns, mean_abs_shap)
plt.title('Mean Absolute SHAP Values for RandomForest')
plt.xticks(rotation=45)
plt.show()

## --------------------------------------------- ##

'''
Kernel Explainers derives SHAP values for any models
Slower than type-specific explainer
'''

import shap

# Create a SHAP Kernel Explainer (Regression Model)
explainer = shap.KernelExplainer(
	# Model's prediction function
	model.predict,
	# Representative summary of dataset
	shap.kmeans(X, 10)
	)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values.mean(axis=0))

plt.bar(X.columns, mean_abs_shap)
plt.title('Mean Absolute SHAP Values for MLPRegressor')
plt.xticks(rotation=45)
plt.show()

## --------------------------------------------- ##

import shap

# Create a SHAP Kernel Explainer
explainer = shap.KernelExplainer(
	# Model's prediction function
	model.predict_proba,
	# Representative summary of dataset
	shap.kmeans(X, 10)
	)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values[:,:,1].mean(axis=0))

plt.bar(X.columns, mean_abs_shap)
plt.title('Mean Absolute SHAP Values for MLPClassifier')
plt.xticks(rotation=45)
plt.show()

## --------------------------------------------- ##

import shap

# Extract model coefficients
coefficients = model.coef_[0]

# Compute SHAP values
explainer = shap.KernelExplainer(
	model.predict_proba,
	shap.kmeans(X,10)
	)
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values[:,:,1].mean(axis=0))

def plot_importances(coefficients,  perm_importances):
    features = X.columns  

    x = np.arange(len(features))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax1 = plt.subplots()

    # Plotting coefficients on the primary y-axis
    rects1 = ax1.bar(x - width/2, np.abs(coefficients), width, label='Coefficients', color='b')
    ax1.set_ylabel('Coefficient Magnitude', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=45)
    
    # Creating a secondary y-axis for permutation importances
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, perm_importances, width, label='SHAP values', color='g')
    ax2.set_ylabel('Mean Absolute SHAP value', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Adding title and adjusting layout
    ax1.set_title('Logistic Regression Coefficients vs. SHAP values')
    fig.tight_layout()

    # Adding legends outside the plot
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()

plot_importances(coefficients, mean_abs_shap)

## --------------------------------------------- ##

## Feature Importance Plots

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Derive shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Plot the feature importance plot
shap.summary_plot(shap_values, X_train, plot_type = "bar")

## --------------------------------------------- ##

## Analyzing Feature Effects with Beeswarm Plots
# Highlights both direction and magnitude of each feature on prediction

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Derive shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Plot the beeswarm plot
shap.summary_plot(shap_values, X_train,
	plot_type = "dot")

## --------------------------------------------- ##

# Assessing Impact with Partial Dependence Plots
# See how changes in features affect probability of admission
# For each sample: vary value of selected feature, holding all other features constant
# ... predict outcome, average results from all samples.

from sklearn.ensemble import RandomForestRegressor
import shap

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Generate the partial dependence plot for CGPA
shap.partial_dependence_plot("CGPA", model.predict, X_train)
# Generate the partial dependence plot for University Rating
shap.partial_dependence_plot("University Rating", model.predict, X_train)

## --------------------------------------------- ##

## Local Explainability Concepts

# SHAP Waterfall Plots: Shows how features increase or decrease model's prediction
# Baseline: Model's average prediction across all samples; starting point for plot.

# sample code:
shap.waterfall_plot(
	shap.Explanation(
		values=shap_values[:,1],
		base_values=explainer.expected_value[1],
		data=test_instance, # where `test_instance = X.iloc[0,:]`
		feature_names=X.columns
		)
	)

import shap

# Create the SHAP explainer
explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X, 10))

# Compute SHAP values for the first instance in X
shap_values = explainer.shap_values(X.iloc[0, :])

print(shap_values)

# Plot the SHAP values using a waterfall plot
shap.waterfall_plot(
	shap.Explanation(
		values=shap_values[:,1],
		base_values=explainer.expected_value[1],
		data=X.iloc[0,:],
		feature_names=X.columns
		)
	)

## --------------------------------------------- ##

## LIME: Local Interpretable Model-Agnostic Explanations
# Explains predictions of complex models, works on individual instances
# Agnostic to model type. Generates perturbations around a sample, sees effect on output.

# Tailored for different types of data:
# - Tabular Explainer
# - Text Explainer
# - Image Explainer

from lime.lime_tabular import LimeTabularExplainer

sample_data_point = X.iloc[2, :]

# Create the explainer
explainer = LimeTabularExplainer(
	X.values,
	feature_names = X.columns,
	mode = "regression"
	)

# Generate the explanation
exp = explainer.explain_instance(
	sample_data_point.values,
	model.predict
	)

# Display the explanation
exp.as_pyplot_figure()
plt.show()

## --------------------------------------------- ##

# Interpreting classifiers locally

from lime.lime_tabular import LimeTabularExplainer

sample_data_point = X.iloc[2, :]

# Create the explainer
explainer = LimeTabularExplainer(
	X.values,
	feature_names = X.columns,
	mode = "classification"
	)

# Generate the explanation
exp = explainer.explain_instance(
	sample_data_point.values,
	model.predict_proba
	)

# Display the explanation
exp.as_pyplot_figure()
plt.show()

## --------------------------------------------- ##

# Text Explainability with LIME
# Explaining sentiment analysis predictions

from lime.lime_text import LimeTextExplainer

text_instance = "Amazing battery life and the camera quality is perfect! I highly recommend this smartphone."

# Create a LIME text explainer
explainer = LimeTextExplainer()

# Assume you have a model_predict function for processing input texts
# Returns class probabilities
def model_predict(instance):
	...
	return class_probabilities

# Generate the explanation
exp = explainer.explain_instance(
	text_instance,
	model_predict
	)

# Display the explanation
exp.as_pyplot_figure()
plt.show()

## --------------------------------------------- ##
# LIME Image Explainer
# Find which parts of image impacts predictions

from lime import lime_image
np.random.seed(10)

# Create a LIME explainer
explainer = lime_image.LimeImageExplainer()

# Generate the explanation
explanation = explainer.explain_instance(image, model_predict, hide_color=0, num_samples=50)

# Display the explanation
temp, _ = explanation.get_image_and_mask(explanation.top_labels[0], hide_rest = True)
plt.imshow(temp)
plt.title('LIME Explanation')
plt.axis('off')
plt.show()

## --------------------------------------------- ##

## Advanced Topics In Explainable AI:
# 		* Explainability Metrics
# 		* Explaining Unsupervised Models
# 		* Explaining Generative AI models

## Principle of Consistency: Assessing stability of explanations when model is trained
# on different subsets (of the full data). Low consistency -> no robust explanations.

## Computing Consistency

from sklearn.metrics.pairwise import cosine_similarity

explainer1 = shap.TreeExplainer(model1)
explainer2 = shap.TreeExplainer(model2)

shap_values1 = explainer1.shap_values(X1)
shap_values2 = explainer2.shap_values(X2)

feature_importance1 = np.mean(np.abs(shap_values1), axis = 0)
feature_importance2 = np.mean(np.abs(shap_values2), axis = 0)

consistency = cosine_similarity([feature_importance1],[feature_importance2])
print("Consistency between SHAP values:", consistency)

## Cosine Similarity: 
# -1: Opposite Explanations, 0: No Consistent Explanations, 1: Highly Consistent Explanations

## --------------------------------------------- ##

## Principle of Faithfulness: Evaluates if important features influence model's predictions
# Low faithfulness -> misleads trust in model reasoning (useful in sensitive applications)

X_instance = X_test.iloc[0]
original_prediction = model.predict_proba(X_instance)[0, 1]
print(f"Original prediction: {original_prediction}")

X_instance['GRE Score'] = 310
new_prediction = model.predict_proba(X_instance)[0, 1]
print(f"Prediction after perturbing {important_feature}: {new_prediction}")

faithfulness_score = np.abs(original_prediction - new_prediction)
print(f"Local Faithfulness Score: {faithfulness_score}")

# High faithfulness score indicates how well it aligns with model's behaviour when
# the most important feature is perturbed.

## --------------------------------------------- ##

## Explaining Unsupervised Models (e.g. Clustering Models)

## Feature Impact on Cluster Quality:

# 	Silhouette Score: Measures clustering's quality (-1,1)
# 	Impact(f2) = Silhouette(f1,f2) - Silhouette(f1)
# 	Impact(f) > 0 implies positive contribution for f, else, f introduces noise

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=2).fit(X)
original_score = silhouette_score(X, kmeans.labels_)

for i in range(X.shape[1]):
	X_reduced = np.delete(X, i, axis = 1)
	kmeans.fit(X_reduced)
	new_score = silhouette_score(X_reduced, kmeans.labels_)
	impact = original_score - new_score
	print(f'Feature {column_names[i]}: Impact = {impact}')

# Explore how individual features impact the clustering model performance

## --------------------------------------------- ##

## Feature Importance for Cluster Assignments

# 	Adjusted Rand Index (ARI): Measures how well cluster assignments match
# 	Maximum ARI = 1 > perfect cluster alignment
# 	Lower ARI > greater difference in clusterings

# Intuition: Remove features one at a time
# 	Importance(f) = 1 - ARI(original clusters, modified clusters)
# 	Low(ARI) -> high(1-ARI) -> important feature

from sklearn.metrics import adjusted_rand_score

kmeans = KMeans(n_clusters=2).fit(X)
original_clusters = kmeans.predict(X)

for i in range(X.shape[1]):
	X_reduced = np.delete(X, i, axis=1)
	reduced_clusters = kmeans.fit_predict(X_reduced)
	importance = 1 - adjusted_rand_score(original_clusters, reduced_clusters)
	print(f'{df.columns[i]}: {importance}')

## --------------------------------------------- ##

# Explainability for LLMs:
# Chain-of-Thought Prompting (Understanding reasoning process)
# Zero-Shot COT Prompting

prompt = """A shop starts with 20 apples. It sells 5 apples and then receives 8 more.
How many apples does the shop have now? Show your reasoning step-by-step."""
response = get_response(prompt)
print(response)

# Self-Consistency (Text Classification): Assess model's confidence in generated answers
# 	- Multiple Explanations Should Agree: internal reasoning (COT) and final output
# 	- Consistent outputs indicate model's reasoning is robust. Reflects reliability.

prompt = """Classify the following review as positive or negative.
You should reply with either "positive" or "negative", nothing else.
Review: 'The customer service was great, but the product itself did not meet my expectations.
'"""
responses = [] # aggregation of results
for i in range(5):	# simulating multiple sampling
	sentiment = get_response(review)
	responses.append(sentiment.lower())

confidence = { 	# inferring model's confidence
	'positive': responses.count('positive') / len(responses),
	'negative': responses.count('negative') / len(responses)
}

print(confidence)

## --------------------------------------------- ##

# Things I need to know:
# Code. Intuition. Theory. Visualizations Plots.
# What each plot communicates. How to read/interpret.
