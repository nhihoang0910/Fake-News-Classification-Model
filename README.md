# Fake-News-Classification-Model

## Table of Content  
1. [Project Background](#project-background)  
2. [Data Processing](#data-processing)  
3. [Model & Result](#model--result)  
4. [Limitations](#limitations)  

---

## Project Background  
This project aimed to build a fake news detection model using Decision Tree, Support Vector Machine (SVM), and Long SHort-Term Memory from Scikit-learn Libraries in Python. We used WELFake Dataset, uploaded on Kaggle, that consisted of 72,134 news articles with 35,028 real and 37,106 fake news. This dataset combined real news from 4 sites - Kaggle, McIntire, Reuters, BuzzFeed Political - to prevent overfitting.  
Find more details [HERE](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)

In this Github repo, I will only include the portion using SVM which was my part in this group project.

---

## Data Processing  
I used Scikit-learn and NLTK libraries to process textual data before training the model:  
1. Removed empty values  
2. Converted text to lowercase  
3. Tokenization: Broke down the whole sentence into smaller units (tokens)  
4. Removed stopwords: (e.g., “the,” “is,” “and”)  
5. Lemmatization: Converted the tokens back to their original form (e.g., “running” to “run”)  
6. Vectorization: Used TF-IDF vectorizer to count how many times each word appears and takes into account their significance

---

## Model & Results  
SVM aims to find the optimal hyperplane in N-dimensional space to separate data points into the targeted classes - fake and real news. SVM uses overfitting protection to handle data with large dimensionality. I trained the model using different kernel threshold functions to maximize the margin between the hyperplane and data points from each class. The model with Linear Kernel functions, which uses a straight line for classification, yielded the highest accuracy score on the testing dataset compared to other kernel functions - Polynomial, Sigmoid, and Radial Basis Function (RBF) classifiers.  
![Screenshot 2025-03-26 at 8 55 29 AM](https://github.com/user-attachments/assets/c6ecac6c-455e-46f1-9c20-15833bc2a783)

Below are the results for SVM model with linear kernel function. I did not find any significant difference in accuracy score after testing out different parameters for gamma, so I proceeded with the auto value for gamma. This model achieved 93.5% for all evaluation metrics, including accuracy score, precision, recall, and F-1 score. The accuracy score for the  Linear Kernel Function is higher than other kernels, but the difference is not significant. The ROC curve shows a slightly better result in making true predictions compared to the ROC curve of Decision Tree. The difference between the number of True Positive and True Negative predictions is less extreme compared to Decision Trees. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/edbcce68-5e78-4366-a53f-624a30a13609" alt="ROC Curve"/>
  <br>
  <em>Figure 1: ROC Curve</em>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/263d2461-8b4e-4353-aa12-d484330ca185" alt="Confusion Matrix" width = 200/>
  <br>
  <em>Figure 2: Confusion Matrix</em>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/6bc4a051-856d-4cee-ad0e-23cd0b9249e0" alt="Other measures" width = 200/>
  <br>
  <em>Figure 3: Other measures</em>
</p>

---

## Limitations  
Most fake news classification model used datasets that focused on particular subjects, mostly politics. Therefore they can perform really well on similar subjects but not on other areas. Furthermore, each industry’s article has different written format, so it is still challenging to train a general approach that performs well on all types of model. My model might not work well for articles that contain partially false and partially correct information
