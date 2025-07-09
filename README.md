# 🚀 University Projects Portfolio - Ashionye Aninze

_MSc Artificial Intelligence | BSc Computer & Data Science_

This repository serves as a dynamic portfolio showcasing my academic projects completed during my Master of Science in Artificial Intelligence and Bachelor of Science in Computer & Data Science at Birmingham City University.

It highlights my practical skills in a wide array of data science and AI domains, including:
* Data Cleaning & Preprocessing 🧹
* Exploratory Data Analysis (EDA) & Visualisation 📊
* Machine Learning & Deep Learning Model Development 🧠
* **Responsible AI & Bias Mitigation** ⚖️ (a key area of interest and expertise, as demonstrated by my dissertation and accepted publication)
* Handling Imbalanced Datasets 💪
* Applying AI/ML to Real-World Problems, particularly in Healthcare 🏥

My primary aim is to develop fair, unbiased, and impactful AI systems that can address significant social challenges.

---

## 📚 Table of Contents

* [Project: Bias Detection in Machine Learning Models](https://github.com/AshAninze/Uni-Projects/tree/main?tab=readme-ov-file#%EF%B8%8F-project-bias-detection-in-machine-learning-models)
* [Project: Brain Tumour Detection from MRI Scans](https://github.com/AshAninze/Uni-Projects/tree/main?tab=readme-ov-file#-project-brain-tumour-detection-from-mri-scans)
* [Project: Credit Card Fraud Detection](https://github.com/AshAninze/Uni-Projects/tree/main?tab=readme-ov-file#-project-credit-card-fraud-detection)
* [Project: Obesity Risk Estimation and Analysis](https://github.com/AshAninze/Uni-Projects/tree/main?tab=readme-ov-file#-project-obesity-risk-estimation-and-analysis)
* [Technical Skills](https://github.com/AshAninze/Uni-Projects/tree/main?tab=readme-ov-file#%EF%B8%8F-technical-skills)
* [Connect With Me](https://github.com/AshAninze/Uni-Projects/tree/main?tab=readme-ov-file#-connect-with-me)

---

## ✨ My Featured Projects

### ⚖️ Project: Bias Detection in Machine Learning Models

* _University Module: Individual Honours Project_

**Description:**
Dedicated to advancing responsible AI, this project provides a deep dive into how to **detect bias within machine learning models**, by exploring the methods and metrics necessary to quantify potential disparities impacting various demographic groups. The goal was to understand the origins and manifestations of bias, and to evaluate its impact on model fairness, laying groundwork for developing more equitable and responsible AI systems. This work directly aligns with my dissertation research and published paper on ethical AI applications.

**Key Features/Techniques Used:**
* **Language:** Python 🐍
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, **Fairlearn**
* **Techniques:**
    * Bias Definition & Categorisation (e.g., demographic parity, equalised odds)
    * Fairness Metrics Calculation (e.g., Demographic Parity Difference, Equal Opportunity Difference)
    * Sensitive Attribute Identification & Analysis
    * Data Preprocessing for Fairness (e.g., reweighing)
    * Machine Learning Models (e.g., Logistic Regression, Decision Trees)
    * Exploratory Data Analysis (EDA) focused on identifying data disparities.
* **Concepts:** Ethical AI, Algorithmic Fairness, Responsible AI, Data Bias, Model Interpretability, Social Impact of AI.
* **Tools:** Google Colaboratory (Colab)

**Results/Insights:**
This project successfully demonstrated how to identify and measure bias in predictive models, revealing specific areas where models exhibited unfair performance or were impacted by data imbalances. It underscored the absolute necessity of integrating fairness considerations throughout the AI lifecycle to ensure equitable and trustworthy AI solutions.

➡️ [View Jupyter Notebook](https://github.com/AshAninze/Uni-Projects/blob/main/Bias_Detection.ipynb)

### 🧠 Project: Brain Tumour Detection from MRI Scans

* _University Module: Deep Neural Networks_

**Description:**
This project explores the application of deep learning techniques, specifically Convolutional Neural Networks (CNNs), for the classification of brain tumours from MRI images. The goal was to build a robust model capable of distinguishing between tumourous and non-tumourous scans, contributing to early detection and diagnosis.

**Key Features/Techniques Used:**
* **Language:** Python 🐍
* **Libraries:** Keras, TensorFlow, Matplotlib
* **Techniques:**
    * Image Preprocessing (resising, normalisation)
    * **Data Augmentation** (rotation, scaling, flipping to increase dataset diversity by 25% for improved robustness)
    * Convolutional Neural Networks (CNNs) for image classification
    * Model Training and Validation
    * Performance Evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
    * Optimising CNN layers (dropout, normalisation, activation functions)
* **Tools:** Google Colaboratory (Colab)

**Results/Insights:**
The developed CNN model demonstrated strong generalisation capabilities in identifying tumour cases, showcasing the effectiveness of deep learning for medical image analysis.

➡️ [View Jupyter Notebook](https://github.com/AshAninze/Uni-Projects/blob/main/Tumour_MRI.ipynb)

### 💳 Project: Credit Card Fraud Detection

* _University Module: Artificial Intelligence & Machine Learning_

**Description:**
This project focuses on building and evaluating machine learning models for **detecting fraudulent credit card transactions**. Utilising a highly imbalanced dataset, the core challenge was to accurately identify rare fraud instances while minimising false positives. The project involved extensive data preprocessing, exploration of various classification algorithms, and rigorous evaluation using appropriate metrics for imbalanced datasets, demonstrating expertise in handling real-world data challenges and developing robust anomaly detection systems.

**Key Features/Techniques Used:**
* **Language:** Python 🐍
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Imbalanced-learn
* **Techniques:**
    * Data Preprocessing (e.g., scaling, handling missing values)
    * Exploratory Data Analysis (EDA)
    * **Handling Imbalanced Datasets** (e.g., SMOTE, undersampling, cost-sensitive learning)
    * Machine Learning Models: (e.g., Logistic Regression, Random Forests, SVMs, Gradient Boosting)
    * Model Evaluation for Imbalanced Data (e.g., Precision, Recall, F1-Score, Confusion Matrix)
    * Cross-validation
* **Concepts:** Classification, Anomaly Detection, Supervised Learning

**Results/Insights:**
The models developed showcased strong capabilities in identifying fraudulent transactions, highlighting the critical importance of specific evaluation metrics and techniques when working with imbalanced datasets.

➡️ [View Jupyter Notebook](https://github.com/AshAninze/Uni-Projects/blob/main/Credit_Card_Fraud_Dataset.ipynb)

### 🍏 Project: Obesity Risk Estimation and Analysis

* _University Module: Computing for AI_

**Description:**
This project focused on the development of machine learning models to estimate and analyse the risk factors associated with obesity. Leveraging a dataset containing diverse health and lifestyle metrics, the aim was to identify significant predictors of obesity and build predictive models to aid in early risk assessment and intervention strategies, contributing to understanding public health challenges.

**Key Features/Techniques Used:**
* **Language:** Python 🐍
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Techniques:**
    * Data Preprocessing & Cleaning (e.g., label encoding, standardisation)
    * Exploratory Data Analysis (EDA) & Data Visualisation
    * Feature Selection/Engineering
    * Machine Learning Models: **Random Forest** (achieved 99.29% accuracy), and comparative analysis of other algorithms
    * Model Optimisation using randomised search cross-validation
    * Model Evaluation (Precision, Recall, F1-score)
* **Concepts:** Classification, Public Health Data Analysis, Predictive Modeling in Healthcare.

**Results/Insights:**
The developed models successfully identified key lifestyle and demographic factors correlated with obesity risk and achieved high predictive accuracy, providing actionable insights for public health interventions.

➡️ [View Jupyter Notebook](https://github.com/AshAninze/Uni-Projects/blob/main/Estimation_of_Obesity.ipynb)

---

## 🛠️ Technical Skills

Here's an overview of the technical skills demonstrated across these projects and my broader experience:

* **Languages:** Python, Java, SQL, JavaScript, Linux, R
* **Developer Tools:** Git, Docker, Terraform, MLFlow, Apache Airflow, Eclipse, Colab
* **Libraries:** Pandas, NumPy, Matplotlib, TensorFlow, PyTorch, Keras, Scikit-learn, Fairlearn, Databricks, PySpark
* **Key Skills:** Machine Learning, Data Science, Deep Learning, **Bias Mitigation**, **Ethical AI**, Reproducible Analytical Pipelines (RAP)

---

## 📞 Connect With Me

I'm always open to discussing data science, AI, and ethical applications. Feel free to connect!

* [LinkedIn](https://uk.linkedin.com/in/ashionye-a-6a26ba205)
* [Email](mailto:aaaninze@gmail.com)
* [ORCID](https://orcid.org/0009-0007-8902-367X)
