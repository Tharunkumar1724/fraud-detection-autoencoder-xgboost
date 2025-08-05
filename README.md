# 🕵️‍♂️ Fraud Detection System: Autoencoder + XGBoost

This project detects fraudulent activity in customer billing data using a hybrid **unsupervised + supervised machine learning pipeline**, paired with an interactive frontend dashboard.

---

## 🔧 Tech Stack

- **Autoencoder (Keras)** – for anomaly detection
- **XGBoost** – for binary fraud classification
- **FastAPI** – backend API for fraud investigation & dashboard data
- **React + Chart.js** – dynamic frontend dashboard with fraud insights
- **MySQL** – structured data storage (customer, billing, transaction)
- **Google Colab** – model training and experimentation environment

---

## 🧠 Key Techniques

- Feature engineering & cleaning of customer billing datasets
- Dimensionality reduction via Autoencoders
- Fraud scoring based on anomaly reconstruction error
- Classification using XGBoost with threshold tuning
- Real-time fraud detection dashboard with region-wise insights

---

## 📊 React Frontend Overview

The frontend is built using **React** with the following features:

- 🔍 **Searchable interface** for customer accounts
- 📊 **Interactive visualizations** using `Chart.js` (consumption, billing, fraud probability)
- 📁 **Modular layout** (Dashboard, Customer Info, Fraud Panel, Settings)
- 📱 **Responsive UI** for desktop & mobile
- 🔌 Ready to connect with FastAPI backend for real-time updates

React folder structure:

