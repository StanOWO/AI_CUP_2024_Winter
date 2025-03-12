# AI CUP 2024 Winter

This repository contains the work for two competitions organized during the AI CUP 2024 Winter:

1. **Competition to Predict Power Generation from Regional Microclimate Data**  
   This challenge focuses on forecasting the 10-minute average solar panel power generation (in mW) based on microclimate measurements from 17 monitoring sites.

2. **YuShan AI Public Challenge – RAG & LLM in Financial Q&A**  
   This challenge involves applying a Retrieval-Augmented Generation (RAG) approach combined with Large Language Models (LLMs) for financial question answering. In this project, financial documents (insurance policies, finance reports, FAQ data) are preprocessed and indexed, and a language model–based retrieval algorithm is used to match questions with the most relevant document.

Additionally, the repository includes an overall project report called **report.pdf** that provides background, methodology, and competition results.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Competitions Overview](#competitions-overview)
  - [Competition to Predict Power Generation from Regional Microclimate Data](#competition-to-predict-power-generation-from-regional-microclimate-data)
  - [YuShan AI Public Challenge – RAG & LLM in Financial Q&A](#yushan-ai-public-challenge--rag--llm-in-financial-qa)
- [Installation and Dependencies](#installation-and-dependencies)
- [Usage Instructions](#usage-instructions)
- [Results and Evaluation](#results-and-evaluation)
- [Contact](#contact)

---

## Repository Structure

The directory structure is organized as follows:

```
AI_CUP_2024_Winter/
│
├── Competition to predict power generation based on regional/  
│   ├── run.ipynb                  # Jupyter Notebook for the indirect multi-layer prediction pipeline  
│   ├── run.py                     # Command-line script for running the prediction pipeline  
│   ├── utils/                     # Helper functions for preprocessing data  
│   ├── model/                     # Modules for model training and evaluation (XGBoost, KNN, Decision Trees, etc.)
│   │   ├── pytorch_model.py       # PyTorch-based deep learning regression models
│   ├── dataset/                   # Folder for training CSV files (microclimate data)  
│   └── submit/                    # Folder where prediction output CSV files are saved  
│
├── YuShan AI Public Challenge – RAG & LLM in Financial Q&A/  
│   ├── run.ipynb                  # Notebook demonstrating the retrieval pipeline using LMIR  
│   ├── reference/                 # Contains reference documents and FAQ data:  
│   │    ├── faq/                 # FAQ JSON files (e.g., pid_map_content.json)  
│   │    ├── insurance/           # PDF files for insurance policies  
│   │    └── finance/             # PDF files for finance reports  
│   └── dataset/preliminary/       # Contains sample questions (questions_example.json), ground truths (ground_truths_example.json), stopwords, dictionary files, and prompt text  
│
├── README.md                      # This file  
└── report.pdf                     # Comprehensive project report with background, methods, and results
```

---

## Competitions Overview

### Competition to Predict Power Generation from Regional Microclimate Data

**Objective:**  
Predict the 10-minute average power generation (in mW) of solar panels using microclimate data collected from 17 sites over approximately 2000 days. The data includes:
- **Features:**  
  - Location code  
  - DateTime components (year, month, day, hour, minute, second)  
  - Microclimate variables: WindSpeed (m/s), Pressure (hPa), Temperature (°C), Humidity (%), Sunlight (Lux)
- **Target:**  
  - Average power generation per minute (mW)

**Approach:**  
- **Indirect Multi-Layer Prediction Pipeline:**  
  A four-layer strategy is adopted where:  
  1. **Layer 1:** Predicts initial microclimate features (WindSpeed and Pressure) from basic temporal attributes.  
  2. **Layer 2:** Uses basic features plus Layer 1 outputs to predict Temperature and Humidity.  
  3. **Layer 3:** Incorporates previous predictions along with basic features to predict Sunlight (Lux).  
  4. **Layer 4:** Uses all the accumulated features to directly forecast power generation.

- **Models Implemented:**  
  The solution leverages traditional machine learning models (such as XGBoost, K-Nearest Neighbors, and Decision Trees) for each layer, with additional experiments using PyTorch-based deep learning models.

### YuShan AI Public Challenge – RAG & LLM in Financial Q&A

**Objective:**  
Develop a system that retrieves the most relevant document for a given financial query. The system is designed to serve as the retrieval component of a RAG architecture, which could later be extended with LLM-based answer generation.

**Data Sources:**
- **FAQ Data:** Provided in JSON format (e.g.,pid_map_content.json).
- **Insurance Documents:** PDF files containing policy clauses.
- **Finance Reports:** PDF files with financial disclosures.

**Approach:**  
- **Data Extraction:**  
  The system extracts text from PDFs using pdfplumber and processes FAQ data from JSON files.

- **Text Preprocessing:**  
  Chinese text is cleaned and segmented using the jieba tokenizer. A custom dictionary (dict.txt.big) and a list of domain-specific keywords (from prompt.txt) enhance the segmentation quality.

- **Retrieval using LMIR:**  
  A language model–based retrieval algorithm with Dirichlet smoothing (LMIR) tokenizes both queries and documents. It computes the probability that a document “generates” a query and then ranks the documents accordingly. The document with the highest score for each query is selected as the answer.

- **Evaluation:**  
  Accuracy is computed by matching predicted document IDs with ground truth entries across categories (insurance, finance, FAQ).

---

## Installation and Dependencies

**Requirements:**
- Python 3.10 or later

**Example Installation:**  
Use the provided `requirements.txt` (generated by pipreqs)
```bash
pip install -r requirements.txt
```

or install dependencies via pip:
```bash
pip install pdfplumber jieba torch numpy pandas tqdm scikit-learn statsmodels xgboost
```

Make sure to adjust package versions if necessary.

---

## Usage Instructions

### For the Power Generation Prediction Competition
1. **Data Preparation:**  
   Place the 17 training CSV files into the `Competition to predict power generation based on regional/dataset/` folder and ensure that the test CSV file is available in `Competition to predict power generation based on regional/submit/`.

2. **Running the Pipeline:**  
   - **Notebook:** Open `run.ipynb` within the respective folder to run the layer-by-layer prediction interactively.
   - **Script:** Run the following command from the command line:
     ```bash
     python run.py
     ```
   The final power generation predictions will be saved as a CSV file in the `submit/` folder.

### For the YuShan AI Public Challenge – RAG & LLM in Financial Q&A
1. **Data Preparation:**  
   Verify that the sample data (questions, ground truths, stopwords, dictionary, prompt) are in `YuShan AI Public Challenge – RAG & LLM in Financial Q&A/dataset/preliminary/` and reference PDFs are organized under the `reference/` subfolders.

2. **Running the Retrieval Pipeline:**  
   - **Notebook:** Open `run.ipynb` within this folder to run through the retrieval process, which includes text extraction, preprocessing, and document retrieval using the LMIR algorithm.
   - **Script:** Alternatively, execute:
     ```bash
     python run.py
     ```
   The retrieval results will be output as a JSON file (e.g., `pred_retrieve.json`) in the appropriate directory.

3. **Evaluation:**  
   The system will evaluate the retrieval accuracy by comparing the predicted document IDs against the ground truth data and display category-wise and overall accuracy.

---

## Results and Evaluation

- **Scoring (Power Generation Competition):**  
  The evaluation metric is based on the absolute error between predicted and actual 10-minute average power values. Various performance metrics (including MSE and R² scores) are computed on the validation set.

- **Scoring (Financial Q&A Challenge):**  
  Retrieval performance is measured via accuracy based on how many queries retrieve the correct document (as per the ground truth). Output includes both category-specific and overall accuracy.

Reference the overall **report.pdf** for detailed analysis, competition results, and insights.

---

## Rank

- **Competition to Predict Power Generation from Regional Microclimate Data**: 35/934
- **YuShan AI Public Challenge – RAG & LLM in Financial Q&A**: 60/218

---
## Contact

For questions or contributions, please contact:  
l125879368@gmail.com

