# Spam_Email_Classifier-
This project aims to build a Spam Email Classifier using a Logistic Regression model implemented in PyTorch. The classifier is trained to distinguish between spam and non-spam (ham) emails based on their textual content. The model uses natural language processing (NLP) techniques to vectorize email text and apply a binary classification approach to detect spam.
# Features
- 🧠 **Logistic Regression Model** implemented in PyTorch
- 📝 **Binary classification**: Spam vs. Non-Spam
- 🔤 Text preprocessing using `CountVectorizer`
- 📊 Model evaluation using **accuracy score** and **confusion matrix**
- 📦 Lightweight and easy to train
# Tech Stack
- Python
- PyTorch
- Pandas
- Scikit-learn
- Matplotlib
# Dataset: "/kaggle/input/email-spam-classification-dataset/combined_data.csv"
# How It Works

1. Load Data: Reads email data from a CSV file.
2. Vectorization: Uses `CountVectorizer` to convert text into binary feature vectors.
3. Model Architecture:
   - One linear layer (`nn.Linear`)
   - Sigmoid activation for binary classification
4. Training:
   - Optimizer: Stochastic Gradient Descent (SGD)
   - Loss Function: Binary Cross-Entropy Loss
   - Trained over 100 epochs
5. Evaluation:
   - Computes prediction accuracy
   - Plots a confusion matrix
# Installation & Usage
Clone the Repository
git clone https://github.com/yourusername/spam-email-classifier-pytorch.git
cd spam-email-classifier-pytorch
# Install Dependencies
pip install torch pandas scikit-learn matplotlib
# Run the Script
python spam_classifier.py
