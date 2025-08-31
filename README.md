# News-topic-classifier-using-Bert

# 📰 News Topic Classifier using BERT

This project is a **News Topic Classification System** that uses **BERT (Bidirectional Encoder Representations from Transformers)** to classify news articles into predefined categories (e.g., politics, sports, technology, business, etc.). The project leverages **deep learning with transformers** for natural language understanding and provides a **Streamlit web app** for user interaction.

---

## 📌 Project Overview
- Implemented a **fine-tuned BERT model** for text classification.
- Classifies news content into multiple categories with high accuracy.
- Built an interactive **Streamlit interface** where users can paste or upload news text to get instant predictions.
- Logging system implemented to track predictions and model performance.

---

## 🚀 Features
- **BERT-based deep learning model** for text classification.
- **Real-time predictions** via a simple Streamlit web app.
- **User-friendly interface** to input or paste news articles.
- **Logging system** to monitor and analyze predictions.
- Scalable for adding more categories and datasets.

---

## 🛠️ Tech Stack
- **Python** 🐍
- **Transformers (Hugging Face)** 🤗
- **PyTorch / TensorFlow** (depending on backend)
- **Scikit-learn**
- **Pandas & NumPy**
- **Streamlit** (for interactive UI)

---

## 📂 Project Structure
News-Topic-Classifier-BERT/
│-- main.py 
|-- Inferece.py
|-- train.py
│-- model/ # Trained BERT model & tokenizer
│-- logs/ # Logs of predictions
│-- README.md # Project documentation

📊 Dataset

Used a News Classification Dataset (e.g., AG News or any multi-class dataset).

Dataset was preprocessed and split into training, validation, and test sets.

🔮 Future Improvements

Add support for multilingual classification.

Integrate news article scraping for real-time classification.

Improve model efficiency with DistilBERT or ALBERT.

Deploy the app on Hugging Face Spaces or Streamlit Cloud.
