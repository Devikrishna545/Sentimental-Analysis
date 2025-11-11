# Sentiment Analysis

A machine learning project for performing multiclass sentiment analysis on text data using Python and Jupyter Notebook. This project classifies text into three sentiment categories: **Positive**, **Neutral**, and **Negative**.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a sentiment analysis system that can classify text data into three sentiment categories. Using the multiclass sentiment analysis dataset from Hugging Face, the model is trained to understand and predict the emotional tone of text input.

## ✨ Features

- **Multiclass Classification**: Classifies text into Positive (2), Neutral (1), or Negative (0) sentiments
- **Large Dataset**: Utilizes a comprehensive dataset with 31,232 training samples
- **Interactive Notebook**: Built using Jupyter Notebook for easy experimentation and visualization
- **Hugging Face Integration**: Seamless dataset loading from Hugging Face Hub
- **Data Analysis**: Includes exploratory data analysis and visualization capabilities

## 📊 Dataset

The project uses the [multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) from Hugging Face, which contains:

- **Training Set**: 31,232 labeled text samples
- **Validation Set**: Available for model tuning
- **Test Set**: For final model evaluation

### Dataset Structure

| Column | Description |
|--------|-------------|
| `id` | Unique identifier for each text sample |
| `text` | The text content to be analyzed |
| `label` | Numerical label (0: Negative, 1: Neutral, 2: Positive) |
| `sentiment` | Text representation of the sentiment |

### Sample Data

```
text: "Cooking microwave pizzas, yummy"
label: 2
sentiment: positive
```

## 🛠️ Technologies Used

- **Python 3.x**
- **Jupyter Notebook** - Interactive development environment
- **Pandas** - Data manipulation and analysis
- **Hugging Face Hub** - Dataset management and model hosting
- **Google Colab** - Cloud-based notebook execution (optional)

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Devikrishna545/Sentimental-Analysis.git
   cd Sentimental-Analysis
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install pandas
   pip install huggingface_hub
   pip install jupyter
   ```

4. **Set up Hugging Face Token (if required)**
   - Create an account on [Hugging Face](https://huggingface.co/)
   - Generate an access token from your settings
   - Store it securely (use environment variables or secrets management)

## 🚀 Usage

### Running the Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**
   - Navigate to `Sentimental_Analysis_.ipynb`
   - Run the cells sequentially

### Using Google Colab

1. Click the "Open in Colab" badge in the notebook
2. Upload your Hugging Face token to Colab Secrets
3. Run all cells

### Basic Code Example

```python
from google.colab import userdata
from huggingface_hub import login
import pandas as pd

# Login to Hugging Face
HF_TOKEN = userdata.get('secret_token_hugface')
login(HF_TOKEN)

# Load the dataset
splits = {'train': 'train_df.csv', 'validation': 'val_df.csv', 'test': 'test_df.csv'}
df = pd.read_csv("hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/" + splits["train"])

# View the data
print(df.head())
```

## 📁 Project Structure

```
Sentimental-Analysis/
│
├── Sentimental_Analysis_.ipynb    # Main Jupyter notebook
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
└── requirements.txt               # Python dependencies (to be added)
```

## 📈 Results

The sentiment analysis model categorizes text into three classes:

- **Positive (Label 2)**: Texts expressing positive emotions, satisfaction, or approval
- **Neutral (Label 1)**: Texts with neutral tone or factual statements
- **Negative (Label 0)**: Texts expressing negative emotions, dissatisfaction, or criticism

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Add more machine learning models (LSTM, BERT, Transformers)
- Implement model evaluation metrics
- Create visualization dashboards
- Add data preprocessing techniques
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [Sp1786](https://huggingface.co/Sp1786) on Hugging Face
- Hugging Face for their excellent tools and infrastructure
- The open-source community for various libraries and tools

## 📧 Contact

**Devikrishna545** - [@Devikrishna545](https://github.com/Devikrishna545)

Project Link: [https://github.com/Devikrishna545/Sentimental-Analysis](https://github.com/Devikrishna545/Sentimental-Analysis)

---

⭐ If you found this project helpful, please consider giving it a star!