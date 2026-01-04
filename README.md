# 🔍 Fake News Detector - AI-Powered Verification System

A beautiful, user-friendly Streamlit web application that uses 4 powerful machine learning models to detect fake news articles with 98-99% accuracy.

![Fake News Detector](https://img.shields.io/badge/Accuracy-99%25-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red) ![ML Models](https://img.shields.io/badge/ML%20Models-4-orange)

## ✨ Features

- **📝 Single Article Analysis** - Analyze individual articles with instant results
- **📊 Batch Processing** - Upload and analyze multiple articles at once
- **🔗 URL Analysis** - Extract and analyze articles directly from web URLs
- **📜 History Tracking** - Save and review past analyses
- **💾 Export Functionality** - Download results as CSV or JSON
- **📈 Beautiful Visualizations** - Interactive charts and confidence scores
- **🎨 Modern UI/UX** - Color-coded results with clear visual indicators

## 🤖 ML Models

The system uses **4 different machine learning algorithms** working together:

1. **Logistic Regression** - 98.62% accuracy
2. **Decision Tree** - 99.55% accuracy
3. **Gradient Boosting** - 99.54% accuracy
4. **Random Forest** - 99.73% accuracy

Each model independently analyzes articles, and the system provides both individual predictions and a consensus result.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (one-time setup, takes 5-10 minutes):
   ```bash
   python train_models.py
   ```

   This will:
   - Load 44,878 news articles (fake and true)
   - Train all 4 ML models
   - Save models as `.pkl` files in the `models/` folder
   - Display accuracy scores

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
Fake News/
├── app.py                      # Main Streamlit application
├── train_models.py             # Model training script
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration and colors
│
├── models/                     # Trained models (after running train_models.py)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── random_forest.pkl
│   ├── vectorizer.pkl
│   └── model_metadata.json
│
├── utils/                      # Utility modules
│   ├── text_preprocessing.py  # Text cleaning functions
│   ├── model_manager.py       # Model loading and caching
│   ├── prediction.py          # Prediction logic
│   └── web_scraper.py         # URL content extraction
│
├── components/                 # Streamlit UI components
│   ├── single_analysis.py     # Single article analysis
│   ├── batch_analysis.py      # Batch processing
│   ├── url_analysis.py        # URL analysis
│   ├── history_viewer.py      # History management
│   └── visualizations.py      # Charts and graphs
│
├── data/                       # Training datasets
│   ├── Fake.csv               # 23,481 fake news articles
│   └── True.csv               # 21,417 true news articles
│
└── history/                    # Analysis history
    └── analysis_history.json  # Saved analyses
```

## 💡 How to Use

### Single Article Analysis

1. Navigate to **"📝 Single Analysis"** from the sidebar
2. Paste a news article in the text area (minimum 50 characters)
3. Click **"🔍 Analyze Article"**
4. View predictions from all 4 models
5. See the consensus result
6. Optionally save to history or download the summary

### Batch Analysis

1. Navigate to **"📊 Batch Analysis"**
2. Upload a CSV file with a 'text' column, or TXT file with articles
3. Or manually paste multiple articles (separated by blank lines)
4. Click **"🔍 Analyze All"**
5. View results in table format
6. Download results as CSV or JSON

### URL Analysis

1. Navigate to **"🔗 URL Analysis"**
2. Paste a news article URL
3. Click **"📥 Fetch Article"**
4. Review the extracted text (edit if needed)
5. Click **"🔍 Analyze"**
6. View predictions

### History

1. Navigate to **"📜 History"**
2. View all past analyses
3. Filter by result type or agreement level
4. Export history as CSV or JSON
5. Delete individual entries or clear all

## 🎨 UI/UX Features

- **Color-Coded Results:**
  - 🟢 Green = True News
  - 🔴 Red = Fake News
  - 🟡 Orange = Warning

- **Visual Indicators:**
  - ✅ Checkmarks for true predictions
  - ❌ X-marks for fake predictions
  - 🎯 Consensus banners
  - Progress bars for confidence scores

- **Interactive Charts:**
  - Confidence score bar charts
  - Model agreement visualizations
  - History timeline graphs
  - Batch analysis statistics

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 98.62%   | 98.6%     | 98.6%  | 98.6%    |
| Decision Tree       | 99.55%   | 99.6%     | 99.5%  | 99.5%    |
| Gradient Boosting   | 99.54%   | 99.5%     | 99.5%  | 99.5%    |
| Random Forest       | 99.73%   | 99.7%     | 99.7%  | 99.7%    |

**Average Accuracy: 99.36%**

## 🔧 Configuration

Edit `config.py` to customize:

- Color scheme
- Minimum text length
- Maximum history entries
- TF-IDF features
- File paths

## 📦 Dependencies

- **streamlit** - Web framework
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **plotly** - Interactive visualizations
- **beautifulsoup4** - Web scraping
- **validators** - URL validation
- **reportlab** - PDF generation

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Improve documentation
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: Fake and real news articles dataset
- Built with Streamlit
- Machine learning models: scikit-learn
- Visualizations: Plotly

## 📧 Support

If you encounter any issues or have questions:

1. Check the console for error messages
2. Ensure all dependencies are installed
3. Make sure models are trained (`python train_models.py`)
4. Verify that `data/Fake.csv` and `data/True.csv` exist

## 🎯 Future Enhancements

- [ ] Add more ML models
- [ ] Implement BERT-based models
- [ ] Add multilingual support
- [ ] Create mobile app version
- [ ] Add real-time news feed analysis
- [ ] Implement user accounts
- [ ] Add API endpoint for external integrations

---

**Made with ❤️ using Streamlit and Machine Learning**

*Accuracy is based on training data and may vary with different types of articles. Always verify important news from multiple reliable sources.*
