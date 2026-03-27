# 📱 PlayStore Review Analyzer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Dreem--R-black.svg)](https://github.com/Dreem-R)

## 📌 Overview

**PlayStore Review Analyzer** is an AI-powered web application that extracts valuable insights from Google Play Store app reviews. Using advanced Natural Language Processing (NLP) and machine learning models, it analyzes user feedback to identify key issues, sentiment trends, and actionable improvement suggestions for app developers.

The application provides:
- **Real-time Review Scraping** from Google Play Store
- **AI-Powered Sentiment Analysis** with visual dashboards
- **Intelligent Issue Extraction** using transformers and keyword analysis
- **Automated Improvement Suggestions** based on user feedback
- **Exportable Analytics** in CSV format

## ✨ Key Features

### 🔍 Smart Review Analysis
- Scrapes up to 500 recent reviews from any Google Play Store app
- Categorizes reviews into Positive, Neutral, and Negative sentiments
- Identifies rating-based sentiment patterns

### 🤖 AI-Powered Insights
- Uses **DistilBART** or **T5-Small** transformers for review summarization
- Extracts key issues from negative reviews using hybrid keyword + ML approach
- Identifies patterns across multiple reviews for comprehensive analysis

### 💡 Actionable Suggestions
- Generates targeted improvement recommendations based on detected issues
- Covers categories: UI/UX, Performance, Security, App Size, Ads, Authentication
- Provides specific, implementation-focused suggestions

### 📊 Interactive Visualizations
- Beautiful sentiment distribution pie chart powered by Plotly
- Real-time analysis feedback with loading indicators
- Responsive, dark-themed UI for better readability

### 📥 Data Export
- Download complete analysis as CSV file
- Includes user names, ratings, dates, reviews, and sentiment classification
- Perfect for further analysis or team sharing

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.35.0 |
| **Data Processing** | Pandas 2.2.2 |
| **ML/NLP** | Transformers 4.41.2, PyTorch 2.3.1 |
| **Web Scraping** | google-play-scraper 1.2.7 |
| **Visualization** | Plotly 5.22.0 |
| **Text Processing** | NLTK 3.8.1 |
| **Language Tokenization** | SentencePiece 0.2.0 |

## 📋 Requirements

- Python 3.8 or higher
- ~4GB RAM (minimum for transformer models)
- Internet connection (for Play Store API access)
- Pip package manager

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Dreem-R/PlayStore-review-analyzer.git
cd PlayStore-review-analyzer
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data (Optional but Recommended)
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### Step 5: Run the Application
```bash
streamlit run play_store.py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### Basic Workflow

1. **Enter App Name**
   - Type the name of any Google Play Store app in the search box
   - Default search term is "Facebook"

2. **Select Number of Reviews**
   - Choose between analyzing Top 50, Top 100, or Top 500 reviews
   - More reviews = more comprehensive analysis (but takes longer)

3. **Choose Your App**
   - Select from the top 5 search results
   - The app ID will be automatically extracted

4. **Click Analyze**
   - Wait for the analysis to complete
   - The app will fetch and process reviews in real-time

5. **Review Results**
   - **Sentiment Distribution**: Pie chart showing positive/neutral/negative breakdown
   - **Key Issues**: Top problems identified from negative reviews
   - **Suggestions**: Specific recommendations for app improvement
   - **Full Data**: Complete review table with all details

6. **Export Data**
   - Click "Download CSV" to save results for further analysis

### Example Analysis

```
App: Facebook
Reviews Analyzed: 100
Sentiment Distribution:
  - Positive: 45%
  - Neutral: 25%
  - Negative: 30%

Key Issues:
  - "App crashes frequently on startup"
  - "Too many ads interrupting user experience"
  - "Privacy concerns with data collection"

Suggestions:
  - Optimize loading times and fix reported bugs/crashes
  - Minimize intrusive ads and allow ad-free experience if possible
  - Enhance user data protection and address privacy concerns
```

## 🔧 Configuration

### Custom Model Selection
The application automatically uses:
1. **Primary**: `sshleifer/distilbart-cnn-6-6` (faster, more accurate)
2. **Fallback**: `t5-small` (if primary fails)

Models are cached locally in the `hf_models/` directory.

### Adjust Analysis Parameters

Edit `play_store.py` to customize:

```python
# Line 194-198: Modify review count options
review_options = {
    "Top 50": 50,
    "Top 100": 100,
    "Top 500": 500,
    "Top 1000": 1000  # Add custom option
}

# Line 91-98: Add/modify issue keywords
issue_keywords = {
    "your_category": ["keyword1", "keyword2", "keyword3"],
    # ... more categories
}

# Line 174-185: Customize suggestion generation
# Add new condition blocks for different issue types
```

## 📊 Output Structure

### Sentiment Classification
- **Positive** (Rating ≥ 4): User satisfaction with app
- **Neutral** (Rating = 3): Mixed feelings or average experience
- **Negative** (Rating ≤ 2): User dissatisfaction and complaints

### Issue Categories
- **Interface/UI**: Design, navigation, user experience
- **Performance**: Speed, crashes, bugs, stability
- **Security**: Privacy concerns, data protection
- **Size**: App bloat, storage requirements
- **Ads**: Advertisement placement and frequency
- **Authentication**: Login, signup, verification issues

## 🎯 Use Cases

### For App Developers
- Prioritize bug fixes and feature improvements based on user feedback
- Track sentiment trends over time
- Identify recurring issues affecting user retention

### For Product Managers
- Make data-driven decisions on app roadmap
- Monitor competitive app reviews
- Understand user pain points and expectations

### For QA Teams
- Identify most reported bugs and issues
- Prioritize testing based on user complaints
- Track fix effectiveness through review analysis

### For Marketing Teams
- Understand customer satisfaction levels
- Identify marketing opportunities (positive review highlights)
- Monitor brand perception changes

## 🐛 Troubleshooting

### Issue: "Failed to load NLTK punkt_tab"
**Solution**: The app will fallback to summarization. To fix:
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### Issue: "Failed to load transformer models"
**Solution**: Ensure you have:
- Sufficient storage (~4GB for models)
- Stable internet connection
- PyTorch properly installed
```bash
pip install --upgrade torch transformers
```

### Issue: "Error fetching reviews"
**Possible causes**:
- App doesn't exist or is not available in US region
- Google Play Store rate limiting (try after a few minutes)
- Network connectivity issues

### Issue: Slow Performance
**Solutions**:
- Analyze fewer reviews (start with Top 50)
- Use a machine with more RAM
- Run on GPU if available (update `device=-1` to `device=0` in line 56)

### Issue: Out of Memory Error
**Solutions**:
- Reduce review count
- Close other applications
- Increase virtual memory
- Upgrade system RAM

## 📈 Performance Tips

1. **Batch Analysis**: Analyze 50-100 reviews at a time for optimal speed
2. **Timing**: Run analysis during off-peak hours for faster execution
3. **Model Caching**: First run downloads models; subsequent runs are faster
4. **GPU Acceleration**: If available, modify device parameter for faster processing

## 📝 Project Structure

```
PlayStore-review-analyzer/
├── play_store.py              # Main application file
├── requirements.txt           # Python dependencies
├── runtime.txt               # Python version specification
├── .devcontainer/            # Development container config
├── hf_models/                # Cached transformer models
├── nltk_data/                # NLTK data directory
└── README.md                 # This file
```

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Suggested Improvements
- Add support for multiple languages
- Implement app comparison feature
- Add sentiment trend analysis over time
- Create API endpoint for integration
- Add email notification for critical issues

## 🐛 Known Limitations

1. **Language**: Currently supports English reviews only
2. **Rate Limiting**: Google Play Store may rate-limit requests if too frequent
3. **Regional Availability**: Only analyzes reviews available in US region
4. **Model Size**: Requires significant disk space for ML models
5. **Real-time**: Reviews are cached; real-time updates require app restart

## 🔒 Privacy & Security

- ✅ No user data is stored permanently
- ✅ Reviews are processed locally
- ✅ No external database connections
- ✅ API calls only to official Google Play Store
- ⚠️ Review content cached during session lifetime

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Dreem-R**
- GitHub: [@Dreem-R](https://github.com/Dreem-R)
- Repository: [PlayStore-review-analyzer](https://github.com/Dreem-R/PlayStore-review-analyzer)

## 🙏 Acknowledgments

- [google-play-scraper](https://github.com/JoMingyu/google-play-scraper) - Play Store review scraping
- [Streamlit](https://streamlit.io/) - Interactive web framework
- [Hugging Face Transformers](https://huggingface.co/transformers/) - ML models
- [Plotly](https://plotly.com/) - Interactive visualizations

## 📧 Support & Feedback

For issues, questions, or suggestions:
1. Open an [Issue](https://github.com/Dreem-R/PlayStore-review-analyzer/issues)
2. Check existing issues for similar problems
3. Provide detailed error messages and reproduction steps

## 🚀 Roadmap

- [ ] Multi-language support
- [ ] Historical trend analysis
- [ ] Competitor app comparison
- [ ] REST API for integration
- [ ] Cloud deployment guide
- [ ] Batch processing for multiple apps
- [ ] Custom alert notifications
- [ ] Dashboard with persistent storage

---

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ by Dreem-R