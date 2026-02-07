# Fake News Detector 🛡️

A simple **educational rule-based fake news scanner** built with Streamlit.  
It analyzes news headlines and article content to detect common signs of misinformation and clickbait.

**Important:** This is **not** a real AI/ML model — it uses hand-crafted heuristic rules.  
Use it for learning purposes only — never as your only source for verifying news.

## Features

- Rule-based detection of suspicious patterns
- Highlights clickbait phrases, emotional language, excessive punctuation, ALL CAPS, etc.
- Confidence score with clear categorization: **FAKE** / **SUSPICIOUS** / **REAL**
- Three sample articles to test quickly
- Mock report storage (in-memory)
- Simple dashboard with statistics and result distribution chart
- History of recent checks
- Clean, modern interface with responsive design

## What it detects (signals)

- Clickbait phrases ("You won't believe", "Shocking", "Miracle cure", "Breaking exclusive"...)
- Heavy emotional/outrage language
- Excessive exclamation marks (!!!!!)
- Words in ALL CAPS
- Lack of credible source mentions (Reuters, BBC, AP, NPR, NYT...)
- Very short or unusually long articles

## Screenshots

(You can add screenshots later)

<!-- 
![Main Interface](screenshots/main.png)
![Analysis Result](screenshots/result.png)
![Dashboard](screenshots/dashboard.png)
-->

## Installation

### Prerequisites

- Python 3.8+
- pip

### Step-by-step

1. Clone or download the project

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector"# Fake_news_Detection_app" 
