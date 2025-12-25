# ✈️ Dynamic Airfare Prediction System

_A machine learning-powered web application that predicts flight prices based on routes, travel dates, and seasonal patterns using Python, Flask, and Matplotlib._

---

## 📌 Table of Contents
- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Tools & Technologies](#tools--technologies)
- [Project Structure](#project-structure)
- [Data Cleaning & Preparation](#data-cleaning--preparation)
- [Prediction Methodology](#prediction-methodology)
- [Key Features](#key-features)
- [Screenshots](#screenshots)
- [How to Run This Project](#how-to-run-this-project)
- [Future Enhancements](#future-enhancements)
- [Author & Contact](#author--contact)

---

## Overview

This project is a full-stack web application that helps travelers make informed decisions by predicting flight prices based on multiple factors. A complete data pipeline was built using Python for backend processing, Pandas for data analysis, Matplotlib for visualization, and Flask for the web interface.

---

## Business Problem

Travelers face significant challenges when booking flights:
- Airfare prices fluctuate frequently, making booking decisions difficult
- Unclear timing for when to book for best prices
- Hard to compare prices across multiple airlines simultaneously
- Prices vary significantly by season, day of week, and booking window

This project aims to:
- Predict accurate flight prices based on historical data
- Identify optimal travel dates for cost savings
- Compare prices across all major airlines
- Factor in seasonal patterns, holidays, and demand cycles

---

## Dataset

- Flight data CSV located in project root (`flight_data.csv`)
- User credentials stored in auto-generated `users.csv`
- Data includes: Source, Destination, Airline, Price

---

## Tools & Technologies

- Python (Flask, Pandas, NumPy, Matplotlib)
- HTML/CSS (Jinja2 Templates)
- Session-based Authentication
- CSV-based Data Storage
- GitHub

---

## Project Structure
dynamic-airfare-prediction/
│
├── README.md
├── .gitignore
├── requirements.txt
├── LICENSE
│
├── app.py # Main Flask application
├── flight_data.csv # Flight dataset
├── users.csv # User data (auto-generated)
│
├── templates/ # HTML templates
│ ├── login.html
│ ├── register.html
│ ├── welcome.html
│ ├── dashboard.html
│ └── predict.html
│
├── static/ # Static assets
│ └── css/
│ └── style.css
│
└── images/ # Screenshots
├── dashboard.png
└── prediction.png

---

## Data Cleaning & Preparation

- Removed transactions with:
  - Price ≤ ₹500 or Price > ₹30,000 (unrealistic values)
  - Outliers using IQR method (5th-95th percentile)
  - Routes with fewer than 5 flights (insufficient data)
- Normalized text data (Source, Destination, Airline)
- Converted price to numeric and handled missing values
- Created dropdown lists for user-friendly selection

---

## Prediction Methodology

**Base Price Calculation:**
- Uses median price from historical route data
- Falls back to overall median if route data insufficient

**Time-Based Pricing Factors:**
| Days Before Travel | Factor | Reason |
|-------------------|--------|--------|
| < 0 (Past) | 1.35× | Premium pricing |
| 0-3 days | 1.25× | Last minute booking |
| 4-7 days | 1.15× | Short notice |
| 8-14 days | 1.00× | Standard booking |
| 15-30 days | 0.85× | Advance discount |
| 31-60 days | 0.95× | Early planning |
| > 60 days | 1.05× | Very early booking |

**Day of Week Factors:**
| Day | Factor |
|-----|--------|
| Tuesday | 0.90× (Cheapest) |
| Monday/Wednesday | 0.95× |
| Thursday | 1.00× |
| Friday/Sunday | 1.15× |
| Saturday | 1.20× (Most Expensive) |

**Seasonal Factors:**
| Season | Months | Factor |
|--------|--------|--------|
| Winter Holidays | Dec-Jan | 1.25× |
| Summer Vacation | May-Jun | 1.20× |
| Festive Season | Oct | 1.15× |
| Monsoon (Off-peak) | Jul-Sep | 0.90× |

---

## Key Features

1. **User Authentication**: Secure login and registration system with session management
2. **Interactive Dashboard**: Visual analytics showing top destinations, airline pricing, and price distributions
3. **Price Prediction**: Route + date based forecasting with confidence scores
4. **Airline Comparison**: Side-by-side price comparison across all major carriers
5. **Best Days Analysis**: 14-day forecast identifying optimal travel dates
6. **Dynamic Charts**: Real-time Matplotlib visualizations converted to base64 images

---

## Screenshots

**Login Page:**

![Login](images/login.png)

**Welcome Page:**

![Welcome](images/welcome.png)

**Dashboard:**

![Dashboard](images/dashboard.png)

**Price Prediction:**

![Prediction](images/prediction.png)

## How to Run This Project

1. Clone the repository:
```bash
git clone https://github.com/akshat7081/dynamic-airfare-prediction.git
2. Navigate to project:
cd dynamic-airfare-prediction
3. Install dependencies:
pip install -r requirements.txt
4. Run the application:
python app.py
5. Open browser:
http://127.0.0.1:5000

Default Login: Username: admin | Password: 1234

Future Enhancements
- Integrate real-time flight APIs
- Mobile-responsive design
- Fare calendar view
- Add email price alerts

Author & Contact
Akshat Tripathi
Python Developer | Data Enthusiast
📧 Email: akshat3478@gmail.com
www.linkedin.com/in/
akshattripathi7081
Vanity URL name


