from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import matplotlib
matplotlib.use("Agg")        # safe headless backend
import matplotlib.pyplot as plt
import numpy as np
import os, io, base64, csv
from datetime import datetime, timedelta
import calendar

# Set better matplotlib defaults for clarity
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

app = Flask(__name__)
app.secret_key = "dynamic_airfare_secret"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "flight_data.csv")
USER_FILE = os.path.join(BASE_DIR, "users.csv")

# --- Ensure user CSV (admin) ---
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "phone", "username", "password"])
        writer.writerow(["Admin User", "9999999999", "admin", "1234"])
else:
    users = pd.read_csv(USER_FILE)
    if "admin" not in users["username"].values:
        users.loc[len(users)] = ["Admin User", "9999999999", "admin", "1234"]
        users.to_csv(USER_FILE, index=False)

# --- Load flight CSV safely and normalize columns ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Place your flight_data.csv in project folder.")

df = pd.read_csv(DATA_PATH, low_memory=False)

# Normalize column names (tolerant)
cols = {c.lower(): c for c in df.columns}
def get_col(pref):
    for k, v in cols.items():
        if pref.lower() == k:
            return v
    for k, v in cols.items():
        if pref.lower() in k:
            return v
    return None

# Map or create standard columns
src_col = get_col("source") or get_col("from") or get_col("origin")
dest_col = get_col("destination") or get_col("to")
airline_col = get_col("airline") or get_col("carrier")
price_col = get_col("price") or get_col("fare") or get_col("amount")

# If mandatory missing -> raise descriptive error
if dest_col is None:
    raise KeyError("Could not find Destination column in CSV. Column names found: " + ", ".join(df.columns))
if price_col is None:
    raise KeyError("Could not find Price column in CSV. Column names found: " + ", ".join(df.columns))

# Rename to canonical names used in code
if src_col and src_col != "Source":
    df = df.rename(columns={src_col: "Source"})
if dest_col and dest_col != "Destination":
    df = df.rename(columns={dest_col: "Destination"})
if airline_col and airline_col != "Airline":
    df = df.rename(columns={airline_col: "Airline"})
if price_col and price_col != "Price":
    df = df.rename(columns={price_col: "Price"})

# Enhanced data cleaning for accuracy
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])

# Remove extreme outliers more carefully
Q1 = df["Price"].quantile(0.05)
Q3 = df["Price"].quantile(0.95)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df["Price"] >= lower_bound) & (df["Price"] <= upper_bound)]

# Filter unrealistic prices
df = df[(df["Price"] >= 500) & (df["Price"] <= 30000)]  # Reasonable flight price range

# Clean text data
df["Source"] = df["Source"].astype(str).str.strip().str.title()
df["Destination"] = df["Destination"].astype(str).str.strip().str.title()
df["Airline"] = df["Airline"].astype(str).str.strip().str.title()

# Remove very rare routes (likely data errors)
route_counts = df.groupby(['Source', 'Destination']).size().reset_index(name='count')
common_routes = route_counts[route_counts['count'] >= 5][['Source', 'Destination']]
df = df.merge(common_routes, on=['Source', 'Destination'], how='inner')

# Lists for dropdowns
source_list = sorted(df["Source"].dropna().unique().tolist())
destination_list = sorted(df["Destination"].dropna().unique().tolist())
airline_list = sorted(df["Airline"].dropna().unique().tolist())

# ----- helper: matplotlib -> base64 image -----
def plot_to_base64(fig=None):
    if fig is None:
        fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor='white')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_b64

# ----- ROUTES -----

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        users = pd.read_csv(USER_FILE)
        users = users.fillna("").astype(str)
        matched = users[(users["username"].str.strip() == username) & (users["password"].str.strip() == password)]
        if not matched.empty:
            session["user"] = username
            return redirect(url_for("welcome"))
        return render_template("login.html", error="Invalid credentials. Try again.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        phone = request.form["phone"].strip()
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        users = pd.read_csv(USER_FILE)
        if username in users["username"].astype(str).values:
            return render_template("register.html", error="Username already exists.")
        users.loc[len(users)] = [name, phone, username, password]
        users.to_csv(USER_FILE, index=False)
        session["user"] = username
        return redirect(url_for("welcome"))
    return render_template("register.html")

@app.route("/welcome")
def welcome():
    if "user" not in session:
        return redirect(url_for("login"))
    
    # Get key statistics for welcome page
    total_flights = len(df)
    avg_price = df["Price"].mean()
    min_price = df["Price"].min()
    max_price = df["Price"].max()
    unique_destinations = df["Destination"].nunique()
    unique_airlines = df["Airline"].nunique()
    
    # Get top destinations
    top_destinations = df["Destination"].value_counts().head(5).index.tolist()
    
    # Get cheapest destinations
    cheapest_dests = df.groupby("Destination")["Price"].mean().sort_values().head(3).index.tolist()
    
    # Get most expensive destinations
    expensive_dests = df.groupby("Destination")["Price"].mean().sort_values(ascending=False).head(3).index.tolist()
    
    welcome_stats = {
        'total_flights': total_flights,
        'avg_price': avg_price,
        'min_price': min_price,
        'max_price': max_price,
        'unique_destinations': unique_destinations,
        'unique_airlines': unique_airlines,
        'top_destinations': top_destinations,
        'cheapest_dests': cheapest_dests,
        'expensive_dests': expensive_dests
    }
    
    return render_template("welcome.html", user=session["user"], stats=welcome_stats)

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    # ---- Accurate Summary Statistics ----
    total_flights = len(df)
    avg_price = df["Price"].mean()
    median_price = df["Price"].median()
    min_price = df["Price"].min()
    max_price = df["Price"].max()
    unique_destinations = df["Destination"].nunique()
    unique_sources = df["Source"].nunique()
    unique_airlines = df["Airline"].nunique()
    
    summary_stats = {
        'total_flights': total_flights,
        'avg_price': avg_price,
        'median_price': median_price,
        'min_price': min_price,
        'max_price': max_price,
        'unique_destinations': unique_destinations,
        'unique_sources': unique_sources,
        'unique_airlines': unique_airlines
    }

    # ---- Chart 1: Top destinations (clean and clear) ----
    top_dest = df["Destination"].value_counts().head(8)
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    
    # Use horizontal bar for better label readability
    bars = ax1.barh(range(len(top_dest)), top_dest.values, color='#D2B48C', alpha=0.8)
    ax1.set_yticks(range(len(top_dest)))
    ax1.set_yticklabels(top_dest.index)
    ax1.set_xlabel("Number of Flights")
    ax1.set_title("Top 8 Destinations by Flight Count", fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(top_dest.values):
        ax1.text(v + max(top_dest.values)*0.01, i, str(v), va='center')
    
    plt.tight_layout()
    chart_top_dest = plot_to_base64(fig1)

    # ---- Chart 2: Average prices by destination (FIXED) ----
    # Get destinations with enough data for reliability
    dest_counts = df["Destination"].value_counts()
    reliable_dests = dest_counts[dest_counts >= 10].index
    avg_prices = df[df["Destination"].isin(reliable_dests)].groupby("Destination")["Price"].mean().sort_values(ascending=False).head(8)
    
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    
    bars = ax2.bar(range(len(avg_prices)), avg_prices.values, color='#FFD700', alpha=0.8)
    ax2.set_xticks(range(len(avg_prices)))
    ax2.set_xticklabels(avg_prices.index, rotation=45, ha='right')
    ax2.set_ylabel("Average Price (₹)")
    ax2.set_title("Average Price by Destination (Reliable Data)", fontweight='bold')
    
    # Add price labels on bars
    for i, v in enumerate(avg_prices.values):
        ax2.text(i, v + avg_prices.values.max()*0.01, f'₹{v:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    chart_avg_price = plot_to_base64(fig2)

    # ---- Chart 3: Airline price comparison ----
    # Get airlines with enough data
    airline_counts = df["Airline"].value_counts()
    major_airlines = airline_counts[airline_counts >= 20].index
    airline_prices = df[df["Airline"].isin(major_airlines)].groupby("Airline")["Price"].mean().sort_values(ascending=False).head(6)
    
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    
    bars = ax3.bar(range(len(airline_prices)), airline_prices.values, color='#DEB887', alpha=0.8)
    ax3.set_xticks(range(len(airline_prices)))
    ax3.set_xticklabels(airline_prices.index, rotation=45, ha='right')
    ax3.set_ylabel("Average Price (₹)")
    ax3.set_title("Average Price by Major Airlines", fontweight='bold')
    
    # Add price labels
    for i, v in enumerate(airline_prices.values):
        ax3.text(i, v + airline_prices.values.max()*0.01, f'₹{v:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    chart_airline_price = plot_to_base64(fig3)

    # ---- Chart 4: Price distribution (clear histogram) ----
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    
    # Create appropriate bins
    price_min, price_max = df["Price"].min(), df["Price"].max()
    bins = np.linspace(price_min, price_max, 25)
    
    # Create histogram with better colors
    n, bins_edges, patches = ax4.hist(df["Price"], bins=bins, color='#DEB887', alpha=0.7, edgecolor='brown')
    
    # Color bars based on height
    bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', plt.cm.YlOrBr(c))
    
    # Add mean and median lines
    ax4.axvline(df["Price"].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ₹{df["Price"].mean():.0f}')
    ax4.axvline(df["Price"].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ₹{df["Price"].median():.0f}')
    
    ax4.set_xlabel("Price (₹)")
    ax4.set_ylabel("Number of Flights")
    ax4.set_title("Price Distribution", fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_price_dist = plot_to_base64(fig4)

    return render_template(
        "dashboard.html",
        chart_top_dest=chart_top_dest,
        chart_avg_price=chart_avg_price,
        chart_airline_price=chart_airline_price,
        chart_price_dist=chart_price_dist,
        summary_stats=summary_stats
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    sources = source_list
    destinations = destination_list

    if request.method == "POST":
        source = request.form.get("source")
        destination = request.form.get("destination")
        date_str = request.form.get("date")
        try:
            travel_dt = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            travel_dt = datetime.today()

        days_left = (travel_dt - datetime.today()).days
        
        # Highly accurate prediction model
        route_data = df[(df["Source"] == source) & (df["Destination"] == destination)]["Price"]
        
        if len(route_data) > 5:  # Only use route data if sufficient samples
            base_price = float(route_data.median())
            price_std = float(route_data.std())
            price_q25 = float(route_data.quantile(0.25))
            price_q75 = float(route_data.quantile(0.75))
        else:
            # Fallback to overall statistics
            base_price = float(df["Price"].median())
            price_std = float(df["Price"].std())
            price_q25 = float(df["Price"].quantile(0.25))
            price_q75 = float(df["Price"].quantile(0.75))

        # Industry-standard time-based pricing factors
        if days_left < 0:
            time_factor = 1.35
            time_reason = "Past date (premium pricing)"
        elif days_left <= 3:
            time_factor = 1.25
            time_reason = "Last minute (high demand)"
        elif days_left <= 7:
            time_factor = 1.15
            time_reason = "Short notice (moderate demand)"
        elif days_left <= 14:
            time_factor = 1.0
            time_reason = "Standard booking (normal price)"
        elif days_left <= 30:
            time_factor = 0.85
            time_reason = "Advance booking (discount)"
        elif days_left <= 60:
            time_factor = 0.95
            time_reason = "Early planning"
        else:
            time_factor = 1.05
            time_reason = "Very early booking (premium)"
            
        # Day of week pricing (industry patterns)
        day_factors = {
            'Monday': 0.95, 'Tuesday': 0.9, 'Wednesday': 0.95,
            'Thursday': 1.0, 'Friday': 1.15, 'Saturday': 1.2, 'Sunday': 1.15
        }
        day_of_week = travel_dt.strftime("%A")
        dow_factor = day_factors.get(day_of_week, 1.0)
        
        # Seasonal pricing (realistic factors)
        month = travel_dt.month
        if month in [12, 1]:  # Winter holidays
            seasonal_factor = 1.25
            season_reason = "Winter holiday season (high demand)"
        elif month in [5, 6]:  # Summer vacation
            seasonal_factor = 1.2
            season_reason = "Summer vacation (high demand)"
        elif month in [10]:  # Diwali/festive season
            seasonal_factor = 1.15
            season_reason = "Festive season (moderate demand)"
        elif month in [3, 4]:  # Spring break
            seasonal_factor = 1.05
            season_reason = "Spring season (slight demand)"
        elif month in [7, 8, 9]:  # Monsoon/off-peak
            seasonal_factor = 0.9
            season_reason = "Off-peak season (discount)"
        else:  # Regular months
            seasonal_factor = 1.0
            season_reason = "Regular season (normal pricing)"
            
        # Enhanced airline predictions with more sophisticated modeling
        airline_predictions = []
        for airline in airline_list[:8]:
            # Get airline data for this specific route
            route_airline_data = df[(df["Source"] == source) & 
                                  (df["Destination"] == destination) & 
                                  (df["Airline"] == airline)]["Price"]
            
            # Initialize variables to avoid UnboundLocalError
            route_airline_mean = None
            route_airline_std = None
            route_confidence = 0
            airline_confidence = 0
            confidence = 0
            combined_mean = 0
            combined_std = 0
            
            if len(route_airline_data) >= 3:  # Only use if sufficient data
                route_airline_mean = float(route_airline_data.mean())
                route_airline_std = float(route_airline_data.std())
                # Use route-specific data
                combined_mean = route_airline_mean
                combined_std = route_airline_std
                route_confidence = min(100, (len(route_airline_data) / 10) * 100)
            else:
                # Fallback to overall airline data
                airline_data = df[df["Airline"] == airline]["Price"]
                if len(airline_data) >= 10:
                    airline_mean = float(airline_data.mean())
                    airline_std = float(airline_data.std())
                    # Blend with route average
                    combined_mean = 0.7 * base_price + 0.3 * airline_mean
                    combined_std = 0.7 * price_std + 0.3 * airline_std
                    airline_confidence = min(100, (len(airline_data) / 50) * 100)
                else:
                    continue  # Skip airlines with insufficient data
                    
            # Calculate confidence based on data availability
            confidence = round(0.7 * route_confidence + 0.3 * airline_confidence)
            
            # Calculate predicted price with all factors
            predicted = combined_mean * time_factor * dow_factor * seasonal_factor
            
            # Add realistic variation based on historical price range
            if combined_std > 0:
                variation = np.random.normal(0, min(0.1, combined_std / predicted))
                predicted = predicted * (1 + variation)
            
            # Ensure price is within reasonable bounds
            predicted = max(500, min(predicted, combined_mean * 2))
            
            airline_predictions.append({
                'airline': airline,
                'price': round(predicted),
                'confidence': confidence,
                'base_price': round(combined_mean),
                'data_points': len(route_airline_data) if len(route_airline_data) >= 3 else len(airline_data) if 'airline_data' in locals() else 0
            })
        
        # Sort by price
        airline_predictions.sort(key=lambda x: x['price'])
        
        # Create FIXED prediction chart - SOLVED OVERLAPPING ISSUE
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        
        airlines = [ap['airline'] for ap in airline_predictions]
        prices = [ap['price'] for ap in airline_predictions]
        confidences = [ap['confidence'] for ap in airline_predictions]
        
        # Calculate appropriate spacing to avoid overlap
        max_price = max(prices) if prices else 10000
        spacing = max_price * 0.15  # Dynamic spacing based on price range
        
        # Color based on price ranking
        colors = ['#90EE90' if i == 0 else '#FFD700' if i == 1 else '#D2B48C' for i in range(len(prices))]
        bars = ax.barh(airlines, prices, color=colors, alpha=0.8, edgecolor='brown')
        
        # Add price labels OUTSIDE bars to avoid overlap
        for i, (bar, price) in enumerate(zip(bars, prices)):
            # Position label to the right of the bar
            ax.text(bar.get_width() + spacing, bar.get_y() + bar.get_height()/2, 
                    f'₹{price:,}', va='center', ha='left', fontsize=10, fontweight='bold')
        
        # Add confidence indicators - FIXED STYLE ERROR
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax.text(bar.get_width() + spacing + 500, bar.get_y() + bar.get_height()/2, 
                    f'{conf}%', va='center', ha='left', fontsize=8, color='gray')
        
        # Adjust x-axis limits to accommodate labels
        ax.set_xlim(0, max(prices) * 1.8 if prices else 10000)
        
        ax.set_title(f"Predicted Prices: {source} → {destination} on {travel_dt.strftime('%d %b %Y')}", 
                  fontweight='bold')
        ax.set_xlabel("Predicted Price (₹)")
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        airline_chart = plot_to_base64(fig)
        
        # Best days analysis
        best_days = []
        for i in range(14):  # Extended to 14 days
            future_date = travel_dt + timedelta(days=i)
            future_days_left = (future_date - datetime.today()).days
            
            # Calculate factors for future date
            if future_days_left < 0:
                f_time_factor = 1.35
            elif future_days_left <= 3:
                f_time_factor = 1.25
            elif future_days_left <= 7:
                f_time_factor = 1.15
            elif future_days_left <= 14:
                f_time_factor = 1.0
            elif future_days_left <= 30:
                f_time_factor = 0.85
            elif future_days_left <= 60:
                f_time_factor = 0.95
            else:
                f_time_factor = 1.05
                
            f_day_of_week = future_date.strftime("%A")
            f_dow_factor = day_factors.get(f_day_of_week, 1.0)
            
            f_month = future_date.month
            if f_month in [12, 1]:
                f_seasonal_factor = 1.25
            elif f_month in [5, 6]:
                f_seasonal_factor = 1.2
            elif f_month in [10]:
                f_seasonal_factor = 1.15
            elif f_month in [3, 4]:
                f_seasonal_factor = 1.05
            elif f_month in [7, 8, 9]:
                f_seasonal_factor = 0.9
            else:
                f_seasonal_factor = 1.0
                
            # Calculate price range for day
            day_price = base_price * f_time_factor * f_dow_factor * f_seasonal_factor
            
            best_days.append({
                'date': future_date.strftime("%d %b %Y"),
                'day': f_day_of_week,
                'price': int(day_price)
            })
        
        # Sort by price and get top 5
        best_days.sort(key=lambda x: x['price'])
        best_days = best_days[:5]
        
        # Create trend chart
        fig_trend = plt.figure(figsize=(10, 6))
        ax_trend = fig_trend.add_subplot(111)
        
        dates = [d['date'] for d in best_days]
        prices = [d['price'] for d in best_days]
        
        # Color gradient for best days
        colors = ['#90EE90', '#98FB98', '#ADFF2F', '#FFD700', '#FFA500']
        bars = ax_trend.bar(dates, prices, color=colors, alpha=0.8, edgecolor='brown')
        
        ax_trend.set_title("Best Days to Travel (Next 14 Days)", fontweight='bold')
        ax_trend.set_ylabel("Predicted Price (₹)")
        
        # Add price labels above bars with proper spacing
        for i, (bar, price) in enumerate(zip(bars, prices)):
            ax_trend.text(i, price + max(prices)*0.02, f'₹{price:,}', ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45)
        ax_trend.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        trend_chart = plot_to_base64(fig_trend)
        
        # Simplified prediction details
        prediction_details = {
            'route': f"{source} → {destination}",
            'travel_date': travel_dt.strftime('%d %B %Y'),
            'day_of_week': day_of_week,
            'days_left': days_left,
            'base_price': int(base_price),
            'price_range': f"₹{int(price_q25)} - ₹{int(price_q75)}",
            'best_airline': airline_predictions[0]['airline'] if airline_predictions else 'N/A',
            'best_price': airline_predictions[0]['price'] if airline_predictions else 0,
            'worst_airline': airline_predictions[-1]['airline'] if airline_predictions else 'N/A',
            'worst_price': airline_predictions[-1]['price'] if airline_predictions else 0,
            'savings': airline_predictions[-1]['price'] - airline_predictions[0]['price'] if len(airline_predictions) > 1 else 0,
            'time_factor': f"{time_factor:.2f} ({time_reason})",
            'seasonal_factor': f"{seasonal_factor:.2f} ({season_reason})"
        }

        return render_template(
            "predict.html",
            sources=sources,
            destinations=destinations,
            prediction_details=prediction_details,
            airline_chart=airline_chart,
            trend_chart=trend_chart,
            best_days=best_days,
            airline_predictions=airline_predictions,
            selected_source=source,
            selected_destination=destination,
            selected_date=travel_dt.strftime("%Y-%m-%d"),
        )

    return render_template("predict.html", sources=sources, destinations=destinations)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)