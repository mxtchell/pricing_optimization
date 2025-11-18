# Pricing Optimization Skill - Technical Documentation

## Executive Summary

The Pricing Optimization Skill is an AI-powered analytics tool designed to analyze pricing strategies, identify revenue opportunities, and provide data-driven pricing recommendations for consumer packaged goods (CPG). The skill performs competitive benchmarking, elasticity analysis, and revenue impact simulations across multiple product dimensions (brands, segments, pack sizes, sub-categories).

**Primary Use Cases:**
- Competitive pricing benchmarking (brand vs. competition)
- Price-volume tradeoff analysis
- Revenue optimization opportunity identification
- What-if scenario planning for price changes
- Price elasticity measurement

The skill generates interactive visualizations, strategic insights via LLM analysis, and quantifiable metrics to support pricing decisions. It's currently configured for pasta category data but can be adapted to other CPG categories.

---

## Core Capabilities

### 1. **Competitive Comparison Analysis**
Compares a target brand (e.g., BARILLA) against competitive averages across any dimension.

**What it does:**
- Side-by-side grouped column charts showing brand price vs. competition average
- Color-coded bars: Green = premium positioned (above competition), Red = underpriced (below competition)
- Price gap labels on each bar showing both dollar difference and percentage premium/discount
- Price per ounce normalization chart (always by base_size) for apples-to-apples comparison across pack sizes
- Top 5 pricing opportunities table highlighting most underpriced items
- Strategic positioning insights generated via LLM

**Key Metrics Displayed:**
- **Average Premium %**: Unit-weighted average of price premium vs. competition across all dimension values
- **Volume Share %**: Target brand's unit volume as % of total market
- **Price Leaders**: Count of dimension values where target brand is priced above competition
- **Base Sizes Analyzed**: Number of SKU-level items in comparison

**Calculations:**
```python
# Price Premium %
price_premium_pct = ((target_price / competition_avg_price) - 1) * 100

# Unit-Weighted Average Premium
weighted_premium = sum(price_premium_pct * units) / sum(units)

# Volume Share
volume_share = (target_brand_units / total_market_units) * 100

# Price Per Oz (for normalization)
price_per_oz = avg_price / base_size_oz
```

**Requirements:**
- Must filter to a single brand for comparison
- Requires at least 2 dimension values to compare (e.g., multiple pack sizes or segments)

---

### 2. **Price Comparison Analysis**
General market-level pricing analysis without competitive filtering.

**What it does:**
- Horizontal bar chart showing average price by dimension
- Sorted by price (low to high)
- LLM-generated narrative summarizing price spread and market structure

**Key Metrics:**
- **Market Average Price**: Overall weighted average across all items
- **Price Range**: Min to max price across dimension values
- **Items Analyzed**: Count of unique dimension values

**Calculations:**
```python
# Average price per dimension value
avg_price = total_sales / total_units

# Market average (across all dimension values)
market_avg = sum(total_sales) / sum(total_units)

# Price vs. market average
price_vs_avg_pct = ((avg_price / market_avg) - 1) * 100
```

---

### 3. **Price Elasticity Calculation**
Measures demand sensitivity to price changes using time-series analysis.

**What it does:**
- Calculates price elasticity coefficient for each dimension value
- Classifies items as Elastic (|E| > 1), Moderately Elastic (0.5 < |E| ≤ 1), or Inelastic (|E| ≤ 0.5)
- Provides strategic guidance on pricing power

**Calculation Method:**
```python
# Calculate period-over-period changes
price_pct_change = (price_t - price_t-1) / price_t-1 * 100
units_pct_change = (units_t - units_t-1) / units_t-1 * 100

# Elasticity = % change in quantity / % change in price
elasticity = mean(units_pct_change / price_pct_change)
```

**Interpretation:**
- **Elastic (|E| > 1)**: Price-sensitive segment. Price increases will reduce revenue.
- **Moderately Elastic (0.5 < |E| ≤ 1)**: Moderate price sensitivity. Small price increases may be viable.
- **Inelastic (|E| ≤ 0.5)**: Price insensitive. Strong candidates for price increases.

**Limitations:**
- Requires sufficient price variation over time (minimum 3 data points per dimension value)
- Filters out noise by excluding price changes < 0.1%
- Assumes other factors (marketing, distribution, seasonality) remain constant
- Uses simple percentage change method, not regression-based

---

### 4. **Optimization Opportunities**
Identifies specific items with revenue upside potential.

**What it does:**
- Analyzes price-per-volume distribution to identify underpriced items
- Highlights items priced in bottom quartile (P25) with high sales volume
- Calculates potential revenue lift from raising prices to P25 benchmark
- Generates table with top 10 opportunities ranked by revenue impact

**Calculation Logic:**
```python
# Normalize prices by volume (handles different pack sizes)
price_per_volume = total_sales / total_volume

# Calculate quartiles
p25 = quantile(price_per_volume, 0.25)
p75 = quantile(price_per_volume, 0.75)
median = quantile(price_per_volume, 0.50)

# Identify opportunities: Low price AND high volume
opportunity = 'Price Increase Potential' if:
    - price_per_volume < p25 AND
    - total_units > median(total_units)

# Revenue lift: Conservative estimate to bring to P25
potential_lift = (p25 - price_per_volume) * total_volume
```

**Key Metrics:**
- **Potential Lift**: Total revenue gain from bringing all opportunities to P25 price level
- **Market Median**: Benchmark price per volume
- **Opportunities**: Count of items flagged for price increases

**Conservative Assumptions:**
- Uses P25 (not P50 or P75) as target to be conservative
- Does not account for elasticity (assumes no volume loss)
- Focuses on high-volume items to maximize impact while minimizing risk

---

### 5. **What-If Scenario Analysis**
Simulates revenue impact of hypothetical price changes.

**What it does:**
- Models revenue outcome of uniform price increase/decrease across portfolio
- Accounts for volume loss using assumed elasticity
- Shows before/after revenue comparison by dimension

**Calculation:**
```python
# Assumed elasticity for CPG (industry typical)
assumed_elasticity = -0.7

# New price after % change
new_price = current_price * (1 + price_change_pct / 100)

# Volume impact (elasticity effect)
volume_change_pct = assumed_elasticity * price_change_pct
new_units = current_units * (1 + volume_change_pct / 100)

# Projected revenue
projected_revenue = new_price * new_units
revenue_impact = projected_revenue - current_revenue
```

**Example:**
- 10% price increase → -7% volume change (elasticity -0.7) → +2.3% net revenue gain
- 5% price decrease → +3.5% volume change → -1.65% net revenue loss

**Limitations:**
- Uses fixed elasticity of -0.7 (not item-specific)
- Assumes elasticity remains constant across price ranges
- Does not model competitive reactions or cross-elasticity effects

---

## Data Requirements

**Required Columns:**
- `brand`: Brand name (text)
- `segment`: Product segment (e.g., PREMIUM, VALUE)
- `sub_category`: Product sub-category (e.g., SPAGHETTI, PENNE)
- `base_size`: Pack size with unit (e.g., "16 OUNCE", "32 OUNCE")
- `sales`: Total dollar sales (numeric)
- `units`: Total units sold (numeric)
- `volume`: Total volume sold in consistent units like ounces (numeric)
- `month_new`: Time period (date)

**Pre-Processing:**
The skill automatically aggregates raw transaction data by:
```python
total_sales = SUM(sales)
total_units = SUM(units)
total_volume = SUM(volume)
avg_price = total_sales / total_units
```

**Minimum Data Requirements:**
- Competitive comparison: At least 2 brands (target + competitors)
- Price comparison: At least 1 dimension value
- Elasticity: Minimum 3 time periods with price variation > 0.1%
- Optimization: At least 2 items to establish quartile benchmarks

---

## Technical Limitations & Assumptions

### **Data Limitations:**
1. **No cross-elasticity modeling**: Assumes price changes don't affect demand for other items in portfolio
2. **Time granularity**: Requires monthly or finer time periods for elasticity calculations
3. **Outlier sensitivity**: Extreme price or volume changes can skew elasticity calculations
4. **Missing base_size data**: Price-per-oz normalization fails if base_size format is non-standard (requires "XX OUNCE" format)

### **Analytical Assumptions:**
1. **Ceteris paribus**: All analysis assumes other factors (distribution, marketing, competition) remain constant
2. **Fixed elasticity in what-if**: Uses industry average -0.7 elasticity rather than item-specific values
3. **Linear elasticity**: Assumes constant elasticity across price ranges (may not hold for large changes)
4. **Revenue optimization only**: Does not consider profit margins, inventory costs, or strategic positioning goals

### **Competitive Comparison Specifics:**
1. **Top 15 base sizes only**: Price per oz chart limited to top 15 BARILLA base sizes by sales volume to avoid clutter
2. **Inner join logic**: Only shows base sizes where target brand competes (has sales volume)
3. **Competition average**: Treats all non-target brands equally (no weighting by market share)
4. **Color coding logic**: Green if premium > 0%, red if premium ≤ 0% (no neutral zone)

### **Known Edge Cases:**
1. **Single brand in dataset**: Competitive comparison fails gracefully with error message
2. **No time variation**: Elasticity returns "Insufficient data" message
3. **Insufficient dimension values**: Optimization redirects to price comparison view
4. **Non-numeric base_size**: Price-per-oz calculation skips items with formats like "VARIETY PACK"

---

## Visualization Details

### **Competitive Comparison Layout:**
1. **Main Chart**: Grouped column chart (brand vs. competition) with color-coded bars and price gap labels
2. **Price Per Oz Chart**: Always shows base_size dimension for normalization (even if analyzing by segment)
3. **Opportunities Table**: Top 5 underpriced items in grid layout with metrics
4. **Insights Section**: LLM-generated strategic analysis (250 word max)
5. **Parameter Pills**: Brand, Time Period, Dimension, SKU count, Avg Premium %, and any user-applied filters

### **Chart Rendering:**
- Uses Highcharts via AnswerRocket's structured layout framework
- All layouts built with Document → FlexContainer → HighchartsChart hierarchy
- Parent-child relationships managed via `parentId` references
- Wire layout converts JSON to HTML

---

## Filter Pills Logic

When users apply additional filters (e.g., "Compare BARILLA by base_size for LONG CUT segment"), the skill displays filter pills:

```python
# Example filter structure
filters = [
    {'dim': 'brand', 'val': ['BARILLA']},  # Excluded from pills (shown separately)
    {'dim': 'segment', 'val': ['LONG CUT', 'SHORT CUT']}  # Shown as pill
]

# Pill display format
"Segment: LONG CUT, SHORT CUT"
```

**Applied to:**
- Competitive comparison analysis
- Regular price comparison
- All analysis types except elasticity (which uses raw time-series data)

---

## Example Queries

**Competitive Analysis:**
- "Compare BARILLA pricing to competitors by base size in 2024"
- "How does BARILLA price vs competition across segments?"
- "Show me BARILLA's price positioning for LONG CUT pasta"

**Optimization:**
- "What are the pricing optimization opportunities for BARILLA?"
- "Which BARILLA products are underpriced?"
- "Where can BARILLA raise prices?"

**Elasticity:**
- "Calculate price elasticity for BARILLA by segment"
- "How price sensitive is premium pasta?"
- "Show elasticity across brands"

**What-If:**
- "What happens if BARILLA raises prices 10%?"
- "Model 5% price decrease impact on revenue"
- "Simulate price increase across segments"

---

## Future Enhancement Opportunities

1. **Dynamic elasticity**: Calculate item-specific elasticity from data instead of assuming -0.7
2. **Profit optimization**: Incorporate margin data to optimize profit, not just revenue
3. **Competitive response modeling**: Simulate how competitors might react to price changes
4. **Promotional impact**: Separate base price from promotional pricing effects
5. **Geographic variation**: Allow regional pricing comparisons and recommendations
6. **Confidence intervals**: Add statistical confidence bounds to elasticity estimates
7. **Multi-brand optimization**: Recommend portfolio-wide pricing strategies, not just single brand
8. **Price architecture**: Ensure recommended prices maintain logical good-better-best structure

---

## Technical Stack

- **Language**: Python 3.10+
- **Framework**: AnswerRocket Skill Framework
- **Data Access**: AnswerRocketClient (DuckDB backend with CSV files)
- **Visualization**: Highcharts via structured JSON layouts
- **AI**: ArUtils for LLM-generated insights (OpenAI backend)
- **Analytics**: Pandas, NumPy for data manipulation

---

## Maintenance Notes

**Database ID**: Configured via `DATABASE_ID` environment variable (currently: `83c2268f-af77-4d00-8a6b-7181dc06643e`)

**CSV File**: Expects `pasta_2025.csv` with columns: brand, segment, sub_category, base_size, month_new, sales, units, volume

**Debugging**: All functions include `print()` debug statements for troubleshooting query execution and data flow

**Version Control**: Tracked in Git at https://github.com/mxtchell/pricing_optimization.git
