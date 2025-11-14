"""
Pricing Optimization Skill for Pasta Dataset
Analyzes pricing strategies, calculates price elasticity, and identifies optimization opportunities
"""
from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List

from skill_framework import skill, SkillParameter, SkillInput, SkillOutput, SkillVisualization
from skill_framework.layouts import wire_layout
from answer_rocket import AnswerRocketClient
from ar_analytics import ArUtils

# Database ID for pasta dataset
DATABASE_ID = os.getenv('DATABASE_ID', '83c2268f-af77-4d00-8a6b-7181dc06643e')


@skill(
    name="Pricing Optimization",
    llm_name="pricing_optimization",
    description="Analyze pricing strategies, calculate price elasticity, and identify revenue optimization opportunities",
    capabilities="Price analysis, price elasticity calculation, competitive pricing comparison, optimal price recommendations, revenue impact simulation, price-volume tradeoffs, regional pricing analysis, brand positioning analysis",
    limitations="Requires sales, units/volume data. Elasticity calculations need sufficient price variation. Assumes other factors constant.",
    example_questions="What's the optimal price for Barilla pasta? How elastic is demand for premium pasta? Compare average prices across brands. What would happen to revenue if we increased price by 10%? Which products are underpriced? Show price vs volume tradeoff for organic segment.",
    parameter_guidance="Select dimension for analysis (brand, segment, sub_category, state). Specify time period for analysis. Apply filters as needed. For elasticity, ensure sufficient price variation in data.",
    parameters=[
        SkillParameter(
            name="dimension",
            constrained_to="dimensions",
            description="Dimension to analyze (brand, segment, sub_category, manufacturer, state_name)",
            default_value="brand"
        ),
        SkillParameter(
            name="filters",
            constrained_to="filters",
            is_multi=True,
            description="Filters to apply (e.g., specific brands, segments, regions)",
            default_value=[]
        ),
        SkillParameter(
            name="start_date",
            constrained_to="date_filter",
            description="Start date for analysis",
            default_value=None
        ),
        SkillParameter(
            name="end_date",
            constrained_to="date_filter",
            description="End date for analysis",
            default_value=None
        ),
        SkillParameter(
            name="analysis_type",
            constrained_values=["price_comparison", "elasticity", "optimization", "what_if"],
            description="Type of analysis: price_comparison, elasticity, optimization, what_if",
            default_value="price_comparison"
        ),
        SkillParameter(
            name="price_change_pct",
            description="For what-if analysis: percentage price change to simulate (e.g., 10 for +10%)",
            default_value=10
        )
    ]
)
def pricing_optimization(parameters: SkillInput):
    """Pricing optimization analysis"""

    # Extract parameters
    dimension = parameters.arguments.dimension or "brand"
    filters = parameters.arguments.filters or []
    start_date = parameters.arguments.start_date
    end_date = parameters.arguments.end_date
    analysis_type = parameters.arguments.analysis_type or "price_comparison"
    price_change_pct = parameters.arguments.price_change_pct or 10

    print(f"Running pricing optimization: {analysis_type} by {dimension}")

    # Build SQL query
    sql_query = f"""
    SELECT
        {dimension},
        month_new,
        SUM(sales) as total_sales,
        SUM(units) as total_units,
        SUM(volume) as total_volume,
        SUM(sales) / NULLIF(SUM(units), 0) as avg_price_per_unit,
        SUM(sales) / NULLIF(SUM(volume), 0) as avg_price_per_volume
    FROM read_csv('pasta_2025.csv')
    WHERE 1=1
    """

    # Add date filters
    if start_date:
        sql_query += f" AND month_new >= '{start_date}'"
    if end_date:
        sql_query += f" AND month_new <= '{end_date}'"

    # Add dimension filters (case-insensitive)
    if filters:
        for filter_item in filters:
            if isinstance(filter_item, dict) and 'dim' in filter_item and 'val' in filter_item:
                dim = filter_item['dim']
                values = filter_item['val']
                if isinstance(values, list):
                    values_str = "', '".join(str(v).upper() for v in values)
                    sql_query += f" AND UPPER({dim}) IN ('{values_str}')"
                    print(f"DEBUG: Added filter UPPER({dim}) IN ('{values_str}')")
                else:
                    sql_query += f" AND UPPER({dim}) = '{str(values).upper()}'"
                    print(f"DEBUG: Added filter UPPER({dim}) = '{str(values).upper()}'")

    sql_query += f"""
    GROUP BY {dimension}, month_new
    ORDER BY {dimension}, month_new
    """

    print(f"DEBUG: Executing SQL:\n{sql_query}")

    # Execute query
    try:
        client = AnswerRocketClient()
        result = client.data.execute_sql_query(DATABASE_ID, sql_query, row_limit=10000)

        print(f"DEBUG: Query result success: {result.success if hasattr(result, 'success') else 'Unknown'}")

        if not result.success or not hasattr(result, 'df'):
            error_msg = result.error if hasattr(result, 'error') else 'Unknown error'
            print(f"DEBUG: Query failed: {error_msg}")
            return SkillOutput(
                narrative=f"Failed to retrieve pricing data: {error_msg}",
                visualizations=[SkillVisualization(
                    title="Error",
                    layout=f"<p>Unable to load pricing data: {error_msg}</p>"
                )]
            )

        df = result.df
        print(f"DEBUG: Retrieved {len(df)} rows")

        if len(df) > 0:
            print(f"DEBUG: First few rows:\n{df.head()}")
            print(f"DEBUG: Columns: {list(df.columns)}")
            print(f"DEBUG: Unique {dimension} values: {df[dimension].unique()[:10]}")

        if df.empty:
            filter_info = ""
            if filters:
                filter_info = f"<br>Filters applied: {filters}"
            date_info = ""
            if start_date or end_date:
                date_info = f"<br>Date range: {start_date or 'any'} to {end_date or 'any'}"

            no_data_html = f"""
            <div style='padding: 30px; text-align: center; background: #fff3cd; border-left: 4px solid #ffc107;'>
                <h2 style='color: #856404;'>⚠️ No Data Found</h2>
                <p style='font-size: 16px;'>No pricing data matches your criteria:</p>
                <ul style='text-align: left; display: inline-block;'>
                    <li>Dimension: <strong>{dimension}</strong></li>
                    {f"<li>Filters: {filters}</li>" if filters else ""}
                    {f"<li>Date range: {start_date or 'any'} to {end_date or 'any'}</li>" if start_date or end_date else ""}
                </ul>
                <p style='margin-top: 20px; font-size: 14px;'><strong>Suggestions:</strong></p>
                <ul style='text-align: left; display: inline-block; font-size: 14px;'>
                    <li>Try removing date filters</li>
                    <li>Check filter values match data (case-insensitive search enabled)</li>
                    <li>Try a different dimension</li>
                </ul>
            </div>
            """
            print(f"DEBUG: No data returned. Filters: {filters}, Dates: {start_date} to {end_date}")
            return SkillOutput(
                narrative=f"No pricing data found for {dimension} with the specified filters and date range.",
                visualizations=[SkillVisualization(
                    title="No Data",
                    layout=no_data_html
                )]
            )

        # Perform analysis based on type
        if analysis_type == "price_comparison":
            analysis_result = analyze_price_comparison(df, dimension)
        elif analysis_type == "elasticity":
            analysis_result = analyze_price_elasticity(df, dimension)
        elif analysis_type == "optimization":
            analysis_result = analyze_optimization_opportunities(df, dimension)
        elif analysis_type == "what_if":
            analysis_result = analyze_what_if_scenario(df, dimension, price_change_pct)
        else:
            analysis_result = analyze_price_comparison(df, dimension)

        return analysis_result

    except Exception as e:
        print(f"Error in pricing optimization: {e}")
        import traceback
        traceback.print_exc()

        return SkillOutput(
            narrative=f"Error: {str(e)}",
            visualizations=[SkillVisualization(
                title="Error",
                layout=f"<p>An error occurred: {str(e)}</p>"
            )]
        )


def analyze_price_comparison(df: pd.DataFrame, dimension: str):
    """Compare average prices across dimension values"""

    print(f"DEBUG: analyze_price_comparison called with {len(df)} rows, dimension={dimension}")

    # Calculate average metrics by dimension
    summary = df.groupby(dimension).agg({
        'total_sales': 'sum',
        'total_units': 'sum',
        'total_volume': 'sum',
        'avg_price_per_unit': 'mean'
    }).reset_index()

    print(f"DEBUG: Grouped by {dimension}, got {len(summary)} unique values")

    summary['avg_price'] = summary['total_sales'] / summary['total_units']
    summary = summary.sort_values('avg_price', ascending=True).head(15)  # Top 15, ascending for horizontal bar

    print(f"DEBUG: Price range: ${summary['avg_price'].min():.2f} to ${summary['avg_price'].max():.2f}")

    # Calculate price premium/discount vs average
    overall_avg = summary['avg_price'].mean()
    summary['price_vs_avg'] = ((summary['avg_price'] / overall_avg - 1) * 100).round(1)

    print(f"DEBUG: Overall average price: ${overall_avg:.2f}")

    # Create Highcharts bar chart
    categories = summary[dimension].tolist()
    prices = summary['avg_price'].round(2).tolist()

    # Color bars based on price vs average
    colors = ['#dc3545' if pva < -10 else '#ffc107' if pva < 0 else '#28a745' if pva > 10 else '#17a2b8'
              for pva in summary['price_vs_avg']]

    chart_config = {
        "chart": {"type": "bar", "height": 500},
        "title": {"text": f"Price Comparison by {dimension.title()}", "style": {"fontSize": "20px", "fontWeight": "bold"}},
        "subtitle": {"text": f"Average market price: ${overall_avg:.2f}"},
        "xAxis": {
            "categories": categories,
            "title": {"text": dimension.title()}
        },
        "yAxis": {
            "min": 0,
            "title": {"text": "Average Price ($)"},
            "labels": {"format": "${value:.2f}"}
        },
        "plotOptions": {
            "bar": {
                "dataLabels": {
                    "enabled": True,
                    "format": "${point.y:.2f}"
                },
                "colorByPoint": True
            }
        },
        "colors": colors,
        "legend": {"enabled": False},
        "series": [{
            "name": "Average Price",
            "data": prices
        }],
        "credits": {"enabled": False}
    }

    # Create layout using wire_layout
    layout = {
        "sections": [{
            "layout": "vertical",
            "sections": [{
                "layout": "chart",
                "chart_data": chart_config
            }]
        }]
    }

    print(f"DEBUG: Creating chart with wire_layout, {len(categories)} categories")
    try:
        chart_html = wire_layout(layout, {})
        print(f"DEBUG: wire_layout successful, HTML length: {len(chart_html)}")
    except Exception as e:
        print(f"DEBUG: wire_layout failed: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        # Fallback to simple HTML if wire_layout fails
        chart_html = f"<p>Error rendering chart: {str(e)}</p>"

    # Add insights section
    highest = summary.iloc[-1]  # Last item (highest price)
    lowest = summary.iloc[0]   # First item (lowest price)

    insights_html = f"""
    <div style='padding: 20px; background: #f8f9fa; border-left: 4px solid #007bff; margin-top: 20px;'>
        <h3 style='margin-top: 0;'>💡 Key Insights:</h3>
        <ul style='margin-bottom: 0; font-size: 16px;'>
            <li><strong>{highest[dimension]}</strong> commands highest price at <strong>${highest['avg_price']:.2f}</strong> ({highest['price_vs_avg']:+.1f}% vs market)</li>
            <li><strong>{lowest[dimension]}</strong> has lowest price at <strong>${lowest['avg_price']:.2f}</strong> ({lowest['price_vs_avg']:+.1f}% vs market)</li>
            <li>Price spread: <strong>${(highest['avg_price'] - lowest['avg_price']):.2f}</strong> ({((highest['avg_price'] / lowest['avg_price'] - 1) * 100):.1f}% difference)</li>
        </ul>
        <div style='margin-top: 15px; padding: 10px; background: white; border-radius: 4px;'>
            <strong>Color Legend:</strong>
            <span style='color: #dc3545;'>● Significantly Below Average</span> |
            <span style='color: #ffc107;'>● Below Average</span> |
            <span style='color: #17a2b8;'>● Market Average</span> |
            <span style='color: #28a745;'>● Premium Pricing</span>
        </div>
    </div>
    """

    full_html = chart_html + insights_html

    # Generate narrative using LLM
    ar_utils = ArUtils()
    narrative_prompt = f"""Based on this pricing analysis by {dimension}:

- Market average price: ${overall_avg:.2f}
- Highest priced: {highest[dimension]} at ${highest['avg_price']:.2f} (+{highest['price_vs_avg']:.1f}% vs market)
- Lowest priced: {lowest[dimension]} at ${lowest['avg_price']:.2f} ({lowest['price_vs_avg']:.1f}% vs market)
- Price range: ${lowest['avg_price']:.2f} to ${highest['avg_price']:.2f}

Provide a brief executive summary (2-3 sentences) about the pricing landscape and what it suggests about market positioning."""

    narrative = ar_utils.get_llm_response(narrative_prompt) or "Price comparison analysis complete."

    return SkillOutput(
        narrative=narrative,
        visualizations=[
            SkillVisualization(title="Price Comparison", layout=full_html)
        ]
    )


def analyze_price_elasticity(df: pd.DataFrame, dimension: str):
    """Calculate price elasticity for each dimension value"""

    results = []

    for dim_value in df[dimension].unique():
        subset = df[df[dimension] == dim_value].copy()

        if len(subset) < 3:  # Need at least 3 data points
            continue

        subset = subset.sort_values('month_new')
        subset['price'] = subset['total_sales'] / subset['total_units']
        subset['price_pct_change'] = subset['price'].pct_change() * 100
        subset['units_pct_change'] = subset['total_units'].pct_change() * 100

        # Calculate elasticity (% change in quantity / % change in price)
        valid_rows = subset[
            (subset['price_pct_change'].abs() > 0.1) &  # Filter noise
            (subset['units_pct_change'].notna())
        ]

        if len(valid_rows) > 0:
            elasticity = (valid_rows['units_pct_change'] / valid_rows['price_pct_change']).mean()

            results.append({
                dimension: dim_value,
                'elasticity': elasticity,
                'avg_price': subset['price'].mean(),
                'total_units': subset['total_units'].sum(),
                'data_points': len(subset)
            })

    if not results:
        html = "<p>Insufficient data to calculate price elasticity. Need more price variation over time.</p>"
        return SkillOutput(
            narrative="Unable to calculate price elasticity with current data.",
            visualizations=[SkillVisualization(title="Price Elasticity", layout=html)]
        )

    elasticity_df = pd.DataFrame(results).sort_values('elasticity')

    # Create visualization
    html = f"""
    <div style='padding: 20px; font-family: Arial, sans-serif;'>
        <h2>Price Elasticity Analysis</h2>
        <p style='color: #666;'>Elasticity measures how demand changes with price. Negative = normal (price up, demand down).</p>

        <table style='width: 100%; border-collapse: collapse; margin-top: 20px;'>
            <thead style='background: #f5f5f5;'>
                <tr>
                    <th style='padding: 12px; text-align: left; border-bottom: 2px solid #ddd;'>{dimension.title()}</th>
                    <th style='padding: 12px; text-align: right; border-bottom: 2px solid #ddd;'>Elasticity</th>
                    <th style='padding: 12px; text-align: left; border-bottom: 2px solid #ddd;'>Interpretation</th>
                    <th style='padding: 12px; text-align: right; border-bottom: 2px solid #ddd;'>Avg Price</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, row in elasticity_df.head(15).iterrows():
        elasticity = row['elasticity']

        if abs(elasticity) > 1:
            interpretation = "Elastic (sensitive to price)"
            color = '#dc3545'
        elif abs(elasticity) > 0.5:
            interpretation = "Moderately elastic"
            color = '#ffc107'
        else:
            interpretation = "Inelastic (price insensitive)"
            color = '#28a745'

        html += f"""
                <tr style='border-bottom: 1px solid #eee;'>
                    <td style='padding: 12px;'><strong>{row[dimension]}</strong></td>
                    <td style='padding: 12px; text-align: right; color: {color};'><strong>{elasticity:.2f}</strong></td>
                    <td style='padding: 12px; color: {color};'>{interpretation}</td>
                    <td style='padding: 12px; text-align: right;'>${row['avg_price']:.2f}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>

        <div style='margin-top: 30px; padding: 15px; background: #f8f9fa; border-left: 4px solid #007bff;'>
            <h3 style='margin-top: 0;'>Price Elasticity Guide:</h3>
            <ul style='margin-bottom: 0;'>
                <li><strong>Elastic (|E| > 1):</strong> Demand very sensitive to price. Small price increase → large demand drop.</li>
                <li><strong>Inelastic (|E| < 1):</strong> Demand less sensitive to price. Price increases have limited impact on volume.</li>
                <li><strong>Recommendation:</strong> Increase prices on inelastic products, be cautious with elastic products.</li>
            </ul>
        </div>
    </div>
    """

    return SkillOutput(
        narrative="Price elasticity analysis shows which products are most sensitive to price changes.",
        visualizations=[SkillVisualization(title="Price Elasticity", layout=html)]
    )


def analyze_optimization_opportunities(df: pd.DataFrame, dimension: str):
    """Identify pricing optimization opportunities"""

    # Calculate metrics by dimension
    summary = df.groupby(dimension).agg({
        'total_sales': 'sum',
        'total_units': 'sum',
        'total_volume': 'sum'
    }).reset_index()

    summary['avg_price'] = summary['total_sales'] / summary['total_units']
    summary['revenue_per_unit'] = summary['total_sales'] / summary['total_units']

    # Calculate percentiles
    p25 = summary['avg_price'].quantile(0.25)
    p75 = summary['avg_price'].quantile(0.75)
    median = summary['avg_price'].median()

    # Identify opportunities
    summary['opportunity'] = summary.apply(lambda row:
        'Price Increase Potential' if row['avg_price'] < p25 and row['total_units'] > summary['total_units'].median()
        else 'Premium Positioning' if row['avg_price'] > p75
        else 'Well Positioned' if p25 <= row['avg_price'] <= p75
        else 'Monitor', axis=1
    )

    summary['potential_revenue_lift'] = summary.apply(lambda row:
        (median - row['avg_price']) * row['total_units'] if row['opportunity'] == 'Price Increase Potential'
        else 0, axis=1
    )

    summary = summary.sort_values('potential_revenue_lift', ascending=False)

    # Create visualization
    html = f"""
    <div style='padding: 20px; font-family: Arial, sans-serif;'>
        <h2>Pricing Optimization Opportunities</h2>
        <p style='color: #666;'>Market median price: <strong>${median:.2f}</strong></p>

        <h3 style='margin-top: 30px;'>Price Increase Opportunities</h3>
        <table style='width: 100%; border-collapse: collapse;'>
            <thead style='background: #f5f5f5;'>
                <tr>
                    <th style='padding: 12px; text-align: left; border-bottom: 2px solid #ddd;'>{dimension.title()}</th>
                    <th style='padding: 12px; text-align: right; border-bottom: 2px solid #ddd;'>Current Price</th>
                    <th style='padding: 12px; text-align: right; border-bottom: 2px solid #ddd;'>Market Median</th>
                    <th style='padding: 12px; text-align: right; border-bottom: 2px solid #ddd;'>Units</th>
                    <th style='padding: 12px; text-align: right; border-bottom: 2px solid #ddd;'>Potential Lift</th>
                </tr>
            </thead>
            <tbody>
    """

    opportunities = summary[summary['opportunity'] == 'Price Increase Potential'].head(10)

    if len(opportunities) == 0:
        html += "<tr><td colspan='5' style='padding: 12px; text-align: center; color: #666;'>No clear price increase opportunities identified</td></tr>"
    else:
        for _, row in opportunities.iterrows():
            html += f"""
                    <tr style='border-bottom: 1px solid #eee;'>
                        <td style='padding: 12px;'><strong>{row[dimension]}</strong></td>
                        <td style='padding: 12px; text-align: right;'>${row['avg_price']:.2f}</td>
                        <td style='padding: 12px; text-align: right;'>${median:.2f}</td>
                        <td style='padding: 12px; text-align: right;'>{row['total_units']:,.0f}</td>
                        <td style='padding: 12px; text-align: right; color: #28a745;'><strong>${row['potential_revenue_lift']:,.0f}</strong></td>
                    </tr>
            """

    html += """
            </tbody>
        </table>

        <div style='margin-top: 30px; padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107;'>
            <h3 style='margin-top: 0;'>💡 Recommendation:</h3>
    """

    if len(opportunities) > 0:
        total_lift = opportunities['potential_revenue_lift'].sum()
        html += f"""
            <p>Consider gradual price increases for underpriced high-volume products. Potential revenue lift: <strong>${total_lift:,.0f}</strong></p>
            <p><strong>Next Steps:</strong> Test 5-10% price increases, monitor volume impact, adjust based on customer response.</p>
        """
    else:
        html += "<p>Pricing appears well-optimized relative to market. Focus on maintaining position and monitoring competition.</p>"

    html += """
        </div>
    </div>
    """

    return SkillOutput(
        narrative="Analysis identifies products with potential for revenue optimization through strategic pricing adjustments.",
        visualizations=[SkillVisualization(title="Optimization Opportunities", layout=html)]
    )


def analyze_what_if_scenario(df: pd.DataFrame, dimension: str, price_change_pct: float):
    """Simulate revenue impact of price changes"""

    # Assume moderate elasticity of -0.7 (industry typical for CPG)
    assumed_elasticity = -0.7

    summary = df.groupby(dimension).agg({
        'total_sales': 'sum',
        'total_units': 'sum'
    }).reset_index()

    summary['current_price'] = summary['total_sales'] / summary['total_units']
    summary['new_price'] = summary['current_price'] * (1 + price_change_pct / 100)

    # Estimate volume impact using elasticity
    summary['estimated_volume_change_pct'] = assumed_elasticity * price_change_pct
    summary['new_units'] = summary['total_units'] * (1 + summary['estimated_volume_change_pct'] / 100)

    summary['current_revenue'] = summary['total_sales']
    summary['projected_revenue'] = summary['new_price'] * summary['new_units']
    summary['revenue_change'] = summary['projected_revenue'] - summary['current_revenue']
    summary['revenue_change_pct'] = (summary['revenue_change'] / summary['current_revenue'] * 100)

    summary = summary.sort_values('current_revenue', ascending=False).head(12)  # Top 12 by revenue

    total_current = summary['current_revenue'].sum()
    total_projected = summary['projected_revenue'].sum()
    total_change = total_projected - total_current
    total_change_pct = (total_change / total_current * 100)

    # Create Highcharts grouped column chart
    categories = summary[dimension].tolist()
    current_revenue = summary['current_revenue'].round(0).tolist()
    projected_revenue = summary['projected_revenue'].round(0).tolist()

    direction = "increase" if price_change_pct > 0 else "decrease"

    chart_config = {
        "chart": {"type": "column", "height": 500},
        "title": {
            "text": f"Revenue Impact: {abs(price_change_pct):.0f}% Price {direction.title()}",
            "style": {"fontSize": "20px", "fontWeight": "bold"}
        },
        "subtitle": {
            "text": f"Total Impact: {'+' if total_change > 0 else ''}${total_change:,.0f} ({total_change_pct:+.1f}%)",
            "style": {"fontSize": "16px", "color": "#28a745" if total_change > 0 else "#dc3545"}
        },
        "xAxis": {
            "categories": categories,
            "title": {"text": dimension.title()}
        },
        "yAxis": {
            "min": 0,
            "title": {"text": "Revenue ($)"},
            "labels": {"format": "${value:,.0f}"}
        },
        "plotOptions": {
            "column": {
                "dataLabels": {
                    "enabled": True,
                    "format": "${point.y:,.0f}"
                }
            }
        },
        "series": [
            {
                "name": "Current Revenue",
                "data": current_revenue,
                "color": "#6c757d"
            },
            {
                "name": "Projected Revenue",
                "data": projected_revenue,
                "color": "#28a745" if total_change > 0 else "#dc3545"
            }
        ],
        "legend": {
            "align": "center",
            "verticalAlign": "bottom"
        },
        "credits": {"enabled": False}
    }

    # Create layout using wire_layout
    layout = {
        "sections": [{
            "layout": "vertical",
            "sections": [{
                "layout": "chart",
                "chart_data": chart_config
            }]
        }]
    }

    print(f"DEBUG: Creating chart with wire_layout, {len(categories)} categories")
    try:
        chart_html = wire_layout(layout, {})
        print(f"DEBUG: wire_layout successful, HTML length: {len(chart_html)}")
    except Exception as e:
        print(f"DEBUG: wire_layout failed: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        # Fallback to simple HTML if wire_layout fails
        chart_html = f"<p>Error rendering chart: {str(e)}</p>"

    # Create KPI cards for overall impact
    kpi_html = f"""
    <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0;'>
        <div style='background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;'>
            <div style='color: #666; font-size: 14px; margin-bottom: 8px;'>Current Revenue</div>
            <div style='font-size: 28px; font-weight: bold; color: #495057;'>${total_current:,.0f}</div>
        </div>
        <div style='background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;'>
            <div style='color: #666; font-size: 14px; margin-bottom: 8px;'>Projected Revenue</div>
            <div style='font-size: 28px; font-weight: bold; color: {"#28a745" if total_change > 0 else "#dc3545"};'>${total_projected:,.0f}</div>
        </div>
        <div style='background: {"#d4edda" if total_change > 0 else "#f8d7da"}; padding: 20px; border-radius: 8px; text-align: center;'>
            <div style='color: #666; font-size: 14px; margin-bottom: 8px;'>Net Impact</div>
            <div style='font-size: 28px; font-weight: bold; color: {"#155724" if total_change > 0 else "#721c24"};'>
                {'+' if total_change > 0 else ''}${total_change:,.0f}
            </div>
            <div style='font-size: 16px; color: {"#155724" if total_change > 0 else "#721c24"}; margin-top: 4px;'>
                ({total_change_pct:+.1f}%)
            </div>
        </div>
    </div>
    """

    # Add assumptions panel
    assumptions_html = f"""
    <div style='margin-top: 20px; padding: 20px; background: #fff3cd; border-left: 4px solid #ffc107;'>
        <h3 style='margin-top: 0;'>⚠️ Important Assumptions:</h3>
        <ul style='margin-bottom: 0; font-size: 15px;'>
            <li>Assumes price elasticity of <strong>{assumed_elasticity}</strong> (typical for CPG products)</li>
            <li>Elasticity varies by product: premium brands typically less elastic than value brands</li>
            <li>Does not account for competitive response or market share shifts</li>
            <li>Volume impact: <strong>{assumed_elasticity * price_change_pct:.1f}%</strong> change expected</li>
            <li><strong>Recommendation:</strong> Pilot test price changes before full rollout</li>
        </ul>
    </div>
    """

    full_html = kpi_html + chart_html + assumptions_html

    recommendation = "positive" if total_change > 0 else "negative"
    narrative = f"A {abs(price_change_pct):.0f}% price {direction} would have an estimated {recommendation} revenue impact of ${abs(total_change):,.0f} ({abs(total_change_pct):.1f}%), assuming moderate price elasticity of {assumed_elasticity}."

    return SkillOutput(
        narrative=narrative,
        visualizations=[SkillVisualization(title="What-If Analysis", layout=full_html)]
    )


if __name__ == "__main__":
    # Test the skill locally
    print("✓ Pricing Optimization skill loaded successfully!")
    print("✓ 4 analysis types available:")
    print("  - price_comparison: Compare prices with Highcharts bar chart")
    print("  - elasticity: Calculate price elasticity")
    print("  - optimization: Identify pricing opportunities")
    print("  - what_if: Simulate revenue impact with column chart")
    print("\nDeploy this skill to your copilot to use it!")
