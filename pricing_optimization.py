"""
Pricing Optimization Skill for Pasta Dataset
Analyzes pricing strategies, calculates price elasticity, and identifies optimization opportunities
"""
from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np
import jinja2
from datetime import datetime
from typing import Optional, List

from skill_framework import skill, SkillParameter, SkillInput, SkillOutput, SkillVisualization, ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from answer_rocket import AnswerRocketClient
from ar_analytics import ArUtils

# Database ID for pasta dataset
DATABASE_ID = os.getenv('DATABASE_ID', '83c2268f-af77-4d00-8a6b-7181dc06643e')


@skill(
    name="Pricing Optimization",
    llm_name="pricing_optimization",
    description="ALWAYS use this skill for ANY question about brand performance, pricing, competition, or how a brand is doing. Run this skill immediately without asking clarifying questions - defaults are already set.",
    capabilities="Price analysis, price elasticity calculation, competitive pricing comparison, optimal price recommendations, revenue impact simulation, price-volume tradeoffs, regional pricing analysis, brand positioning analysis",
    limitations="Requires sales, units/volume data. Elasticity calculations need sufficient price variation. Assumes other factors constant.",
    example_questions="What's the optimal price for Barilla pasta? How elastic is demand for premium pasta? Compare average prices across brands. What would happen to revenue if we increased price by 10%? Which products are underpriced? Show price vs volume tradeoff for organic segment.",
    parameter_guidance="IMPORTANT: When a brand is mentioned in the question (e.g. 'how is Barilla performing'), you MUST add it as a brand filter. Always extract brand names from the question and include them in filters.",
    parameters=[
        SkillParameter(
            name="dimension",
            constrained_to="dimensions",
            description="Dimension to analyze within brand portfolio (sub_category, segment, base_size)",
            default_value="base_size"
        ),
        SkillParameter(
            name="filters",
            constrained_to="filters",
            is_multi=True,
            description="Filters to apply. MUST include brand filter when a brand is mentioned in the question (e.g. brand=BARILLA). Defaults to SEMOLINA subcategory.",
            default_value=[]
        ),
        SkillParameter(
            name="start_date",
            constrained_to="date_filter",
            description="Start date for analysis",
            default_value="2024-01-01"
        ),
        SkillParameter(
            name="end_date",
            constrained_to="date_filter",
            description="End date for analysis",
            default_value="2024-12-31"
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
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for the chat response (left panel).",
            default_value="Answer user question in 30 words or less using following facts:\n{{facts}}"
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt for the insights panel (right panel).",
            default_value="""Tell a cohesive pricing strategy story. Answer these 3 questions:

**DATA:**
{{facts}}

**ANSWER THESE 3 QUESTIONS:**

1. **Pricing Strategy**: Is the brand maintaining their tier positioning? Is the index change from mix shift or actual pricing?

2. **Price vs Volume Tradeoff**: When competitors cut prices, did they gain share at the brand's expense? Is the premium strategy working or costing them volume?

3. **Pack Opportunities**: Which specific packs should the brand adjust pricing on to gain share/sales?

Be direct and specific. Use the data provided. **250 words maximum.**"""
        )
    ]
)
def pricing_optimization(parameters: SkillInput):
    """Pricing optimization analysis"""

    # Extract parameters
    dimension = "base_size"
    filters = parameters.arguments.filters or []
    start_date = parameters.arguments.start_date
    end_date = parameters.arguments.end_date
    analysis_type = parameters.arguments.analysis_type or "price_comparison"
    price_change_pct = parameters.arguments.price_change_pct or 10
    max_prompt = parameters.arguments.max_prompt or "Answer user question in 30 words or less using following facts:\n{{facts}}"
    insight_prompt_template = parameters.arguments.insight_prompt or """Tell a cohesive pricing strategy story. Answer these 3 questions:

**DATA:**
{{facts}}

**ANSWER THESE 3 QUESTIONS:**

1. **Pricing Strategy**: Is the brand maintaining their tier positioning? Is the index change from mix shift or actual pricing?

2. **Price vs Volume Tradeoff**: When competitors cut prices, did they gain share at the brand's expense? Is the premium strategy working or costing them volume?

3. **Pack Opportunities**: Which specific packs should the brand adjust pricing on to gain share/sales?

Be direct and specific. Use the data provided. **250 words maximum.**"""

    # Apply default subcategory filter if none specified
    has_subcategory_filter = any(
        isinstance(f, dict) and f.get('dim') == 'sub_category'
        for f in filters
    )
    if not has_subcategory_filter:
        filters = list(filters) if filters else []
        filters.append({'dim': 'sub_category', 'val': ['SEMOLINA']})
        print("DEBUG: Applied default subcategory filter: SEMOLINA")

    print(f"Running pricing optimization: {analysis_type} by {dimension}")

    # Validate brand filter is present for optimization analysis
    if analysis_type == "optimization":
        has_brand_filter = False
        brand_value = None
        for filter_item in filters:
            if isinstance(filter_item, dict) and filter_item.get('dim') == 'brand':
                has_brand_filter = True
                brand_value = filter_item.get('val')
                if isinstance(brand_value, list) and len(brand_value) == 1:
                    brand_value = brand_value[0]
                break

        if not has_brand_filter:
            error_layout = {
                "layoutJson": {
                    "type": "Document",
                    "style": {"padding": "20px"},
                    "children": [{
                        "type": "Paragraph",
                        "children": "",
                        "text": "⚠️ Brand filter required: Please filter to a single brand (e.g., BARILLA) to analyze portfolio optimization opportunities within that brand.",
                        "style": {"fontSize": "16px", "color": "#dc3545", "padding": "20px", "backgroundColor": "#f8d7da", "borderRadius": "8px"}
                    }]
                },
                "inputVariables": []
            }
            from skill_framework.layouts import wire_layout
            error_html = wire_layout(error_layout, {})

            return SkillOutput(
                final_prompt="Brand filter required for optimization analysis.",
                narrative="## Brand Filter Required\n\nThis optimization analysis compares products within a brand's portfolio. Please filter to a single brand (e.g., BARILLA) to see pricing optimization opportunities across their product line.",
                visualizations=[SkillVisualization(title="Error", layout=error_html)]
            )

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
    # Map AR dimension names to actual SQL column names
    dim_column_map = {
        'max_time_date': 'month_new'
    }

    if filters:
        for filter_item in filters:
            if isinstance(filter_item, dict) and 'dim' in filter_item and 'val' in filter_item:
                dim = filter_item['dim']
                # Map dimension name to SQL column if needed
                sql_column = dim_column_map.get(dim, dim)
                values = filter_item['val']

                # Handle date dimensions differently (no UPPER, use date comparison)
                if dim == 'max_time_date':
                    if isinstance(values, list) and len(values) >= 2:
                        # Assume first is start, last is end for date range
                        sql_query += f" AND {sql_column} >= '{values[0]}' AND {sql_column} <= '{values[-1]}'"
                        print(f"DEBUG: Added date filter {sql_column} between '{values[0]}' and '{values[-1]}'")
                    elif isinstance(values, list) and len(values) == 1:
                        sql_query += f" AND {sql_column} = '{values[0]}'"
                        print(f"DEBUG: Added date filter {sql_column} = '{values[0]}'")
                    else:
                        sql_query += f" AND {sql_column} = '{values}'"
                        print(f"DEBUG: Added date filter {sql_column} = '{values}'")
                elif isinstance(values, list):
                    values_str = "', '".join(str(v).upper() for v in values)
                    sql_query += f" AND UPPER({sql_column}) IN ('{values_str}')"
                    print(f"DEBUG: Added filter UPPER({sql_column}) IN ('{values_str}')")
                else:
                    sql_query += f" AND UPPER({sql_column}) = '{str(values).upper()}'"
                    print(f"DEBUG: Added filter UPPER({sql_column}) = '{str(values).upper()}'")

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
                final_prompt=f"Failed to retrieve pricing data: {error_msg}",
                narrative=None,
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
                final_prompt=f"No pricing data found for {dimension} with the specified filters and date range.",
                narrative=None,
                visualizations=[SkillVisualization(
                    title="No Data",
                    layout=no_data_html
                )]
            )

        # Extract brand filter if exists
        brand_filter = None
        for filter_item in filters:
            if isinstance(filter_item, dict) and filter_item.get('dim') == 'brand':
                brand_value = filter_item.get('val')
                if isinstance(brand_value, list) and len(brand_value) == 1:
                    brand_filter = brand_value[0]
                elif isinstance(brand_value, str):
                    brand_filter = brand_value
                break

        # Perform analysis based on type
        if analysis_type == "price_comparison":
            analysis_result = analyze_price_comparison(df, dimension, brand_filter, filters, start_date, end_date)
        elif analysis_type == "elasticity":
            analysis_result = analyze_price_elasticity(df, dimension)
        elif analysis_type == "optimization":
            analysis_result = analyze_optimization_opportunities(df, dimension)
        elif analysis_type == "what_if":
            analysis_result = analyze_what_if_scenario(df, dimension, price_change_pct)
        else:
            analysis_result = analyze_price_comparison(df, dimension, brand_filter, filters)

        return analysis_result

    except Exception as e:
        print(f"Error in pricing optimization: {e}")
        import traceback
        traceback.print_exc()

        return SkillOutput(
            final_prompt=f"Error: {str(e)}",
            narrative=None,
            visualizations=[SkillVisualization(
                title="Error",
                layout=f"<p>An error occurred: {str(e)}</p>"
            )]
        )


def analyze_competitive_comparison(df: pd.DataFrame, dimension: str, brand_filter: str, filters: list, start_date: str = None, end_date: str = None):
    """Compare target brand vs competition across dimension values"""

    # Define prompt templates
    max_prompt = "Answer user question in 30 words or less using following facts:\n{{facts}}"
    insight_prompt_template = """Tell a cohesive pricing strategy story. Answer these 3 questions:

**DATA:**
{{facts}}

**ANSWER THESE 3 QUESTIONS:**

1. **Pricing Strategy**: Is the brand maintaining their tier positioning? Is the index change from mix shift or actual pricing?

2. **Price vs Volume Tradeoff**: When competitors cut prices, did they gain share at the brand's expense? Is the premium strategy working or costing them volume?

3. **Pack Opportunities**: Which specific packs should the brand adjust pricing on to gain share/sales?

Be direct and specific. Use the data provided. **250 words maximum.**"""

    print(f"DEBUG: analyze_competitive_comparison for {brand_filter} by {dimension}")

    # Create time period string for pills
    if start_date and end_date:
        time_period = f"{start_date} to {end_date}"
    elif start_date:
        time_period = f"From {start_date}"
    elif end_date:
        time_period = f"Until {end_date}"
    else:
        time_period = "All available data"

    # Re-query to get ALL brands for comparison (not just the filtered brand)
    try:
        client = AnswerRocketClient()

        # Build SQL query without brand filter - always include base_size for price per oz chart
        sql_query = f"""
        SELECT
            {dimension},
            base_size,
            brand,
            month_new,
            SUM(sales) as total_sales,
            SUM(units) as total_units,
            SUM(volume) as total_volume
        FROM read_csv('pasta_2025.csv')
        WHERE 1=1
        """

        # Add non-brand filters
        if filters:
            for filter_item in filters:
                if isinstance(filter_item, dict) and filter_item.get('dim') != 'brand':
                    dim = filter_item['dim']
                    values = filter_item.get('val')
                    if isinstance(values, list):
                        values_str = "', '".join(str(v).upper() for v in values)
                        sql_query += f" AND UPPER({dim}) IN ('{values_str}')"

        sql_query += f" GROUP BY {dimension}, base_size, brand, month_new"

        result = client.data.execute_sql_query(DATABASE_ID, sql_query, row_limit=10000)

        if not result.success or not hasattr(result, 'df'):
            error_msg = result.error if hasattr(result, 'error') else 'Unknown error'
            raise Exception(f"Query failed: {error_msg}")

        full_df = result.df
        print(f"DEBUG: Full dataset retrieved: {len(full_df)} rows")

    except Exception as e:
        print(f"DEBUG: Failed to get competitive data: {e}")
        # Fall back to regular analysis with just the brand data
        return analyze_price_comparison(df, dimension, None, None)

    # Split into target brand vs competition
    target_df = full_df[full_df['brand'].str.upper() == brand_filter.upper()].copy()
    competitors_df = full_df[full_df['brand'].str.upper() != brand_filter.upper()].copy()

    print(f"DEBUG: {brand_filter} data: {len(target_df)} rows, Competitors: {len(competitors_df)} rows")

    # Aggregate by dimension
    target_summary = target_df.groupby(dimension).agg({
        'total_sales': 'sum',
        'total_units': 'sum',
        'total_volume': 'sum'
    }).reset_index()
    target_summary['avg_price'] = target_summary['total_sales'] / target_summary['total_units']
    target_summary['brand_name'] = brand_filter

    comp_summary = competitors_df.groupby(dimension).agg({
        'total_sales': 'sum',
        'total_units': 'sum',
        'total_volume': 'sum'
    }).reset_index()
    comp_summary['avg_price'] = comp_summary['total_sales'] / comp_summary['total_units']
    comp_summary['brand_name'] = 'Competition Avg'

    # Merge on dimension to get side-by-side comparison
    comparison = target_summary.merge(
        comp_summary[[dimension, 'avg_price']],
        on=dimension,
        how='outer',
        suffixes=('_target', '_comp')
    )

    comparison = comparison.sort_values('avg_price_target', ascending=True).head(15)
    comparison['price_premium_pct'] = ((comparison['avg_price_target'] / comparison['avg_price_comp'] - 1) * 100).round(1)

    print(f"DEBUG: Comparison data: {len(comparison)} dimension values")

    # Calculate competitive metrics for KPIs
    total_target_units = target_summary['total_units'].sum()
    total_all_units = full_df['total_units'].sum()
    volume_share = (total_target_units / total_all_units * 100).round(1)

    # Weighted average premium
    comparison_clean = comparison.dropna()
    if len(comparison_clean) > 0:
        weighted_premium = (comparison_clean['price_premium_pct'] * comparison_clean['total_units']).sum() / comparison_clean['total_units'].sum()
        price_leaders = len(comparison_clean[comparison_clean['price_premium_pct'] > 0])
    else:
        weighted_premium = 0
        price_leaders = 0

    num_skus = len(comparison)

    # Capitalize brand name
    brand_display = brand_filter.upper()

    # Create grouped column chart with color-coded bars
    categories = comparison[dimension].fillna('Unknown').tolist()
    target_prices = comparison['avg_price_target'].fillna(0).round(2).tolist()
    comp_prices = comparison['avg_price_comp'].fillna(0).round(2).tolist()

    # Color-code BARILLA bars: green if above competition, red if below
    target_data_with_colors = []
    for idx, (target, comp, gap_pct) in enumerate(zip(target_prices, comp_prices, comparison['price_premium_pct'].fillna(0).tolist())):
        gap_dollars = target - comp
        color = "#22c55e" if gap_pct > 0 else "#ef4444"  # Green if premium, red if discount
        target_data_with_colors.append({
            "y": target,
            "color": color,
            "dataLabels": {
                "enabled": True,
                "format": f"${target:.2f}\n({gap_pct:+.0f}%)",
                "style": {
                    "color": color,
                    "fontSize": "11px",
                    "fontWeight": "bold",
                    "textOutline": "none"
                }
            }
        })

    chart_config = {
        "chart": {"type": "column", "height": 500},
        "title": {
            "text": f"{brand_display} vs Competition by {dimension.replace('_', ' ').title()}",
            "style": {"fontSize": "20px", "fontWeight": "bold"}
        },
        "xAxis": {
            "categories": categories,
            "title": {"text": dimension.replace('_', ' ').title()}
        },
        "yAxis": {
            "min": 0,
            "title": {"text": "Average Price ($)"},
            "labels": {"format": "${value:.2f}"}
        },
        "tooltip": {
            "shared": True,
            "backgroundColor": "rgba(255, 255, 255, 1)",
            "borderColor": "#333",
            "borderWidth": 2,
            "hideDelay": 0,
            "style": {
                "color": "#333",
                "fontSize": "12px",
                "pointerEvents": "none"
            },
            "split": False,
            "outside": False
        },
        "plotOptions": {
            "column": {
                "dataLabels": {
                    "enabled": True
                }
            }
        },
        "series": [
            {
                "name": brand_display,
                "data": target_data_with_colors
            },
            {
                "name": "Competition Avg",
                "data": comp_prices,
                "color": "#9ca3af",
                "dataLabels": {
                    "enabled": True,
                    "format": "${point.y:.2f}"
                }
            }
        ],
        "legend": {
            "align": "center",
            "verticalAlign": "bottom"
        },
        "credits": {"enabled": False}
    }

    # Always create price per oz chart by base_size (regardless of dimension analyzed)
    # Re-aggregate by base_size for normalized comparison
    base_size_target = target_df.groupby('base_size').agg({
        'total_sales': 'sum',
        'total_units': 'sum',
        'total_volume': 'sum'
    }).reset_index()
    base_size_target['avg_price'] = base_size_target['total_sales'] / base_size_target['total_units']

    # Filter to top 15 base sizes by sales (to avoid cluttered chart)
    base_size_target = base_size_target.nlargest(15, 'total_sales')

    base_size_comp = competitors_df.groupby('base_size').agg({
        'total_sales': 'sum',
        'total_units': 'sum',
        'total_volume': 'sum'
    }).reset_index()
    base_size_comp['avg_price'] = base_size_comp['total_sales'] / base_size_comp['total_units']

    # Merge and calculate price per oz - use inner join to only show sizes where BARILLA competes
    base_size_comparison = base_size_target.merge(
        base_size_comp[['base_size', 'avg_price']],
        on='base_size',
        how='inner',
        suffixes=('_target', '_comp')
    ).sort_values('avg_price_target', ascending=True)

    base_size_comparison['price_premium_pct'] = ((base_size_comparison['avg_price_target'] / base_size_comparison['avg_price_comp'] - 1) * 100).round(1)

    # Extract oz from base_size and calculate price per oz
    base_size_comparison['price_per_oz_target'] = base_size_comparison.apply(
        lambda row: row['avg_price_target'] / float(row['base_size'].split()[0]) if pd.notna(row['avg_price_target']) and row['base_size'].split()[0].replace('.','').isdigit() else None,
        axis=1
    )
    base_size_comparison['price_per_oz_comp'] = base_size_comparison.apply(
        lambda row: row['avg_price_comp'] / float(row['base_size'].split()[0]) if pd.notna(row['avg_price_comp']) and row['base_size'].split()[0].replace('.','').isdigit() else None,
        axis=1
    )

    base_size_categories = base_size_comparison['base_size'].tolist()

    price_per_oz_data_target = []
    for idx, row in base_size_comparison.iterrows():
        if pd.notna(row['price_per_oz_target']):
            gap_pct = row['price_premium_pct']
            color = "#22c55e" if gap_pct > 0 else "#ef4444"
            price_per_oz_data_target.append({
                "y": round(row['price_per_oz_target'], 2),
                "color": color
            })
        else:
            price_per_oz_data_target.append(None)

    price_per_oz_data_comp = [round(x, 2) if pd.notna(x) else None for x in base_size_comparison['price_per_oz_comp'].tolist()]

    price_per_oz_chart = {
        "chart": {"type": "column", "height": 400},
        "title": {
            "text": f"Price Per Ounce by Base Size: {brand_display} vs Competition",
            "style": {"fontSize": "18px", "fontWeight": "bold"}
        },
        "xAxis": {
            "categories": base_size_categories,
            "title": {"text": "Base Size"}
        },
        "yAxis": {
            "min": 0,
            "title": {"text": "Price Per Ounce ($)"},
            "labels": {"format": "${value:.2f}"}
        },
        "tooltip": {
            "shared": True,
            "valuePrefix": "$",
            "valueDecimals": 2,
            "valueSuffix": "/oz",
            "backgroundColor": "#ffffff",
            "borderColor": "#999",
            "borderWidth": 2,
            "borderRadius": 4,
            "shadow": True,
            "style": {
                "opacity": 1
            }
        },
        "plotOptions": {
            "column": {
                "dataLabels": {
                    "enabled": True,
                    "format": "${point.y:.2f}/oz"
                }
            }
        },
        "series": [
            {
                "name": brand_display,
                "data": price_per_oz_data_target
            },
            {
                "name": "Competition Avg",
                "data": price_per_oz_data_comp,
                "color": "#9ca3af"
            }
        ],
        "legend": {
            "align": "center",
            "verticalAlign": "bottom"
        },
        "credits": {"enabled": False}
    }

    # Create structured layout
    comparison_layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"backgroundColor": "#ffffff", "padding": "20px"},
            "children": [
                # Banner
                {
                    "name": "Banner",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "style": {
                        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                        "padding": "30px",
                        "borderRadius": "12px",
                        "marginBottom": "25px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
                    }
                },
                {
                    "name": "BannerTitle",
                    "type": "Header",
                    "children": "",
                    "text": f"Competitive Positioning: {brand_display}",
                    "parentId": "Banner",
                    "style": {"fontSize": "28px", "fontWeight": "bold", "color": "white", "marginBottom": "10px"}
                },
                {
                    "name": "BannerSubtitle",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"Comparing {brand_display} vs market average across {num_skus} {dimension.replace('_', ' ')} values",
                    "parentId": "Banner",
                    "style": {"fontSize": "16px", "color": "rgba(255,255,255,0.9)"}
                },
                # KPI Cards Row
                {
                    "name": "KPI_Row",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "row",
                    "extraStyles": "gap: 15px; margin-bottom: 25px;"
                },
                # KPI Card 1: Competitive Premium
                {
                    "name": "KPI_Card1",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#e3f2fd" if weighted_premium > 0 else "#ffebee",
                        "borderLeft": f"4px solid {'#2196f3' if weighted_premium > 0 else '#f44336'}",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI1_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Avg Competitive Premium",
                    "parentId": "KPI_Card1",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI1_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{weighted_premium:+.1f}%",
                    "parentId": "KPI_Card1",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#1976d2" if weighted_premium > 0 else "#d32f2f"}
                },
                # KPI Card 2: Volume Share
                {
                    "name": "KPI_Card2",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#e8f5e9",
                        "borderLeft": "4px solid #4caf50",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI2_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Volume Share",
                    "parentId": "KPI_Card2",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI2_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{volume_share:.1f}%",
                    "parentId": "KPI_Card2",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#388e3c"}
                },
                # KPI Card 3: Price Leaders
                {
                    "name": "KPI_Card3",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#fff3e0",
                        "borderLeft": "4px solid #ff9800",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI3_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Price Leaders",
                    "parentId": "KPI_Card3",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI3_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{price_leaders} of {num_skus}",
                    "parentId": "KPI_Card3",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#f57c00"}
                },
                # KPI Card 4: Total SKUs
                {
                    "name": "KPI_Card4",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#fce4ec",
                        "borderLeft": "4px solid #e91e63",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI4_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{dimension.replace('_', ' ').title()}s Analyzed",
                    "parentId": "KPI_Card4",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI4_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{num_skus}",
                    "parentId": "KPI_Card4",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#c2185b"}
                },
                # Comparison Chart
                {
                    "name": "ComparisonChart",
                    "type": "HighchartsChart",
                    "children": "",
                    "minHeight": "500px",
                    "options": chart_config
                },
                # Opportunities Table Header
                {
                    "name": "TableHeader",
                    "type": "Header",
                    "children": "",
                    "text": "Top Pricing Opportunities",
                    "style": {"fontSize": "20px", "fontWeight": "bold", "marginTop": "30px", "marginBottom": "15px"}
                },
                # Opportunities Table
                {
                    "name": "OpportunitiesTable",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "extraStyles": "display: grid; grid-template-columns: 1.5fr 1fr 1fr 1fr 1.5fr; gap: 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; margin-bottom: 30px;"
                },
                # Table Headers
                {
                    "name": "TH_Size",
                    "type": "Paragraph",
                    "children": "",
                    "text": dimension.replace('_', ' ').title(),
                    "parentId": "OpportunitiesTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd"}
                },
                {
                    "name": "TH_Brand",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{brand_display} Price",
                    "parentId": "OpportunitiesTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}
                },
                {
                    "name": "TH_Comp",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Competition",
                    "parentId": "OpportunitiesTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}
                },
                {
                    "name": "TH_Gap",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Gap ($/%)",
                    "parentId": "OpportunitiesTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}
                },
                {
                    "name": "TH_Action",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Opportunity",
                    "parentId": "OpportunitiesTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd"}
                }
            ] + (
                # Table rows for top 5 underpriced items (only show if price_premium_pct < 0)
                [
                    item
                    for idx, row in comparison[comparison['price_premium_pct'] < 0].nsmallest(5, 'price_premium_pct').iterrows()
                    for item in [
                        {
                            "name": f"TR{idx}_Size",
                            "type": "Paragraph",
                            "children": "",
                            "text": str(row[dimension]),
                            "parentId": "OpportunitiesTable",
                            "style": {"padding": "12px", "fontWeight": "bold", "borderBottom": "1px solid #eee"}
                        },
                        {
                            "name": f"TR{idx}_Brand",
                            "type": "Paragraph",
                            "children": "",
                            "text": f"${row['avg_price_target']:.2f}",
                            "parentId": "OpportunitiesTable",
                            "style": {"padding": "12px", "textAlign": "right", "borderBottom": "1px solid #eee"}
                        },
                        {
                            "name": f"TR{idx}_Comp",
                            "type": "Paragraph",
                            "children": "",
                            "text": f"${row['avg_price_comp']:.2f}",
                            "parentId": "OpportunitiesTable",
                            "style": {"padding": "12px", "textAlign": "right", "borderBottom": "1px solid #eee"}
                        },
                        {
                            "name": f"TR{idx}_Gap",
                            "type": "Paragraph",
                            "children": "",
                            "text": f"${(row['avg_price_target'] - row['avg_price_comp']):.2f} ({row['price_premium_pct']:+.0f}%)",
                            "parentId": "OpportunitiesTable",
                            "style": {"padding": "12px", "textAlign": "right", "color": "#ef4444", "fontWeight": "bold", "borderBottom": "1px solid #eee"}
                        },
                        {
                            "name": f"TR{idx}_Action",
                            "type": "Paragraph",
                            "children": "",
                            "text": "Raise price to match competition",
                            "parentId": "OpportunitiesTable",
                            "style": {"padding": "12px", "fontSize": "13px", "color": "#666", "borderBottom": "1px solid #eee"}
                        }
                    ]
                ] if len(comparison[comparison['price_premium_pct'] < 0]) > 0 else [
                    # No underpricing opportunities message
                    {
                        "name": "NoOpps",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"No underpricing opportunities found - {brand_display} is priced at or above competition across all {dimension.replace('_', ' ')} values",
                        "parentId": "OpportunitiesTable",
                        "style": {"padding": "20px", "textAlign": "center", "color": "#28a745", "fontWeight": "500", "gridColumn": "1 / -1", "fontSize": "15px"}
                    }
                ]
            ) + [
                # Price Per Oz Chart
                {
                    "name": "PricePerOzChart",
                    "type": "HighchartsChart",
                    "children": "",
                    "minHeight": "400px",
                    "options": price_per_oz_chart
                }
            ]
        },
        "inputVariables": []
    }

    html = wire_layout(comparison_layout, {})

    brief_summary = f"{brand_display} positioned at {weighted_premium:+.1f}% vs competition with {volume_share:.1f}% volume share."

    # Create pills (period pills added later after YoY calculation)
    param_pills = [
        ParameterDisplayDescription(key="brand", value=f"Brand: {brand_display}"),
        ParameterDisplayDescription(key="dimension", value=f"Dimension: {dimension.replace('_', ' ').title()}"),
        ParameterDisplayDescription(key="skus", value=f"SKUs: {num_skus}"),
        ParameterDisplayDescription(key="premium", value=f"Avg Premium: {weighted_premium:+.1f}%"),
    ]

    # Add filter pills for any additional filters (excluding brand)
    if filters:
        for filter_item in filters:
            if isinstance(filter_item, dict) and filter_item.get('dim') != 'brand':
                dim = filter_item['dim']
                values = filter_item.get('val')
                if values:
                    if isinstance(values, list):
                        values_display = ', '.join(str(v) for v in values)
                    else:
                        values_display = str(values)
                    param_pills.append(
                        ParameterDisplayDescription(
                            key=f"filter_{dim}",
                            value=f"{dim.replace('_', ' ').title()}: {values_display}"
                        )
                    )

    # ===== TAB 2: COMPETITOR THREAT ANALYSIS =====
    # Calculate competitor metrics using YoY comparison
    competitor_metrics = []

    # Determine current and prior periods based on date filters (YoY)
    from dateutil.relativedelta import relativedelta

    # Parse the date filters to determine current period
    if start_date and end_date:
        curr_start = pd.to_datetime(start_date)
        curr_end = pd.to_datetime(end_date)
        # Prior period = same dates -1 year
        prior_start = curr_start - relativedelta(years=1)
        prior_end = curr_end - relativedelta(years=1)
    elif start_date:
        curr_start = pd.to_datetime(start_date)
        curr_end = full_df['month_new'].max()
        prior_start = curr_start - relativedelta(years=1)
        prior_end = curr_end - relativedelta(years=1)
    else:
        # No date filter - use most recent year vs prior year
        all_months = sorted(full_df['month_new'].unique())
        curr_end = all_months[-1]
        curr_start = curr_end - relativedelta(years=1) + relativedelta(days=1)
        prior_end = curr_start - relativedelta(days=1)
        prior_start = prior_end - relativedelta(years=1) + relativedelta(days=1)

    # Store for pills display
    prior_period_start = prior_start
    prior_period_end = prior_end
    current_period_start = curr_start
    current_period_end = curr_end

    # Convert month_new to datetime for comparison
    full_df['month_new'] = pd.to_datetime(full_df['month_new'])

    # Filter data for each period
    current_df = full_df[(full_df['month_new'] >= curr_start) & (full_df['month_new'] <= curr_end)]
    prior_df = full_df[(full_df['month_new'] >= prior_start) & (full_df['month_new'] <= prior_end)]

    # Calculate total market for each period
    prior_market_units = prior_df['total_units'].sum() if len(prior_df) > 0 else 0
    current_market_units = current_df['total_units'].sum() if len(current_df) > 0 else 0

    for brand in full_df['brand'].unique():
        if brand.upper() == brand_filter.upper():
            continue  # Skip target brand

        # Get brand data for each period
        prior_data = prior_df[prior_df['brand'] == brand]
        current_data = current_df[current_df['brand'] == brand]

        # Prior period metrics
        prior_sales = prior_data['total_sales'].sum() if len(prior_data) > 0 else 0
        prior_units = prior_data['total_units'].sum() if len(prior_data) > 0 else 0
        prior_volume = prior_data['total_volume'].sum() if len(prior_data) > 0 else 0
        prior_price = prior_sales / prior_units if prior_units > 0 else 0
        prior_share = (prior_units / prior_market_units * 100) if prior_market_units > 0 else 0

        # Current period metrics
        current_sales = current_data['total_sales'].sum() if len(current_data) > 0 else 0
        current_units = current_data['total_units'].sum() if len(current_data) > 0 else 0
        current_volume = current_data['total_volume'].sum() if len(current_data) > 0 else 0
        current_price = current_sales / current_units if current_units > 0 else 0
        current_share = (current_units / current_market_units * 100) if current_market_units > 0 else 0

        # Growth metrics (YoY)
        volume_growth = ((current_units - prior_units) / prior_units * 100) if prior_units > 0 else 0
        price_change = ((current_price - prior_price) / prior_price * 100) if prior_price > 0 else 0
        sales_growth = ((current_sales - prior_sales) / prior_sales * 100) if prior_sales > 0 else 0
        share_change = current_share - prior_share

        # Threat score
        threat_score = (volume_growth * 0.7) - (price_change * 0.3)

        competitor_metrics.append({
            'brand': brand,
            'prior_sales': prior_sales,
            'prior_share': prior_share,
            'prior_volume': prior_volume,
            'prior_price': prior_price,
            'current_sales': current_sales,
            'current_share': current_share,
            'current_volume': current_volume,
            'current_price': current_price,
            'market_share': current_share,
            'volume_growth': volume_growth,
            'price_change': price_change,
            'sales_growth': sales_growth,
            'share_change': share_change,
            'threat_score': threat_score
        })

    competitors_df_analysis = pd.DataFrame(competitor_metrics)

    # Filter out tiny competitors (< 1% market share) to avoid noise
    MIN_SHARE_THRESHOLD = 1.0
    competitors_df_analysis = competitors_df_analysis[competitors_df_analysis['market_share'] >= MIN_SHARE_THRESHOLD]

    # Sort by threat score
    competitors_df_analysis = competitors_df_analysis.sort_values('threat_score', ascending=False)

    # Build Bubble Chart
    bubble_data = []
    for _, row in competitors_df_analysis.iterrows():
        # Determine color based on threat
        if row['volume_growth'] > 0 and row['price_change'] <= 0:
            color = '#ef4444'  # Red - highest threat
        elif row['volume_growth'] > 0:
            color = '#fbbf24'  # Yellow - watch
        else:
            color = '#10b981'  # Green - declining

        bubble_data.append({
            'x': round(row['price_change'], 1),
            'y': round(row['volume_growth'], 1),
            'z': round(row['market_share'], 1),
            'name': row['brand'],
            'color': color
        })

    bubble_chart = {
        "chart": {"type": "bubble", "height": 500, "zoomType": "xy"},
        "title": {
            "text": "Competitive Threat Matrix",
            "style": {"fontSize": "20px", "fontWeight": "bold"}
        },
        "subtitle": {
            "text": "Showing competitors with ≥1% market share | Bubble size = Market Share | Red = Threat | Yellow = Watch | Green = Declining",
            "style": {"fontSize": "14px", "color": "#666"}
        },
        "xAxis": {
            "title": {"text": "Price Change (%)"},
            "gridLineWidth": 1,
            "plotLines": [{
                "color": "#999",
                "width": 2,
                "value": 0,
                "dashStyle": "Dash"
            }]
        },
        "yAxis": {
            "title": {"text": "Volume Growth (%)"},
            "gridLineWidth": 1,
            "plotLines": [{
                "color": "#999",
                "width": 2,
                "value": 0,
                "dashStyle": "Dash"
            }]
        },
        "tooltip": {
            "shared": False,
            "backgroundColor": "rgba(255, 255, 255, 1)",
            "borderColor": "#333",
            "borderWidth": 2,
            "useHTML": False,
            "pointFormat": "<b>{point.name}</b><br/>Volume Growth: {point.y}%<br/>Price Change: {point.x}%<br/>Market Share: {point.z}%"
        },
        "plotOptions": {
            "bubble": {
                "minSize": 10,
                "maxSize": 60,
                "dataLabels": {
                    "enabled": True,
                    "format": "{point.name}",
                    "style": {"fontSize": "10px", "fontWeight": "bold", "textOutline": "none"}
                }
            }
        },
        "series": [{
            "name": "Competitors",
            "data": bubble_data
        }],
        "legend": {"enabled": False},
        "credits": {"enabled": False}
    }

    # Build Threat Table
    top_threats = competitors_df_analysis.head(5)

    # Generate insights with LLM (after threat calculation)
    ar_utils = ArUtils()

    # Get top 3 premium and top 3 discount items for pricing insights
    premium_items = comparison_clean.nlargest(3, 'price_premium_pct') if len(comparison_clean) > 0 else pd.DataFrame()
    discount_items = comparison_clean.nsmallest(3, 'price_premium_pct') if len(comparison_clean) > 0 else pd.DataFrame()

    # Format top threats for LLM prompt
    threat_summary = []
    for _, row in top_threats.iterrows():
        threat_level = "🔴 HIGH" if (row['volume_growth'] > 0 and row['price_change'] <= 0) else ("🟡 WATCH" if row['volume_growth'] > 0 else "🟢 LOW")
        sales_display = f"${row['current_sales']/1e6:.1f}M" if row['current_sales'] >= 1e6 else f"${row['current_sales']/1e3:.0f}K"
        threat_summary.append(f"- {row['brand']} ({sales_display}, {row['market_share']:.1f}% share): {threat_level} - Volume {row['volume_growth']:+.1f}%, Price {row['price_change']:+.1f}%")

    # Store data for LLM insight generation (will be done after price index analysis)
    insight_data = {
        'brand_filter': brand_filter,
        'dimension': dimension,
        'weighted_premium': weighted_premium,
        'volume_share': volume_share,
        'premium_items': premium_items,
        'discount_items': discount_items,
        'threat_summary': threat_summary,
        'price_leaders': price_leaders,
        'num_skus': num_skus
    }

    threat_table_layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"padding": "20px"},
            "children": [
                {
                    "name": "ThreatHeader",
                    "type": "Header",
                    "children": "",
                    "text": "Top 5 Competitor Threats",
                    "style": {"fontSize": "20px", "fontWeight": "bold", "marginBottom": "15px"}
                },
                {
                    "name": "ThreatTable",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "extraStyles": "display: grid; grid-template-columns: 1.4fr repeat(12, 1fr) 0.8fr; gap: 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; font-size: 12px;"
                },
                # Headers - 3 columns per metric: Prior | Current | Growth
                {"name": "TH_Brand", "type": "Paragraph", "children": "", "text": "Brand", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd"}},
                {"name": "TH_PriorSales", "type": "Paragraph", "children": "", "text": "Prior $", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#e8f4fc", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_CurrSales", "type": "Paragraph", "children": "", "text": "Curr $", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_SalesGrowth", "type": "Paragraph", "children": "", "text": "$ Chg", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#fff3cd", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_PriorShare", "type": "Paragraph", "children": "", "text": "Prior Shr", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#e8f4fc", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_CurrShare", "type": "Paragraph", "children": "", "text": "Curr Shr", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_ShareGrowth", "type": "Paragraph", "children": "", "text": "Shr Chg", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#fff3cd", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_PriorVol", "type": "Paragraph", "children": "", "text": "Prior Vol", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#e8f4fc", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_CurrVol", "type": "Paragraph", "children": "", "text": "Curr Vol", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_VolGrowth", "type": "Paragraph", "children": "", "text": "Vol Chg", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#fff3cd", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_PriorPrice", "type": "Paragraph", "children": "", "text": "Prior Prc", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#e8f4fc", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_CurrPrice", "type": "Paragraph", "children": "", "text": "Curr Prc", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_PriceChg", "type": "Paragraph", "children": "", "text": "Prc Chg", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#fff3cd", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "TH_Threat", "type": "Paragraph", "children": "", "text": "Threat", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "center"}}
            ] + [
                item
                for idx, row in top_threats.iterrows()
                for item in [
                    {"name": f"TR{idx}_Brand", "type": "Paragraph", "children": "", "text": row['brand'], "parentId": "ThreatTable", "style": {"padding": "8px 6px", "fontWeight": "bold", "borderBottom": "1px solid #eee"}},
                    {"name": f"TR{idx}_PriorSales", "type": "Paragraph", "children": "", "text": f"${row['prior_sales']/1e6:.1f}M" if row['prior_sales'] >= 1e6 else f"${row['prior_sales']/1e3:.0f}K", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "borderBottom": "1px solid #eee", "backgroundColor": "#f8fbfe"}},
                    {"name": f"TR{idx}_CurrSales", "type": "Paragraph", "children": "", "text": f"${row['current_sales']/1e6:.1f}M" if row['current_sales'] >= 1e6 else f"${row['current_sales']/1e3:.0f}K", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "borderBottom": "1px solid #eee"}},
                    {"name": f"TR{idx}_SalesGrowth", "type": "Paragraph", "children": "", "text": f"{row['sales_growth']:+.1f}%", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "color": "#22c55e" if row['sales_growth'] > 0 else "#ef4444", "fontWeight": "bold", "borderBottom": "1px solid #eee", "backgroundColor": "#fffbeb"}},
                    {"name": f"TR{idx}_PriorShare", "type": "Paragraph", "children": "", "text": f"{row['prior_share']:.1f}%", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "borderBottom": "1px solid #eee", "backgroundColor": "#f8fbfe"}},
                    {"name": f"TR{idx}_CurrShare", "type": "Paragraph", "children": "", "text": f"{row['current_share']:.1f}%", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "borderBottom": "1px solid #eee"}},
                    {"name": f"TR{idx}_ShareGrowth", "type": "Paragraph", "children": "", "text": f"{row['share_change']:+.1f}pp", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "color": "#22c55e" if row['share_change'] > 0 else "#ef4444", "fontWeight": "bold", "borderBottom": "1px solid #eee", "backgroundColor": "#fffbeb"}},
                    {"name": f"TR{idx}_PriorVol", "type": "Paragraph", "children": "", "text": f"{row['prior_volume']/1e6:.1f}M" if row['prior_volume'] >= 1e6 else f"{row['prior_volume']/1e3:.0f}K", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "borderBottom": "1px solid #eee", "backgroundColor": "#f8fbfe"}},
                    {"name": f"TR{idx}_CurrVol", "type": "Paragraph", "children": "", "text": f"{row['current_volume']/1e6:.1f}M" if row['current_volume'] >= 1e6 else f"{row['current_volume']/1e3:.0f}K", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "borderBottom": "1px solid #eee"}},
                    {"name": f"TR{idx}_VolGrowth", "type": "Paragraph", "children": "", "text": f"{row['volume_growth']:+.1f}%", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "color": "#22c55e" if row['volume_growth'] > 0 else "#ef4444", "fontWeight": "bold", "borderBottom": "1px solid #eee", "backgroundColor": "#fffbeb"}},
                    {"name": f"TR{idx}_PriorPrice", "type": "Paragraph", "children": "", "text": f"${row['prior_price']:.2f}", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "borderBottom": "1px solid #eee", "backgroundColor": "#f8fbfe"}},
                    {"name": f"TR{idx}_CurrPrice", "type": "Paragraph", "children": "", "text": f"${row['current_price']:.2f}", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "borderBottom": "1px solid #eee"}},
                    {"name": f"TR{idx}_PriceChg", "type": "Paragraph", "children": "", "text": f"{row['price_change']:+.1f}%", "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "right", "color": "#ef4444" if row['price_change'] < 0 else "#22c55e", "fontWeight": "bold", "borderBottom": "1px solid #eee", "backgroundColor": "#fffbeb"}},
                    {"name": f"TR{idx}_Threat", "type": "Paragraph", "children": "", "text": "🔴" if (row['volume_growth'] > 0 and row['price_change'] <= 0) else ("🟡" if row['volume_growth'] > 0 else "🟢"), "parentId": "ThreatTable", "style": {"padding": "8px 6px", "textAlign": "center", "fontWeight": "bold", "borderBottom": "1px solid #eee"}}
                ]
            ]
        },
        "inputVariables": []
    }

    # Combine bubble chart and threat table in Tab 2 - merge layouts
    tab2_layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"padding": "20px"},
            "children": [
                {
                    "name": "BubbleChart",
                    "type": "HighchartsChart",
                    "children": "",
                    "minHeight": "500px",
                    "options": bubble_chart
                }
            ] + threat_table_layout["layoutJson"]["children"]
        },
        "inputVariables": []
    }

    combined_tab2_html = wire_layout(tab2_layout, {})

    # Add YoY period pills
    prior_start_str = str(prior_period_start)[:7]  # YYYY-MM
    prior_end_str = str(prior_period_end)[:7]
    curr_start_str = str(current_period_start)[:7]
    curr_end_str = str(current_period_end)[:7]

    if prior_start_str == prior_end_str:
        prior_text = f"Prior: {prior_start_str}"
    else:
        prior_text = f"Prior: {prior_start_str} to {prior_end_str}"

    if curr_start_str == curr_end_str:
        curr_text = f"Current: {curr_start_str}"
    else:
        curr_text = f"Current: {curr_start_str} to {curr_end_str}"

    param_pills.append(ParameterDisplayDescription(key="prior_period", value=prior_text))
    param_pills.append(ParameterDisplayDescription(key="current_period", value=curr_text))

    # ===== TAB 4: PRICE INDEX =====
    # CPG Price Index Analysis with Tier Classification and Mix vs Price Diagnosis

    # Helper function to classify tier
    def get_tier(index_val):
        if index_val < 80:
            return ("Value", "#9ca3af")  # gray
        elif index_val < 120:
            return ("Core", "#3b82f6")  # blue
        elif index_val < 200:
            return ("Premium", "#22c55e")  # green
        else:
            return ("Super Premium", "#a855f7")  # purple

    # Get monthly price data for all brands
    monthly_prices = full_df.groupby(['month_new', 'brand']).agg({
        'total_sales': 'sum',
        'total_units': 'sum'
    }).reset_index()
    monthly_prices['avg_price'] = monthly_prices['total_sales'] / monthly_prices['total_units']

    # Calculate category average price per month
    category_avg = full_df.groupby('month_new').agg({
        'total_sales': 'sum',
        'total_units': 'sum'
    }).reset_index()
    category_avg['category_avg_price'] = category_avg['total_sales'] / category_avg['total_units']

    # Merge and calculate index
    monthly_prices = monthly_prices.merge(category_avg[['month_new', 'category_avg_price']], on='month_new')
    monthly_prices['price_index'] = (monthly_prices['avg_price'] / monthly_prices['category_avg_price']) * 100

    # Calculate current index for target brand to find peer tier
    target_current = monthly_prices[(monthly_prices['brand'].str.upper() == brand_filter.upper()) &
                                     (monthly_prices['month_new'] >= curr_start) &
                                     (monthly_prices['month_new'] <= curr_end)]
    target_curr_idx = target_current['price_index'].mean() if len(target_current) > 0 else 100
    target_tier, target_tier_color = get_tier(target_curr_idx)

    # Get all brands with their current index and tier
    all_brands_index = []
    for brand in full_df['brand'].unique():
        brand_current = monthly_prices[(monthly_prices['brand'].str.upper() == brand.upper()) &
                                        (monthly_prices['month_new'] >= curr_start) &
                                        (monthly_prices['month_new'] <= curr_end)]
        brand_prior = monthly_prices[(monthly_prices['brand'].str.upper() == brand.upper()) &
                                      (monthly_prices['month_new'] >= prior_start) &
                                      (monthly_prices['month_new'] <= prior_end)]
        curr_idx = brand_current['price_index'].mean() if len(brand_current) > 0 else 0
        prior_idx = brand_prior['price_index'].mean() if len(brand_prior) > 0 else 0
        idx_change = curr_idx - prior_idx
        tier, tier_color = get_tier(curr_idx)

        # Get volume for sorting
        brand_vol = full_df[full_df['brand'].str.upper() == brand.upper()]['total_units'].sum()

        all_brands_index.append({
            'brand': brand,
            'prior_index': prior_idx,
            'current_index': curr_idx,
            'index_change': idx_change,
            'tier': tier,
            'tier_color': tier_color,
            'volume': brand_vol,
            'flagged': abs(idx_change) > 5  # Flag if change > 5 points
        })

    # Sort by volume descending
    all_brands_index = sorted(all_brands_index, key=lambda x: x['volume'], reverse=True)

    # Get top brands for chart (including target)
    top_brands_for_index = [b['brand'] for b in all_brands_index[:6]]
    if brand_filter.upper() not in [b.upper() for b in top_brands_for_index]:
        top_brands_for_index = [brand_filter] + top_brands_for_index[:5]

    # Filter for peer group (same tier as target)
    peer_brands = [b for b in all_brands_index if b['tier'] == target_tier][:10]

    # Build line chart series - filter to 2023-2024 only
    index_series = []
    all_months = sorted(monthly_prices['month_new'].unique())
    # Filter to 2023 and 2024 only
    months_list = [m for m in all_months if pd.to_datetime(m).year >= 2023]
    month_labels = [str(m)[:7] for m in months_list]

    for brand in top_brands_for_index:
        brand_data = monthly_prices[monthly_prices['brand'].str.upper() == brand.upper()].sort_values('month_new')
        brand_index_values = []
        for m in months_list:
            val = brand_data[brand_data['month_new'] == m]['price_index'].values
            brand_index_values.append(round(val[0], 1) if len(val) > 0 else None)

        color = "#3b82f6" if brand.upper() == brand_filter.upper() else None
        line_width = 3 if brand.upper() == brand_filter.upper() else 1.5
        index_series.append({
            "name": brand,
            "data": brand_index_values,
            "color": color,
            "lineWidth": line_width
        })

    # Add tier band plotBands
    price_index_chart = {
        "chart": {"type": "line", "height": 400},
        "title": {"text": "Price Index Trend (Category Avg = 100)"},
        "xAxis": {
            "categories": month_labels,
            "title": {"text": "Month"},
            "labels": {"rotation": -45}
        },
        "yAxis": {
            "title": {"text": "Price Index"},
            "plotBands": [
                {"from": 0, "to": 80, "color": "rgba(156, 163, 175, 0.1)", "label": {"text": "Value", "style": {"color": "#9ca3af", "fontSize": "10px"}}},
                {"from": 80, "to": 120, "color": "rgba(59, 130, 246, 0.1)", "label": {"text": "Core", "style": {"color": "#3b82f6", "fontSize": "10px"}}},
                {"from": 120, "to": 200, "color": "rgba(34, 197, 94, 0.1)", "label": {"text": "Premium", "style": {"color": "#22c55e", "fontSize": "10px"}}},
                {"from": 200, "to": 400, "color": "rgba(168, 85, 247, 0.1)", "label": {"text": "Super Premium", "style": {"color": "#a855f7", "fontSize": "10px"}}}
            ],
            "plotLines": [{"value": 100, "color": "#666", "dashStyle": "dash", "width": 2, "label": {"text": "Category Avg", "align": "right"}}]
        },
        "tooltip": {"shared": True, "valueSuffix": ""},
        "series": index_series,
        "legend": {"align": "center", "verticalAlign": "bottom"},
        "credits": {"enabled": False}
    }

    # Build price index table by brand with tier and flagging
    index_table_data = [b for b in all_brands_index if b['brand'].upper() in [t.upper() for t in top_brands_for_index]]

    # ===== BASE SIZE MIX ANALYSIS =====
    # Compare prior vs current period index by base_size to detect mix issues

    # Current period base_size index
    current_brand_df = full_df[(full_df['brand'].str.upper() == brand_filter.upper()) &
                                (full_df['month_new'] >= curr_start) &
                                (full_df['month_new'] <= curr_end)]
    prior_brand_df = full_df[(full_df['brand'].str.upper() == brand_filter.upper()) &
                              (full_df['month_new'] >= prior_start) &
                              (full_df['month_new'] <= prior_end)]

    # Current period category avg by base_size
    current_cat_df = full_df[(full_df['month_new'] >= curr_start) & (full_df['month_new'] <= curr_end)]
    prior_cat_df = full_df[(full_df['month_new'] >= prior_start) & (full_df['month_new'] <= prior_end)]

    base_size_analysis = []

    # Get all base_sizes the brand sells
    brand_base_sizes = full_df[full_df['brand'].str.upper() == brand_filter.upper()]['base_size'].unique()

    for bs in brand_base_sizes:
        # Current period
        curr_brand_bs = current_brand_df[current_brand_df['base_size'] == bs]
        curr_cat_bs = current_cat_df[current_cat_df['base_size'] == bs]

        curr_brand_price = curr_brand_bs['total_sales'].sum() / curr_brand_bs['total_units'].sum() if curr_brand_bs['total_units'].sum() > 0 else 0
        curr_cat_price = curr_cat_bs['total_sales'].sum() / curr_cat_bs['total_units'].sum() if curr_cat_bs['total_units'].sum() > 0 else 0
        curr_idx = (curr_brand_price / curr_cat_price * 100) if curr_cat_price > 0 else 0
        curr_vol = curr_brand_bs['total_units'].sum()

        # Prior period
        prior_brand_bs = prior_brand_df[prior_brand_df['base_size'] == bs]
        prior_cat_bs = prior_cat_df[prior_cat_df['base_size'] == bs]

        prior_brand_price = prior_brand_bs['total_sales'].sum() / prior_brand_bs['total_units'].sum() if prior_brand_bs['total_units'].sum() > 0 else 0
        prior_cat_price = prior_cat_bs['total_sales'].sum() / prior_cat_bs['total_units'].sum() if prior_cat_bs['total_units'].sum() > 0 else 0
        prior_idx = (prior_brand_price / prior_cat_price * 100) if prior_cat_price > 0 else 0
        prior_vol = prior_brand_bs['total_units'].sum()

        idx_change = curr_idx - prior_idx
        vol_change = ((curr_vol / prior_vol - 1) * 100) if prior_vol > 0 else 0

        # Calculate mix share
        total_curr_vol = current_brand_df['total_units'].sum()
        total_prior_vol = prior_brand_df['total_units'].sum()
        curr_mix = (curr_vol / total_curr_vol * 100) if total_curr_vol > 0 else 0
        prior_mix = (prior_vol / total_prior_vol * 100) if total_prior_vol > 0 else 0
        mix_change = curr_mix - prior_mix

        if curr_idx > 0 or prior_idx > 0:  # Only include if has data
            base_size_analysis.append({
                'base_size': bs,
                'prior_index': prior_idx,
                'current_index': curr_idx,
                'index_change': idx_change,
                'prior_mix': prior_mix,
                'current_mix': curr_mix,
                'mix_change': mix_change,
                'flagged': abs(idx_change) > 5
            })

    # Sort by current volume mix descending
    base_size_analysis = sorted(base_size_analysis, key=lambda x: x['current_mix'], reverse=True)

    # ===== DIAGNOSTIC INSIGHT =====
    # Determine if index change is due to mix or pricing
    target_data = next((b for b in all_brands_index if b['brand'].upper() == brand_filter.upper()), None)
    target_idx_change = target_data['index_change'] if target_data else 0

    diagnostic_text = ""
    diagnostic_color = "#6b7280"  # gray

    if abs(target_idx_change) > 5:
        # Check if base_sizes maintained their index (mix issue) or changed (pricing issue)
        core_sizes = [bs for bs in base_size_analysis if bs['current_mix'] > 10]  # Top sizes by mix
        core_idx_changes = [bs['index_change'] for bs in core_sizes]
        avg_core_change = sum(core_idx_changes) / len(core_idx_changes) if core_idx_changes else 0

        # Check mix shifts
        mix_shifts = [bs for bs in base_size_analysis if abs(bs['mix_change']) > 5]

        if abs(avg_core_change) < 3 and mix_shifts:
            # Core sizes stable but mix shifted = MIX ISSUE
            diagnostic_text = f"Index {'dropped' if target_idx_change < 0 else 'increased'} by {abs(target_idx_change):.1f} pts. Core pack sizes maintained index - this appears to be a MIX shift (selling {'more lower-priced' if target_idx_change < 0 else 'more premium'} packs)."
            diagnostic_color = "#f59e0b"  # amber
        else:
            # Core sizes also changed = PRICING ISSUE
            # Check peer group behavior
            peer_changes = [p['index_change'] for p in peer_brands if p['brand'].upper() != brand_filter.upper()]
            avg_peer_change = sum(peer_changes) / len(peer_changes) if peer_changes else 0

            if abs(avg_peer_change) > 3 and (avg_peer_change * target_idx_change > 0):
                # Peers moved in same direction = market-wide
                diagnostic_text = f"Index {'dropped' if target_idx_change < 0 else 'increased'} by {abs(target_idx_change):.1f} pts. Similar movement seen in peer group (avg {avg_peer_change:+.1f} pts) - likely market-wide pricing adjustment."
                diagnostic_color = "#3b82f6"  # blue
            else:
                # We moved differently than peers = competitive issue
                if target_idx_change < 0:
                    diagnostic_text = f"Index dropped by {abs(target_idx_change):.1f} pts while {target_tier} peers {'held steady' if abs(avg_peer_change) < 3 else f'moved {avg_peer_change:+.1f} pts'}. Investigate: Did we reduce price? Did competitors raise prices?"
                    diagnostic_color = "#ef4444"  # red
                else:
                    diagnostic_text = f"Index increased by {target_idx_change:.1f} pts while {target_tier} peers {'held steady' if abs(avg_peer_change) < 3 else f'moved {avg_peer_change:+.1f} pts'}. Strong pricing position vs competition."
                    diagnostic_color = "#22c55e"  # green
    else:
        diagnostic_text = f"Index stable (changed {target_idx_change:+.1f} pts). Within normal range of +/-5 pts."
        diagnostic_color = "#22c55e"  # green

    # Build Tab 4 layout
    price_index_layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"padding": "20px"},
            "children": [
                # Target brand tier badge
                {
                    "name": "TierBadge",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "row",
                    "style": {"marginBottom": "20px", "gap": "15px", "alignItems": "center"}
                },
                {"name": "TierLabel", "type": "Paragraph", "children": "", "text": f"{brand_filter} Price Tier:", "parentId": "TierBadge", "style": {"fontWeight": "600", "fontSize": "14px", "color": "#374151"}},
                {"name": "TierValue", "type": "Paragraph", "children": "", "text": target_tier, "parentId": "TierBadge", "style": {"backgroundColor": target_tier_color, "color": "white", "padding": "4px 12px", "borderRadius": "12px", "fontWeight": "600", "fontSize": "12px"}},
                {"name": "TierIndex", "type": "Paragraph", "children": "", "text": f"Index: {target_curr_idx:.0f}", "parentId": "TierBadge", "style": {"fontSize": "13px", "color": "#6b7280"}},

                # Diagnostic insight box
                {
                    "name": "DiagnosticBox",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "style": {"backgroundColor": "#f8fafc", "border": f"2px solid {diagnostic_color}", "borderRadius": "8px", "padding": "16px", "marginBottom": "20px"}
                },
                {"name": "DiagHeader", "type": "Paragraph", "children": "", "text": "Index Diagnostic", "parentId": "DiagnosticBox", "style": {"fontWeight": "bold", "fontSize": "14px", "color": diagnostic_color, "marginBottom": "8px"}},
                {"name": "DiagText", "type": "Paragraph", "children": "", "text": diagnostic_text, "parentId": "DiagnosticBox", "style": {"fontSize": "14px", "lineHeight": "1.5"}},

                # Chart
                {
                    "name": "IndexChart",
                    "type": "HighchartsChart",
                    "children": "",
                    "minHeight": "400px",
                    "options": price_index_chart
                },

                # Peer Group Comparison Header
                {
                    "name": "PeerHeader",
                    "type": "Header",
                    "children": "",
                    "text": f"{target_tier} Tier Peer Comparison",
                    "style": {"fontSize": "18px", "fontWeight": "bold", "marginTop": "30px", "marginBottom": "15px"}
                },

                # Peer Group Table
                {
                    "name": "PeerTable",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "extraStyles": "display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 1fr; gap: 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; font-size: 13px;"
                },
                {"name": "PTH_Brand", "type": "Paragraph", "children": "", "text": "Brand", "parentId": "PeerTable", "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd"}},
                {"name": "PTH_Tier", "type": "Paragraph", "children": "", "text": "Tier", "parentId": "PeerTable", "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "center"}},
                {"name": "PTH_Prior", "type": "Paragraph", "children": "", "text": "Prior", "parentId": "PeerTable", "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "PTH_Current", "type": "Paragraph", "children": "", "text": "Current", "parentId": "PeerTable", "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "PTH_Change", "type": "Paragraph", "children": "", "text": "Chg", "parentId": "PeerTable", "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}}
            ] + [
                item
                for i, row in enumerate(peer_brands[:8])
                for item in [
                    {"name": f"PT{i}_Brand", "type": "Paragraph", "children": "", "text": row['brand'], "parentId": "PeerTable", "style": {"padding": "10px 12px", "fontWeight": "bold" if row['brand'].upper() == brand_filter.upper() else "normal", "borderBottom": "1px solid #eee", "backgroundColor": "#eff6ff" if row['brand'].upper() == brand_filter.upper() else "transparent"}},
                    {"name": f"PT{i}_Tier", "type": "Paragraph", "children": "", "text": row['tier'], "parentId": "PeerTable", "style": {"padding": "10px 12px", "textAlign": "center", "borderBottom": "1px solid #eee", "backgroundColor": "#eff6ff" if row['brand'].upper() == brand_filter.upper() else "transparent", "color": row['tier_color'], "fontWeight": "600"}},
                    {"name": f"PT{i}_Prior", "type": "Paragraph", "children": "", "text": f"{row['prior_index']:.0f}", "parentId": "PeerTable", "style": {"padding": "10px 12px", "textAlign": "right", "borderBottom": "1px solid #eee", "backgroundColor": "#eff6ff" if row['brand'].upper() == brand_filter.upper() else "transparent"}},
                    {"name": f"PT{i}_Current", "type": "Paragraph", "children": "", "text": f"{row['current_index']:.0f}", "parentId": "PeerTable", "style": {"padding": "10px 12px", "textAlign": "right", "borderBottom": "1px solid #eee", "backgroundColor": "#eff6ff" if row['brand'].upper() == brand_filter.upper() else "transparent"}},
                    {"name": f"PT{i}_Change", "type": "Paragraph", "children": "", "text": f"{row['index_change']:+.0f}", "parentId": "PeerTable", "style": {"padding": "10px 12px", "textAlign": "right", "borderBottom": "1px solid #eee", "backgroundColor": "#fef2f2" if row['flagged'] and row['index_change'] < 0 else ("#f0fdf4" if row['flagged'] and row['index_change'] > 0 else ("#eff6ff" if row['brand'].upper() == brand_filter.upper() else "transparent")), "color": "#dc2626" if row['index_change'] < -5 else ("#16a34a" if row['index_change'] > 5 else "#374151"), "fontWeight": "bold" if row['flagged'] else "normal"}}
                ]
            ] + [
                # Base Size Mix Analysis Header
                {
                    "name": "MixHeader",
                    "type": "Header",
                    "children": "",
                    "text": f"Pack Size Mix Analysis ({brand_filter})",
                    "style": {"fontSize": "18px", "fontWeight": "bold", "marginTop": "30px", "marginBottom": "10px"}
                },
                {
                    "name": "MixSubtext",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Compare index changes by pack size to identify if overall index change is due to pricing or mix shift",
                    "style": {"fontSize": "13px", "color": "#6b7280", "marginBottom": "15px"}
                },

                # Base Size Table
                {
                    "name": "MixTable",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "extraStyles": "display: grid; grid-template-columns: 1.8fr 0.8fr 0.8fr 0.8fr 0.8fr 0.8fr 0.8fr; gap: 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; font-size: 12px;"
                },
                {"name": "MTH_Size", "type": "Paragraph", "children": "", "text": "Pack Size", "parentId": "MixTable", "style": {"padding": "10px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd"}},
                {"name": "MTH_PriorIdx", "type": "Paragraph", "children": "", "text": "Prior Idx", "parentId": "MixTable", "style": {"padding": "10px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "MTH_CurrIdx", "type": "Paragraph", "children": "", "text": "Curr Idx", "parentId": "MixTable", "style": {"padding": "10px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "MTH_IdxChg", "type": "Paragraph", "children": "", "text": "Idx Chg", "parentId": "MixTable", "style": {"padding": "10px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "MTH_PriorMix", "type": "Paragraph", "children": "", "text": "Prior Mix", "parentId": "MixTable", "style": {"padding": "10px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "MTH_CurrMix", "type": "Paragraph", "children": "", "text": "Curr Mix", "parentId": "MixTable", "style": {"padding": "10px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}},
                {"name": "MTH_MixChg", "type": "Paragraph", "children": "", "text": "Mix Chg", "parentId": "MixTable", "style": {"padding": "10px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}}
            ] + [
                item
                for i, row in enumerate(base_size_analysis[:10])
                for item in [
                    {"name": f"MT{i}_Size", "type": "Paragraph", "children": "", "text": row['base_size'], "parentId": "MixTable", "style": {"padding": "8px 10px", "borderBottom": "1px solid #eee"}},
                    {"name": f"MT{i}_PriorIdx", "type": "Paragraph", "children": "", "text": f"{row['prior_index']:.0f}", "parentId": "MixTable", "style": {"padding": "8px 10px", "textAlign": "right", "borderBottom": "1px solid #eee"}},
                    {"name": f"MT{i}_CurrIdx", "type": "Paragraph", "children": "", "text": f"{row['current_index']:.0f}", "parentId": "MixTable", "style": {"padding": "8px 10px", "textAlign": "right", "borderBottom": "1px solid #eee"}},
                    {"name": f"MT{i}_IdxChg", "type": "Paragraph", "children": "", "text": f"{row['index_change']:+.0f}", "parentId": "MixTable", "style": {"padding": "8px 10px", "textAlign": "right", "borderBottom": "1px solid #eee", "backgroundColor": "#fef2f2" if row['flagged'] and row['index_change'] < 0 else ("#f0fdf4" if row['flagged'] and row['index_change'] > 0 else "transparent"), "color": "#dc2626" if row['index_change'] < -5 else ("#16a34a" if row['index_change'] > 5 else "#374151"), "fontWeight": "bold" if row['flagged'] else "normal"}},
                    {"name": f"MT{i}_PriorMix", "type": "Paragraph", "children": "", "text": f"{row['prior_mix']:.1f}%", "parentId": "MixTable", "style": {"padding": "8px 10px", "textAlign": "right", "borderBottom": "1px solid #eee"}},
                    {"name": f"MT{i}_CurrMix", "type": "Paragraph", "children": "", "text": f"{row['current_mix']:.1f}%", "parentId": "MixTable", "style": {"padding": "8px 10px", "textAlign": "right", "borderBottom": "1px solid #eee"}},
                    {"name": f"MT{i}_MixChg", "type": "Paragraph", "children": "", "text": f"{row['mix_change']:+.1f}%", "parentId": "MixTable", "style": {"padding": "8px 10px", "textAlign": "right", "borderBottom": "1px solid #eee", "color": "#dc2626" if row['mix_change'] < -5 else ("#16a34a" if row['mix_change'] > 5 else "#374151"), "fontWeight": "bold" if abs(row['mix_change']) > 5 else "normal"}}
                ]
            ]
        },
        "inputVariables": []
    }

    price_index_html = wire_layout(price_index_layout, {})

    # ===== GENERATE LLM INSIGHTS =====
    # Now that we have price index analysis, generate comprehensive insights

    # Format peer comparison for LLM
    peer_summary = []
    for p in peer_brands[:5]:
        if p['brand'].upper() != brand_filter.upper():
            peer_summary.append(f"- {p['brand']}: Index {p['current_index']:.0f} ({p['index_change']:+.0f} pts YoY)")

    # Format mix analysis for LLM
    mix_summary = []
    for bs in base_size_analysis[:5]:
        if abs(bs['index_change']) > 5 or abs(bs['mix_change']) > 5:
            mix_summary.append(f"- {bs['base_size']}: Index {bs['index_change']:+.0f} pts, Mix {bs['mix_change']:+.1f}%")

    # Calculate target brand's sales/volume/share changes (not in competitors_df since it's filtered out)
    target_prior = prior_df[prior_df['brand'].str.upper() == brand_filter.upper()]
    target_current = current_df[current_df['brand'].str.upper() == brand_filter.upper()]

    target_prior_units = target_prior['total_units'].sum() if len(target_prior) > 0 else 0
    target_current_units = target_current['total_units'].sum() if len(target_current) > 0 else 0
    target_prior_sales = target_prior['total_sales'].sum() if len(target_prior) > 0 else 0
    target_current_sales = target_current['total_sales'].sum() if len(target_current) > 0 else 0
    target_prior_price = target_prior_sales / target_prior_units if target_prior_units > 0 else 0
    target_current_price = target_current_sales / target_current_units if target_current_units > 0 else 0
    target_prior_share = (target_prior_units / prior_market_units * 100) if prior_market_units > 0 else 0
    target_current_share = (target_current_units / current_market_units * 100) if current_market_units > 0 else 0

    target_vol_growth = ((target_current_units - target_prior_units) / target_prior_units * 100) if target_prior_units > 0 else 0
    target_share_change = target_current_share - target_prior_share
    target_price_change = ((target_current_price - target_prior_price) / target_prior_price * 100) if target_prior_price > 0 else 0

    # Build facts string for template rendering
    facts = f"""- Brand: {brand_filter}
- Price Tier: {target_tier} (Index: {target_curr_idx:.0f}, {target_idx_change:+.1f} pts YoY)
- {brand_filter} Performance: Volume {target_vol_growth:+.1f}%, Share {target_share_change:+.1f}pp, Price {target_price_change:+.1f}%
- Position vs Competition: {insight_data['weighted_premium']:+.1f}% avg premium
- Competitors cutting prices & gaining: {'; '.join(insight_data['threat_summary'][:3]) if insight_data['threat_summary'] else 'None identified'}
- Pack sizes with significant changes: {'; '.join(mix_summary[:3]) if mix_summary else 'No major shifts'}"""

    # Render max_prompt template with facts for chat response
    try:
        max_response_prompt = jinja2.Template(max_prompt).render(facts=facts)
    except Exception as e:
        print(f"DEBUG: max_prompt template render failed: {e}")
        max_response_prompt = brief_summary

    # Render insight_prompt template with facts for narrative
    try:
        rendered_insight_prompt = jinja2.Template(insight_prompt_template).render(facts=facts)
        detailed_narrative = ar_utils.get_llm_response(rendered_insight_prompt)
        if not detailed_narrative:
            detailed_narrative = f"""## Pricing Strategy Assessment

**1. Pricing Strategy**: {brand_filter} is a {target_tier} tier brand (Index: {target_curr_idx:.0f}). Index changed {target_idx_change:+.1f} pts YoY. {diagnostic_text}

**2. Price vs Volume**: {brand_filter} volume changed {target_vol_growth:+.1f}% while share moved {target_share_change:+.1f}pp. Premium positioning at {insight_data['weighted_premium']:+.1f}% vs competition.

**3. Pack Opportunities**: Review packs where {brand_filter} is significantly over/under-priced vs competition for adjustment opportunities.
"""
    except Exception as e:
        print(f"DEBUG: LLM insight generation failed: {e}")
        detailed_narrative = f"## Pricing Strategy\n\n{brand_filter} operates in the {target_tier} tier (Index: {target_curr_idx:.0f})."

    return SkillOutput(
        final_prompt=max_response_prompt,
        narrative=detailed_narrative,
        visualizations=[
            SkillVisualization(title="Competitive Comparison", layout=html),
            SkillVisualization(title="Competitor Threats", layout=combined_tab2_html),
            SkillVisualization(title="Price Index", layout=price_index_html)
        ],
        parameter_display_descriptions=param_pills
    )


def analyze_price_comparison(df: pd.DataFrame, dimension: str, brand_filter: str = None, filters: list = None, start_date: str = None, end_date: str = None):
    """Compare average prices across dimension values, or competitive comparison if brand filter exists"""

    print(f"DEBUG: analyze_price_comparison called with {len(df)} rows, dimension={dimension}, brand_filter={brand_filter}")

    # If brand filter exists, do competitive comparison
    if brand_filter:
        return analyze_competitive_comparison(df, dimension, brand_filter, filters, start_date, end_date)

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

    # Add insights section
    highest = summary.iloc[-1]  # Last item (highest price)
    lowest = summary.iloc[0]   # First item (lowest price)

    # Handle single vs. multiple items differently
    if len(summary) == 1:
        # Single item - show rich time-series analysis with charts
        print(f"DEBUG: Single item detected, creating time-series dashboard")

        # Use the original df which has monthly data (not the aggregated summary)
        brand_df = df[df[dimension] == highest[dimension]].copy()

        # Convert month_new to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(brand_df['month_new']):
            brand_df['month_new'] = pd.to_datetime(brand_df['month_new'])

        brand_df = brand_df.sort_values('month_new')

        # Calculate monthly price
        brand_df['price'] = brand_df['total_sales'] / brand_df['total_units']

        # Get time series data
        months = brand_df['month_new'].dt.strftime('%b').tolist()  # Jan, Feb, etc
        prices = brand_df['price'].round(2).tolist()
        revenues = (brand_df['total_sales'] / 1000000).round(1).tolist()  # In millions

        # Calculate trends
        price_change = ((prices[-1] - prices[0]) / prices[0] * 100)
        revenue_change = ((revenues[-1] - revenues[0]) / revenues[0] * 100)

        print(f"DEBUG: Time series - {len(months)} months, price trend: {price_change:+.1f}%")

        # Use flat structure with parentId like Price Variance Deep Dive
        print(f"DEBUG: Creating rich dashboard with charts")

        kpi_layout = {
            "layoutJson": {
                "type": "Document",
                "style": {
                    "backgroundColor": "#ffffff",
                    "padding": "20px"
                },
                "children": [
                    {
                        "name": "Header_Title",
                        "type": "Header",
                        "children": "",
                        "text": f"{highest[dimension]} - Pricing Summary",
                        "style": {
                            "fontSize": "24px",
                            "fontWeight": "bold",
                            "marginBottom": "20px"
                        }
                    },
                    {
                        "name": "KPI_Row",
                        "type": "FlexContainer",
                        "children": "",
                        "direction": "row",
                        "extraStyles": "gap: 15px; margin-bottom: 20px;"
                    },
                    {
                        "name": "KPI_Card1",
                        "type": "FlexContainer",
                        "children": "",
                        "direction": "column",
                        "parentId": "KPI_Row",
                        "style": {
                            "padding": "20px",
                            "backgroundColor": "#eff6ff",
                            "borderRadius": "8px",
                            "borderLeft": "4px solid #3b82f6"
                        }
                    },
                    {
                        "name": "KPI1_Label",
                        "type": "Paragraph",
                        "children": "",
                        "text": "Average Price",
                        "parentId": "KPI_Card1",
                        "style": {"fontSize": "14px", "marginBottom": "10px", "color": "#666"}
                    },
                    {
                        "name": "KPI1_Value",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"${highest['avg_price']:.2f}",
                        "parentId": "KPI_Card1",
                        "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#000"}
                    },
                    {
                        "name": "KPI1_Trend",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"{'↑' if price_change > 0 else '↓'} {abs(price_change):.1f}% vs start",
                        "parentId": "KPI_Card1",
                        "style": {
                            "fontSize": "13px",
                            "color": "#10b981" if price_change > 0 else "#ef4444",
                            "marginTop": "5px"
                        }
                    },
                    {
                        "name": "KPI_Card2",
                        "type": "FlexContainer",
                        "children": "",
                        "direction": "column",
                        "parentId": "KPI_Row",
                        "style": {
                            "padding": "20px",
                            "backgroundColor": "#f0fdf4",
                            "borderRadius": "8px",
                            "borderLeft": "4px solid #10b981"
                        }
                    },
                    {
                        "name": "KPI2_Label",
                        "type": "Paragraph",
                        "children": "",
                        "text": "Total Revenue",
                        "parentId": "KPI_Card2",
                        "style": {"fontSize": "14px", "marginBottom": "10px", "color": "#666"}
                    },
                    {
                        "name": "KPI2_Value",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"${highest['total_sales']/1000000:.1f}M",
                        "parentId": "KPI_Card2",
                        "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#000"}
                    },
                    {
                        "name": "KPI2_Trend",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"{'↑' if revenue_change > 0 else '↓'} {abs(revenue_change):.1f}% trend",
                        "parentId": "KPI_Card2",
                        "style": {
                            "fontSize": "13px",
                            "color": "#10b981" if revenue_change > 0 else "#ef4444",
                            "marginTop": "5px"
                        }
                    },
                    {
                        "name": "KPI_Card3",
                        "type": "FlexContainer",
                        "children": "",
                        "direction": "column",
                        "parentId": "KPI_Row",
                        "style": {
                            "padding": "20px",
                            "backgroundColor": "#fef3c7",
                            "borderRadius": "8px",
                            "borderLeft": "4px solid #f59e0b"
                        }
                    },
                    {
                        "name": "KPI3_Label",
                        "type": "Paragraph",
                        "children": "",
                        "text": "Total Units",
                        "parentId": "KPI_Card3",
                        "style": {"fontSize": "14px", "marginBottom": "10px", "color": "#666"}
                    },
                    {
                        "name": "KPI3_Value",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"{highest['total_units']/1000000:.1f}M",
                        "parentId": "KPI_Card3",
                        "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#000"}
                    },
                    {
                        "name": "KPI3_Trend",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"Across {len(months)} months",
                        "parentId": "KPI_Card3",
                        "style": {
                            "fontSize": "13px",
                            "color": "#666",
                            "marginTop": "5px"
                        }
                    },
                    {
                        "name": "PriceTrendChart",
                        "type": "HighchartsChart",
                        "children": "",
                        "minHeight": "350px",
                        "options": {
                            "chart": {"type": "line", "height": 350},
                            "title": {"text": "Price Trend Over Time", "style": {"fontSize": "18px", "fontWeight": "bold"}},
                            "xAxis": {"categories": months, "title": {"text": "Month"}},
                            "yAxis": {"title": {"text": "Price ($)"}, "labels": {"format": "${value}"}},
                            "series": [{
                                "name": "Average Price",
                                "data": prices,
                                "color": "#3b82f6",
                                "marker": {"enabled": True, "radius": 4}
                            }],
                            "tooltip": {
                                "valuePrefix": "$",
                                "valueDecimals": 2,
                                "backgroundColor": "#ffffff",
                                "borderColor": "#999",
                                "borderWidth": 2,
                                "borderRadius": 4,
                                "shadow": True,
                                "style": {
                                    "opacity": 1
                                }
                            },
                            "credits": {"enabled": False}
                        }
                    },
                    {
                        "name": "RevenueTrendChart",
                        "type": "HighchartsChart",
                        "children": "",
                        "minHeight": "350px",
                        "options": {
                            "chart": {"type": "area", "height": 350},
                            "title": {"text": "Revenue Trend Over Time", "style": {"fontSize": "18px", "fontWeight": "bold"}},
                            "xAxis": {"categories": months, "title": {"text": "Month"}},
                            "yAxis": {"title": {"text": "Revenue ($M)"}, "labels": {"format": "${value}M"}},
                            "series": [{
                                "name": "Monthly Revenue",
                                "data": revenues,
                                "color": "#10b981",
                                "fillOpacity": 0.3
                            }],
                            "tooltip": {
                                "valueSuffix": "M",
                                "valuePrefix": "$",
                                "backgroundColor": "#ffffff",
                                "borderColor": "#999",
                                "borderWidth": 2,
                                "borderRadius": 4,
                                "shadow": True,
                                "style": {
                                    "opacity": 1
                                }
                            },
                            "credits": {"enabled": False}
                        }
                    }
                ]
            },
            "inputVariables": []
        }

        # Use wire_layout with the structured JSON layout
        print(f"DEBUG: Calling wire_layout with structured JSON layout")
        try:
            full_html = wire_layout(kpi_layout, {})
            print(f"DEBUG: wire_layout successful, HTML length: {len(full_html)}")
        except Exception as e:
            print(f"DEBUG: wire_layout failed: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            # Fallback to simple text
            full_html = f"<p>{highest[dimension]}: ${highest['avg_price']:.2f}, Revenue: ${highest['total_sales']:,.0f}, Units: {highest['total_units']:,.0f}</p>"

    else:
        # Multiple items - use Highcharts bar chart
        print(f"DEBUG: Multiple items detected, creating Highcharts bar chart")

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

        # Calculate market-level KPIs
        total_market_revenue = summary['total_sales'].sum()
        total_market_units = summary['total_units'].sum()
        num_items = len(summary)

        # Use structured layout with banner, KPI cards, chart, and insights
        comparison_layout = {
            "layoutJson": {
                "type": "Document",
                "style": {"backgroundColor": "#ffffff", "padding": "20px"},
                "children": [
                    # Banner
                    {
                        "name": "Banner",
                        "type": "FlexContainer",
                        "children": "",
                        "direction": "column",
                        "style": {
                            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                            "padding": "30px",
                            "borderRadius": "12px",
                            "marginBottom": "25px",
                            "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
                        }
                    },
                    {
                        "name": "BannerTitle",
                        "type": "Header",
                        "children": "",
                        "text": f"Pricing Analysis: {dimension.replace('_', ' ').title()}",
                        "parentId": "Banner",
                        "style": {"fontSize": "28px", "fontWeight": "bold", "color": "white", "marginBottom": "10px"}
                    },
                    {
                        "name": "BannerSubtitle",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"Comparing {num_items} {dimension.replace('_', ' ')} values across the market",
                        "parentId": "Banner",
                        "style": {"fontSize": "16px", "color": "rgba(255,255,255,0.9)"}
                    },
                    # KPI Cards Row
                    {
                        "name": "KPI_Row",
                        "type": "FlexContainer",
                        "children": "",
                        "direction": "row",
                        "extraStyles": "gap: 15px; margin-bottom: 25px;"
                    },
                    # KPI Card 1: Average Price
                    {
                        "name": "KPI_Card1",
                        "type": "FlexContainer",
                        "children": "",
                        "direction": "column",
                        "parentId": "KPI_Row",
                        "style": {
                            "flex": "1",
                            "padding": "20px",
                            "backgroundColor": "#e3f2fd",
                            "borderLeft": "4px solid #2196f3",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                        }
                    },
                    {
                        "name": "KPI1_Label",
                        "type": "Paragraph",
                        "children": "",
                        "text": "Average Market Price",
                        "parentId": "KPI_Card1",
                        "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                    },
                    {
                        "name": "KPI1_Value",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"${overall_avg:.2f}",
                        "parentId": "KPI_Card1",
                        "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#1976d2"}
                    },
                    # KPI Card 2: Total Revenue
                    {
                        "name": "KPI_Card2",
                        "type": "FlexContainer",
                        "children": "",
                        "direction": "column",
                        "parentId": "KPI_Row",
                        "style": {
                            "flex": "1",
                            "padding": "20px",
                            "backgroundColor": "#e8f5e9",
                            "borderLeft": "4px solid #4caf50",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                        }
                    },
                    {
                        "name": "KPI2_Label",
                        "type": "Paragraph",
                        "children": "",
                        "text": "Total Market Revenue",
                        "parentId": "KPI_Card2",
                        "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                    },
                    {
                        "name": "KPI2_Value",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"${total_market_revenue/1000000:.1f}M",
                        "parentId": "KPI_Card2",
                        "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#388e3c"}
                    },
                    # KPI Card 3: Total Units
                    {
                        "name": "KPI_Card3",
                        "type": "FlexContainer",
                        "children": "",
                        "direction": "column",
                        "parentId": "KPI_Row",
                        "style": {
                            "flex": "1",
                            "padding": "20px",
                            "backgroundColor": "#fff3e0",
                            "borderLeft": "4px solid #ff9800",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                        }
                    },
                    {
                        "name": "KPI3_Label",
                        "type": "Paragraph",
                        "children": "",
                        "text": "Total Market Units",
                        "parentId": "KPI_Card3",
                        "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                    },
                    {
                        "name": "KPI3_Value",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"{total_market_units/1000000:.1f}M",
                        "parentId": "KPI_Card3",
                        "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#f57c00"}
                    },
                    # KPI Card 4: Number of Items
                    {
                        "name": "KPI_Card4",
                        "type": "FlexContainer",
                        "children": "",
                        "direction": "column",
                        "parentId": "KPI_Row",
                        "style": {
                            "flex": "1",
                            "padding": "20px",
                            "backgroundColor": "#fce4ec",
                            "borderLeft": "4px solid #e91e63",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                        }
                    },
                    {
                        "name": "KPI4_Label",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"{dimension.replace('_', ' ').title()} Values",
                        "parentId": "KPI_Card4",
                        "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                    },
                    {
                        "name": "KPI4_Value",
                        "type": "Paragraph",
                        "children": "",
                        "text": f"{num_items}",
                        "parentId": "KPI_Card4",
                        "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#c2185b"}
                    },
                    # Comparison Chart
                    {
                        "name": "ComparisonChart",
                        "type": "HighchartsChart",
                        "children": "",
                        "minHeight": "500px",
                        "options": chart_config
                    }
                ]
            },
            "inputVariables": []
        }

        print(f"DEBUG: Creating structured comparison layout, {len(categories)} categories")
        try:
            full_html = wire_layout(comparison_layout, {})
            print(f"DEBUG: wire_layout successful, HTML length: {len(full_html)}")
        except Exception as e:
            print(f"DEBUG: wire_layout failed: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            # Fallback to simple message
            fallback_layout = {
                "layoutJson": {
                    "type": "Document",
                    "children": [{"type": "Paragraph", "children": "", "text": f"Error rendering chart: {str(e)}"}]
                },
                "inputVariables": []
            }
            full_html = wire_layout(fallback_layout, {})

    # Generate detailed narrative using LLM
    print(f"DEBUG: Generating narrative for {len(summary)} {dimension} value(s)")

    if len(summary) == 1:
        # Single item - simpler narrative
        brief_summary = f"{highest[dimension]} has an average price of ${highest['avg_price']:.2f}."
        detailed_narrative = f"""## Pricing Summary for {highest[dimension]}

**Key Metrics:**
- Average Price: ${highest['avg_price']:.2f}
- Total Revenue: ${highest['total_sales']:,.0f}
- Units Sold: {highest['total_units']:,.0f}

**Next Steps:**
Remove the {dimension} filter to compare {highest[dimension]} against other {dimension} values in the market.
"""
        print(f"DEBUG: Using simple narrative for single item")
    else:
        # Multiple items - use LLM for richer narrative
        ar_utils = ArUtils()

        # Create data summary for LLM
        top_3 = summary.nlargest(3, 'avg_price')
        bottom_3 = summary.nsmallest(3, 'avg_price')

        narrative_prompt = f"""Analyze this pricing comparison data and provide detailed insights:

**Market Overview:**
- Dimension: {dimension.replace('_', ' ').title()}
- Number of {dimension} values analyzed: {len(summary)}
- Market average price: ${overall_avg:.2f}
- Total market revenue: ${total_market_revenue:,.0f}
- Total market units: {total_market_units:,.0f}

**Price Leaders (Highest):**
{chr(10).join([f"- {row[dimension]}: ${row['avg_price']:.2f} ({row['price_vs_avg']:+.1f}% vs market)" for _, row in top_3.iterrows()])}

**Value Segment (Lowest):**
{chr(10).join([f"- {row[dimension]}: ${row['avg_price']:.2f} ({row['price_vs_avg']:+.1f}% vs market)" for _, row in bottom_3.iterrows()])}

**Price Distribution:**
- Price range: ${lowest['avg_price']:.2f} to ${highest['avg_price']:.2f}
- Price spread: ${(highest['avg_price'] - lowest['avg_price']):.2f} ({((highest['avg_price'] / lowest['avg_price'] - 1) * 100):.1f}% difference)

Provide a comprehensive analysis with the following sections:
1. **Market Structure**: What does the pricing spread tell us about market segmentation?
2. **Positioning Insights**: How do premium vs value segments compare?
3. **Opportunities**: What pricing opportunities or risks do you see?
4. **Recommendations**: Strategic recommendations for pricing optimization.

Use markdown formatting with clear headers and bullet points. **Limit response to 250 words maximum.**"""

        print(f"DEBUG: Calling ArUtils.get_llm_response for detailed narrative")
        try:
            detailed_narrative = ar_utils.get_llm_response(narrative_prompt)
            if not detailed_narrative:
                detailed_narrative = f"""## Price Comparison Analysis

**Market Overview:**
- Analyzed {len(summary)} {dimension.replace('_', ' ')} values
- Market average: ${overall_avg:.2f}
- Price range: ${lowest['avg_price']:.2f} to ${highest['avg_price']:.2f}

**Key Findings:**
- {highest[dimension]} commands the highest price at ${highest['avg_price']:.2f} (+{highest['price_vs_avg']:.1f}% vs market)
- {lowest[dimension]} has the lowest price at ${lowest['avg_price']:.2f} ({lowest['price_vs_avg']:.1f}% vs market)
- Price spread of ${(highest['avg_price'] - lowest['avg_price']):.2f} suggests significant market segmentation
"""
            brief_summary = f"Price analysis shows {len(summary)} {dimension.replace('_', ' ')} values ranging from ${lowest['avg_price']:.2f} to ${highest['avg_price']:.2f}."
            print(f"DEBUG: LLM narrative generated, length: {len(detailed_narrative)}")
        except Exception as e:
            print(f"DEBUG: LLM narrative failed: {e}")
            detailed_narrative = f"""## Price Comparison Analysis

**Market Overview:**
- Analyzed {len(summary)} {dimension.replace('_', ' ')} values
- Market average: ${overall_avg:.2f}
- Price range: ${lowest['avg_price']:.2f} to ${highest['avg_price']:.2f}
"""
            brief_summary = f"Price analysis shows {len(summary)} {dimension.replace('_', ' ')} values."

    print(f"DEBUG: Narrative preview: {brief_summary[:100]}...")

    # Create parameter pills
    param_pills = [
        ParameterDisplayDescription(key="dimension", value=f"Dimension: {dimension.replace('_', ' ').title()}"),
        ParameterDisplayDescription(key="items_analyzed", value=f"Items: {len(summary)}"),
    ]

    if len(summary) > 1:
        param_pills.extend([
            ParameterDisplayDescription(key="avg_price", value=f"Avg Price: ${overall_avg:.2f}"),
            ParameterDisplayDescription(key="price_range", value=f"Range: ${lowest['avg_price']:.2f} - ${highest['avg_price']:.2f}"),
        ])

    # Add filter pills for any additional filters
    if filters:
        for filter_item in filters:
            if isinstance(filter_item, dict):
                dim = filter_item.get('dim')
                values = filter_item.get('val')
                if dim and values:
                    if isinstance(values, list):
                        values_display = ', '.join(str(v) for v in values)
                    else:
                        values_display = str(values)
                    param_pills.append(
                        ParameterDisplayDescription(
                            key=f"filter_{dim}",
                            value=f"{dim.replace('_', ' ').title()}: {values_display}"
                        )
                    )

    return SkillOutput(
        final_prompt=brief_summary,
        narrative=detailed_narrative,
        visualizations=[
            SkillVisualization(title="Price Comparison", layout=full_html)
        ],
        parameter_display_descriptions=param_pills
    )


def analyze_price_elasticity(df: pd.DataFrame, dimension: str):
    """Calculate price elasticity for each dimension value"""

    print(f"DEBUG: analyze_price_elasticity called with {len(df)} rows, dimension={dimension}")
    print(f"DEBUG: Unique {dimension} values: {df[dimension].unique()}")

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

    print(f"DEBUG: Calculated elasticity for {len(results)} {dimension} values")

    if not results:
        html = "<p>Insufficient data to calculate price elasticity. Need more price variation over time.</p>"
        return SkillOutput(
            final_prompt="Unable to calculate price elasticity with current data.",
            narrative=None,
            visualizations=[SkillVisualization(title="Price Elasticity", layout=html)]
        )

    elasticity_df = pd.DataFrame(results).sort_values('elasticity')

    # Calculate elasticity segments
    num_elastic = len(elasticity_df[elasticity_df['elasticity'].abs() > 1])
    num_moderate = len(elasticity_df[(elasticity_df['elasticity'].abs() > 0.5) & (elasticity_df['elasticity'].abs() <= 1)])
    num_inelastic = len(elasticity_df[elasticity_df['elasticity'].abs() <= 0.5])

    # Build table rows
    table_rows = []
    for idx, (_, row) in enumerate(elasticity_df.head(15).iterrows()):
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

        table_rows.extend([
            {
                "name": f"Row{idx}_Name",
                "type": "Paragraph",
                "children": "",
                "text": str(row[dimension]),
                "parentId": "ElasticityTable",
                "style": {"padding": "12px", "fontWeight": "bold"}
            },
            {
                "name": f"Row{idx}_Elasticity",
                "type": "Paragraph",
                "children": "",
                "text": f"{elasticity:.2f}",
                "parentId": "ElasticityTable",
                "style": {"padding": "12px", "textAlign": "right", "color": color, "fontWeight": "bold"}
            },
            {
                "name": f"Row{idx}_Interpretation",
                "type": "Paragraph",
                "children": "",
                "text": interpretation,
                "parentId": "ElasticityTable",
                "style": {"padding": "12px", "color": color}
            },
            {
                "name": f"Row{idx}_Price",
                "type": "Paragraph",
                "children": "",
                "text": f"${row['avg_price']:.2f}",
                "parentId": "ElasticityTable",
                "style": {"padding": "12px", "textAlign": "right"}
            }
        ])

    # Create structured layout
    elasticity_layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"backgroundColor": "#ffffff", "padding": "20px"},
            "children": [
                # Banner
                {
                    "name": "Banner",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "style": {
                        "background": "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
                        "padding": "30px",
                        "borderRadius": "12px",
                        "marginBottom": "25px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
                    }
                },
                {
                    "name": "BannerTitle",
                    "type": "Header",
                    "children": "",
                    "text": "Price Elasticity Analysis",
                    "parentId": "Banner",
                    "style": {"fontSize": "28px", "fontWeight": "bold", "color": "#333", "marginBottom": "10px"}
                },
                {
                    "name": "BannerSubtitle",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Elasticity measures how demand changes with price. Negative = normal (price up, demand down)",
                    "parentId": "Banner",
                    "style": {"fontSize": "16px", "color": "#555"}
                },
                # KPI Cards Row
                {
                    "name": "KPI_Row",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "row",
                    "extraStyles": "gap: 15px; margin-bottom: 25px;"
                },
                # KPI Card 1: Elastic
                {
                    "name": "KPI_Card1",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#f8d7da",
                        "borderLeft": "4px solid #dc3545",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI1_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Elastic (|E| > 1)",
                    "parentId": "KPI_Card1",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI1_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{num_elastic}",
                    "parentId": "KPI_Card1",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#dc3545"}
                },
                # KPI Card 2: Moderate
                {
                    "name": "KPI_Card2",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#fff3cd",
                        "borderLeft": "4px solid #ffc107",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI2_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Moderate (0.5-1)",
                    "parentId": "KPI_Card2",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI2_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{num_moderate}",
                    "parentId": "KPI_Card2",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#ffc107"}
                },
                # KPI Card 3: Inelastic
                {
                    "name": "KPI_Card3",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#d4edda",
                        "borderLeft": "4px solid #28a745",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI3_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Inelastic (|E| < 0.5)",
                    "parentId": "KPI_Card3",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI3_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{num_inelastic}",
                    "parentId": "KPI_Card3",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#28a745"}
                },
                # Table Section
                {
                    "name": "TableTitle",
                    "type": "Header",
                    "children": "",
                    "text": "Elasticity by Product",
                    "style": {"fontSize": "20px", "fontWeight": "bold", "marginBottom": "15px"}
                },
                {
                    "name": "ElasticityTable",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "extraStyles": "display: grid; grid-template-columns: 2fr 1fr 2fr 1fr; gap: 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; margin-bottom: 25px;"
                },
                # Table Headers
                {
                    "name": "Header_Name",
                    "type": "Paragraph",
                    "children": "",
                    "text": dimension.replace('_', ' ').title(),
                    "parentId": "ElasticityTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd"}
                },
                {
                    "name": "Header_Elasticity",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Elasticity",
                    "parentId": "ElasticityTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}
                },
                {
                    "name": "Header_Interpretation",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Interpretation",
                    "parentId": "ElasticityTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd"}
                },
                {
                    "name": "Header_Price",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Avg Price",
                    "parentId": "ElasticityTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}
                }
            ] + table_rows + [
                # Guide Section
                {
                    "name": "GuideContainer",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "style": {
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "borderLeft": "4px solid #007bff",
                        "borderRadius": "8px"
                    }
                },
                {
                    "name": "GuideTitle",
                    "type": "Header",
                    "children": "",
                    "text": "Price Elasticity Guide",
                    "parentId": "GuideContainer",
                    "style": {"fontSize": "18px", "fontWeight": "bold", "marginBottom": "15px", "marginTop": "0"}
                },
                {
                    "name": "Guide1",
                    "type": "Paragraph",
                    "children": "",
                    "text": "• Elastic (|E| > 1): Demand very sensitive to price. Small price increase → large demand drop.",
                    "parentId": "GuideContainer",
                    "style": {"fontSize": "15px", "marginBottom": "8px"}
                },
                {
                    "name": "Guide2",
                    "type": "Paragraph",
                    "children": "",
                    "text": "• Inelastic (|E| < 1): Demand less sensitive to price. Price increases have limited impact on volume.",
                    "parentId": "GuideContainer",
                    "style": {"fontSize": "15px", "marginBottom": "8px"}
                },
                {
                    "name": "Guide3",
                    "type": "Paragraph",
                    "children": "",
                    "text": "• Recommendation: Increase prices on inelastic products, be cautious with elastic products.",
                    "parentId": "GuideContainer",
                    "style": {"fontSize": "15px", "fontWeight": "bold"}
                }
            ]
        },
        "inputVariables": []
    }

    html = wire_layout(elasticity_layout, {})

    return SkillOutput(
        final_prompt="Price elasticity analysis shows which products are most sensitive to price changes.",
        narrative=None,
        visualizations=[SkillVisualization(title="Price Elasticity", layout=html)]
    )


def analyze_optimization_opportunities(df: pd.DataFrame, dimension: str):
    """Identify pricing optimization opportunities"""

    print(f"DEBUG: analyze_optimization_opportunities called with {len(df)} rows, dimension={dimension}")

    # Calculate metrics by dimension
    summary = df.groupby(dimension).agg({
        'total_sales': 'sum',
        'total_units': 'sum',
        'total_volume': 'sum'
    }).reset_index()

    print(f"DEBUG: Grouped by {dimension}, got {len(summary)} unique values")

    summary['avg_price'] = summary['total_sales'] / summary['total_units']
    summary['revenue_per_unit'] = summary['total_sales'] / summary['total_units']

    # Check if we have enough data for comparison
    if len(summary) < 2:
        print(f"DEBUG: Only {len(summary)} unique value(s) - cannot compare across {dimension}")

        if len(summary) == 1:
            # Use price_comparison for single brand - it has the rich dashboard!
            print(f"DEBUG: Redirecting to price_comparison for single-brand rich dashboard")
            return analyze_price_comparison(df, dimension)
        else:
            # No data at all - use simple structured layout
            no_data_layout = {
                "layoutJson": {
                    "type": "Document",
                    "style": {"padding": "20px"},
                    "children": [{
                        "type": "Paragraph",
                        "children": "",
                        "text": "No data available for optimization analysis. Try removing filters.",
                        "style": {"fontSize": "16px"}
                    }]
                },
                "inputVariables": []
            }
            html = wire_layout(no_data_layout, {})
            narrative = "No data available. Try removing filters."

            return SkillOutput(
                final_prompt=narrative,
                narrative=None,
                visualizations=[SkillVisualization(title="Pricing Summary", layout=html)]
            )

    # Calculate price per volume for normalization across different pack sizes
    summary['price_per_volume'] = summary['total_sales'] / summary['total_volume']
    summary['avg_price'] = summary['total_sales'] / summary['total_units']

    # Use price per volume as the comparison metric (normalizes across pack sizes)
    print(f"DEBUG: Using price_per_volume to normalize across {dimension} values")

    # Calculate percentiles on price per volume
    p25 = summary['price_per_volume'].quantile(0.25)
    p75 = summary['price_per_volume'].quantile(0.75)
    median = summary['price_per_volume'].median()

    print(f"DEBUG: Price per volume stats - p25: ${p25:.2f}, median: ${median:.2f}, p75: ${p75:.2f}")

    # Identify opportunities - products with low price/volume AND high sales volume
    summary['opportunity'] = summary.apply(lambda row:
        'Price Increase Potential' if row['price_per_volume'] < p25 and row['total_units'] > summary['total_units'].median()
        else 'Premium Positioning' if row['price_per_volume'] > p75
        else 'Well Positioned' if p25 <= row['price_per_volume'] <= p75
        else 'Monitor', axis=1
    )

    # Calculate lift: bring low-priced items up to p25 (conservative target)
    # Multiply by total_volume since we're using price_per_volume
    summary['potential_revenue_lift'] = summary.apply(lambda row:
        (p25 - row['price_per_volume']) * row['total_volume'] if row['opportunity'] == 'Price Increase Potential'
        else 0, axis=1
    )

    summary = summary.sort_values('potential_revenue_lift', ascending=False)

    opportunities = summary[summary['opportunity'] == 'Price Increase Potential'].head(10)
    total_lift = opportunities['potential_revenue_lift'].sum() if len(opportunities) > 0 else 0

    # Calculate market-level KPIs
    total_market_revenue = summary['total_sales'].sum()
    total_market_units = summary['total_units'].sum()
    num_items = len(summary)
    num_opportunities = len(opportunities)

    # Build table rows for opportunities
    table_rows = []
    if len(opportunities) == 0:
        table_rows.append({
            "name": f"NoOpps",
            "type": "Paragraph",
            "children": "",
            "text": "No clear price increase opportunities identified",
            "parentId": "OpportunityTable",
            "style": {"padding": "12px", "textAlign": "center", "color": "#666", "gridColumn": "1 / -1"}
        })
    else:
        for idx, (_, row) in enumerate(opportunities.iterrows()):
            table_rows.extend([
                {
                    "name": f"Row{idx}_Name",
                    "type": "Paragraph",
                    "children": "",
                    "text": str(row[dimension]),
                    "parentId": "OpportunityTable",
                    "style": {"padding": "12px", "fontWeight": "bold"}
                },
                {
                    "name": f"Row{idx}_Current",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"${row['price_per_volume']:.2f}/vol",
                    "parentId": "OpportunityTable",
                    "style": {"padding": "12px", "textAlign": "right"}
                },
                {
                    "name": f"Row{idx}_Target",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"${p25:.2f}/vol",
                    "parentId": "OpportunityTable",
                    "style": {"padding": "12px", "textAlign": "right"}
                },
                {
                    "name": f"Row{idx}_Units",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{row['total_units']:,.0f}",
                    "parentId": "OpportunityTable",
                    "style": {"padding": "12px", "textAlign": "right"}
                },
                {
                    "name": f"Row{idx}_Lift",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"${row['potential_revenue_lift']:,.0f}",
                    "parentId": "OpportunityTable",
                    "style": {"padding": "12px", "textAlign": "right", "color": "#28a745", "fontWeight": "bold"}
                }
            ])

    # Recommendation text
    if len(opportunities) > 0:
        recommendation = f"Consider gradual price increases for underpriced high-volume products. Potential revenue lift: ${total_lift:,.0f}. Next Steps: Test 5-10% price increases, monitor volume impact, adjust based on customer response."
    else:
        recommendation = "Pricing appears well-optimized relative to market. Focus on maintaining position and monitoring competition."

    # Create structured layout with banner, KPI cards, table, and insights
    optimization_layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"backgroundColor": "#ffffff", "padding": "20px"},
            "children": [
                # Banner
                {
                    "name": "Banner",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "style": {
                        "background": "linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%)",
                        "padding": "30px",
                        "borderRadius": "12px",
                        "marginBottom": "25px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
                    }
                },
                {
                    "name": "BannerTitle",
                    "type": "Header",
                    "children": "",
                    "text": f"Pricing Optimization Opportunities",
                    "parentId": "Banner",
                    "style": {"fontSize": "28px", "fontWeight": "bold", "color": "white", "marginBottom": "10px"}
                },
                {
                    "name": "BannerSubtitle",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"Analyzing {num_items} {dimension.replace('_', ' ')} values • Market median: ${median:.2f}",
                    "parentId": "Banner",
                    "style": {"fontSize": "16px", "color": "rgba(255,255,255,0.9)"}
                },
                # KPI Cards Row
                {
                    "name": "KPI_Row",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "row",
                    "extraStyles": "gap: 15px; margin-bottom: 25px;"
                },
                # KPI Card 1: Median Price
                {
                    "name": "KPI_Card1",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#e3f2fd",
                        "borderLeft": "4px solid #2196f3",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI1_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Market Median Price",
                    "parentId": "KPI_Card1",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI1_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"${median:.2f}",
                    "parentId": "KPI_Card1",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#1976d2"}
                },
                # KPI Card 2: Total Revenue
                {
                    "name": "KPI_Card2",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#e8f5e9",
                        "borderLeft": "4px solid #4caf50",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI2_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Total Revenue",
                    "parentId": "KPI_Card2",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI2_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"${total_market_revenue/1000000:.1f}M",
                    "parentId": "KPI_Card2",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#388e3c"}
                },
                # KPI Card 3: Opportunities
                {
                    "name": "KPI_Card3",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#fff3e0",
                        "borderLeft": "4px solid #ff9800",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI3_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Opportunities Found",
                    "parentId": "KPI_Card3",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI3_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{num_opportunities}",
                    "parentId": "KPI_Card3",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#f57c00"}
                },
                # KPI Card 4: Potential Lift
                {
                    "name": "KPI_Card4",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#fce4ec",
                        "borderLeft": "4px solid #e91e63",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI4_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Potential Revenue Lift",
                    "parentId": "KPI_Card4",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI4_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"${total_lift/1000000:.2f}M" if total_lift > 1000000 else f"${total_lift:,.0f}",
                    "parentId": "KPI_Card4",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#c2185b"}
                },
                # Table Section
                {
                    "name": "TableTitle",
                    "type": "Header",
                    "children": "",
                    "text": "Price Increase Opportunities",
                    "style": {"fontSize": "20px", "fontWeight": "bold", "marginBottom": "15px"}
                },
                {
                    "name": "OpportunityTable",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "extraStyles": f"display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 1fr; gap: 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; margin-bottom: 25px;"
                },
                # Table Headers
                {
                    "name": "Header_Name",
                    "type": "Paragraph",
                    "children": "",
                    "text": dimension.replace('_', ' ').title(),
                    "parentId": "OpportunityTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd"}
                },
                {
                    "name": "Header_Current",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Current $/Vol",
                    "parentId": "OpportunityTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}
                },
                {
                    "name": "Header_Target",
                    "type": "Paragraph",
                    "children": "",
                    "text": "P25 Target",
                    "parentId": "OpportunityTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}
                },
                {
                    "name": "Header_Units",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Units",
                    "parentId": "OpportunityTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}
                },
                {
                    "name": "Header_Lift",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Potential Lift",
                    "parentId": "OpportunityTable",
                    "style": {"padding": "12px", "backgroundColor": "#f5f5f5", "fontWeight": "bold", "borderBottom": "2px solid #ddd", "textAlign": "right"}
                }
            ] + table_rows + [
                # Recommendation Section
                {
                    "name": "RecommendationContainer",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "style": {
                        "padding": "20px",
                        "backgroundColor": "#fff3cd",
                        "borderLeft": "4px solid #ffc107",
                        "borderRadius": "8px"
                    }
                },
                {
                    "name": "RecommendationTitle",
                    "type": "Header",
                    "children": "",
                    "text": "💡 Recommendation",
                    "parentId": "RecommendationContainer",
                    "style": {"fontSize": "18px", "fontWeight": "bold", "marginBottom": "10px", "marginTop": "0"}
                },
                {
                    "name": "RecommendationText",
                    "type": "Paragraph",
                    "children": "",
                    "text": recommendation,
                    "parentId": "RecommendationContainer",
                    "style": {"fontSize": "15px", "lineHeight": "1.6", "marginBottom": "0"}
                }
            ]
        },
        "inputVariables": []
    }

    print(f"DEBUG: Creating optimization layout with {num_opportunities} opportunities")
    html = wire_layout(optimization_layout, {})

    # Generate LLM insights
    ar_utils = ArUtils()

    if num_opportunities > 0:
        top_opportunities = opportunities.nlargest(3, 'potential_revenue_lift')

        insight_prompt = f"""Analyze this pricing optimization analysis and provide strategic recommendations:

**Market Context:**
- Dimension: {dimension.replace('_', ' ').title()}
- Items analyzed: {num_items}
- Market median price: ${median:.2f}
- Total revenue: ${total_market_revenue:,.0f}

**Optimization Opportunities Found: {num_opportunities}**
{chr(10).join([f"- {row[dimension]}: Currently ${row['avg_price']:.2f} (vs median ${median:.2f}), {row['total_units']:,.0f} units → Potential lift ${row['potential_revenue_lift']:,.0f}" for _, row in top_opportunities.iterrows()])}

**Total Potential Revenue Lift: ${total_lift:,.0f}**

Provide a comprehensive analysis with:
1. **Opportunity Assessment**: Which products have the most potential and why?
2. **Risk Analysis**: What are the risks of raising prices on these high-volume products?
3. **Implementation Strategy**: How should price increases be rolled out?
4. **Success Metrics**: What KPIs should be tracked?

Use markdown formatting with clear headers and bullet points. **Limit response to 250 words maximum.**"""
    else:
        insight_prompt = f"""Analyze this pricing optimization analysis:

**Market Context:**
- Dimension: {dimension.replace('_', ' ').title()}
- Items analyzed: {num_items}
- Market median price: ${median:.2f}
- Total revenue: ${total_market_revenue:,.0f}

**Finding:** No clear price increase opportunities were identified based on current pricing vs market median.

Provide analysis on:
1. Why no opportunities were found
2. What this suggests about current pricing strategy
3. Alternative optimization approaches to consider

Use markdown formatting. **Limit response to 250 words maximum.**"""

    try:
        detailed_narrative = ar_utils.get_llm_response(insight_prompt)
        if not detailed_narrative:
            if num_opportunities > 0:
                detailed_narrative = f"""## Pricing Optimization Analysis

**Opportunities Found: {num_opportunities}**

Potential revenue lift of ${total_lift:,.0f} identified through strategic price increases for underpriced, high-volume products.

**Top Opportunities:**
{chr(10).join([f"- **{row[dimension]}**: Raise from ${row['avg_price']:.2f} to ${median:.2f} → ${row['potential_revenue_lift']:,.0f} lift" for _, row in top_opportunities.iterrows()])}

**Recommendation:** Test 5-10% price increases, monitor volume impact, adjust based on customer response.
"""
            else:
                detailed_narrative = "## Pricing Optimization Analysis\n\nNo clear price increase opportunities identified. Current pricing appears well-optimized relative to market median."
    except Exception as e:
        print(f"DEBUG: LLM insight generation failed: {e}")
        detailed_narrative = "## Pricing Optimization Analysis\n\nAnalysis complete. See recommendations below for next steps."

    brief_summary = f"Found {num_opportunities} pricing optimization opportunities with ${total_lift:,.0f} potential revenue lift." if num_opportunities > 0 else "No clear pricing optimization opportunities identified."

    # Create pills
    param_pills = [
        ParameterDisplayDescription(key="dimension", value=f"Dimension: {dimension.replace('_', ' ').title()}"),
        ParameterDisplayDescription(key="items", value=f"Items: {num_items}"),
        ParameterDisplayDescription(key="median_price", value=f"Median: ${median:.2f}"),
        ParameterDisplayDescription(key="opportunities", value=f"Opportunities: {num_opportunities}"),
    ]

    if total_lift > 0:
        param_pills.append(
            ParameterDisplayDescription(key="potential_lift", value=f"Potential Lift: ${total_lift/1000000:.1f}M" if total_lift > 1000000 else f"${total_lift:,.0f}")
        )

    return SkillOutput(
        final_prompt=brief_summary,
        narrative=detailed_narrative,
        visualizations=[SkillVisualization(title="Optimization Opportunities", layout=html)],
        parameter_display_descriptions=param_pills
    )


def analyze_what_if_scenario(df: pd.DataFrame, dimension: str, price_change_pct: float):
    """Simulate revenue impact of price changes"""

    print(f"DEBUG: analyze_what_if_scenario called with {len(df)} rows, dimension={dimension}, price_change_pct={price_change_pct}")

    # Assume moderate elasticity of -0.7 (industry typical for CPG)
    assumed_elasticity = -0.7

    summary = df.groupby(dimension).agg({
        'total_sales': 'sum',
        'total_units': 'sum'
    }).reset_index()

    print(f"DEBUG: Grouped by {dimension}, got {len(summary)} unique values")

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

    # Create structured layout with banner, KPI cards, chart, and assumptions
    volume_impact = assumed_elasticity * price_change_pct

    whatif_layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"backgroundColor": "#ffffff", "padding": "20px"},
            "children": [
                # Banner
                {
                    "name": "Banner",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "style": {
                        "background": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
                        "padding": "30px",
                        "borderRadius": "12px",
                        "marginBottom": "25px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
                    }
                },
                {
                    "name": "BannerTitle",
                    "type": "Header",
                    "children": "",
                    "text": f"What-If Analysis: {abs(price_change_pct):.0f}% Price {direction.title()}",
                    "parentId": "Banner",
                    "style": {"fontSize": "28px", "fontWeight": "bold", "color": "white", "marginBottom": "10px"}
                },
                {
                    "name": "BannerSubtitle",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"Simulating revenue impact across {len(categories)} {dimension.replace('_', ' ')} values",
                    "parentId": "Banner",
                    "style": {"fontSize": "16px", "color": "rgba(255,255,255,0.9)"}
                },
                # KPI Cards Row
                {
                    "name": "KPI_Row",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "row",
                    "extraStyles": "gap: 15px; margin-bottom: 25px;"
                },
                # KPI Card 1: Current Revenue
                {
                    "name": "KPI_Card1",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "borderLeft": "4px solid #6c757d",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI1_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Current Revenue",
                    "parentId": "KPI_Card1",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI1_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"${total_current/1000000:.1f}M" if total_current > 1000000 else f"${total_current:,.0f}",
                    "parentId": "KPI_Card1",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#495057"}
                },
                # KPI Card 2: Projected Revenue
                {
                    "name": "KPI_Card2",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "borderLeft": f"4px solid {'#28a745' if total_change > 0 else '#dc3545'}",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI2_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Projected Revenue",
                    "parentId": "KPI_Card2",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI2_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"${total_projected/1000000:.1f}M" if total_projected > 1000000 else f"${total_projected:,.0f}",
                    "parentId": "KPI_Card2",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#28a745" if total_change > 0 else "#dc3545"}
                },
                # KPI Card 3: Net Impact
                {
                    "name": "KPI_Card3",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "parentId": "KPI_Row",
                    "style": {
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#d4edda" if total_change > 0 else "#f8d7da",
                        "borderLeft": f"4px solid {'#155724' if total_change > 0 else '#721c24'}",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.08)"
                    }
                },
                {
                    "name": "KPI3_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Net Impact",
                    "parentId": "KPI_Card3",
                    "style": {"fontSize": "14px", "color": "#666", "marginBottom": "8px"}
                },
                {
                    "name": "KPI3_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{'+' if total_change > 0 else ''}${total_change/1000000:.2f}M" if abs(total_change) > 1000000 else f"{'+' if total_change > 0 else ''}${total_change:,.0f}",
                    "parentId": "KPI_Card3",
                    "style": {"fontSize": "32px", "fontWeight": "bold", "color": "#155724" if total_change > 0 else "#721c24"}
                },
                {
                    "name": "KPI3_Subtitle",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"({total_change_pct:+.1f}%)",
                    "parentId": "KPI_Card3",
                    "style": {"fontSize": "16px", "color": "#155724" if total_change > 0 else "#721c24", "marginTop": "4px"}
                },
                # Chart
                {
                    "name": "RevenueChart",
                    "type": "HighchartsChart",
                    "children": "",
                    "minHeight": "500px",
                    "options": chart_config
                },
                # Assumptions Section
                {
                    "name": "AssumptionsContainer",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "style": {
                        "padding": "20px",
                        "backgroundColor": "#fff3cd",
                        "borderLeft": "4px solid #ffc107",
                        "borderRadius": "8px",
                        "marginTop": "25px"
                    }
                },
                {
                    "name": "AssumptionsTitle",
                    "type": "Header",
                    "children": "",
                    "text": "⚠️ Important Assumptions",
                    "parentId": "AssumptionsContainer",
                    "style": {"fontSize": "18px", "fontWeight": "bold", "marginBottom": "15px", "marginTop": "0"}
                },
                {
                    "name": "Assumption1",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"• Assumes price elasticity of {assumed_elasticity} (typical for CPG products)",
                    "parentId": "AssumptionsContainer",
                    "style": {"fontSize": "15px", "marginBottom": "8px"}
                },
                {
                    "name": "Assumption2",
                    "type": "Paragraph",
                    "children": "",
                    "text": "• Elasticity varies by product: premium brands typically less elastic than value brands",
                    "parentId": "AssumptionsContainer",
                    "style": {"fontSize": "15px", "marginBottom": "8px"}
                },
                {
                    "name": "Assumption3",
                    "type": "Paragraph",
                    "children": "",
                    "text": "• Does not account for competitive response or market share shifts",
                    "parentId": "AssumptionsContainer",
                    "style": {"fontSize": "15px", "marginBottom": "8px"}
                },
                {
                    "name": "Assumption4",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"• Volume impact: {volume_impact:.1f}% change expected",
                    "parentId": "AssumptionsContainer",
                    "style": {"fontSize": "15px", "marginBottom": "8px"}
                },
                {
                    "name": "Assumption5",
                    "type": "Paragraph",
                    "children": "",
                    "text": "• Recommendation: Pilot test price changes before full rollout",
                    "parentId": "AssumptionsContainer",
                    "style": {"fontSize": "15px", "fontWeight": "bold"}
                }
            ]
        },
        "inputVariables": []
    }

    print(f"DEBUG: Creating what-if layout with {len(categories)} categories")
    full_html = wire_layout(whatif_layout, {})

    recommendation = "positive" if total_change > 0 else "negative"
    narrative = f"A {abs(price_change_pct):.0f}% price {direction} would have an estimated {recommendation} revenue impact of ${abs(total_change):,.0f} ({abs(total_change_pct):.1f}%), assuming moderate price elasticity of {assumed_elasticity}."

    return SkillOutput(
        final_prompt=narrative,
        narrative=None,
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
