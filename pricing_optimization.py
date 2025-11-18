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

from skill_framework import skill, SkillParameter, SkillInput, SkillOutput, SkillVisualization, ParameterDisplayDescription
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
            description="Dimension to analyze within brand portfolio (sub_category, segment, base_size)",
            default_value="base_size"
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
    dimension = parameters.arguments.dimension or "base_size"
    filters = parameters.arguments.filters or []
    start_date = parameters.arguments.start_date
    end_date = parameters.arguments.end_date
    analysis_type = parameters.arguments.analysis_type or "price_comparison"
    price_change_pct = parameters.arguments.price_change_pct or 10

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
                "format": f"${target:.2f}<br><span style='font-size:10px; color:{color}'>({gap_pct:+.0f}%)</span>"
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
            "valuePrefix": "$",
            "valueDecimals": 2
        },
        "plotOptions": {
            "column": {
                "dataLabels": {
                    "enabled": True,
                    "useHTML": True
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
            "valueSuffix": "/oz"
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
            ] + [
                # Table rows for top 5 underpriced items
                item
                for idx, row in comparison.nsmallest(5, 'price_premium_pct').iterrows()
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
            ] + [
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

    # Generate insights with LLM
    ar_utils = ArUtils()

    # Get top 3 premium and top 3 discount items
    premium_items = comparison_clean.nlargest(3, 'price_premium_pct') if len(comparison_clean) > 0 else pd.DataFrame()
    discount_items = comparison_clean.nsmallest(3, 'price_premium_pct') if len(comparison_clean) > 0 else pd.DataFrame()

    insight_prompt = f"""Analyze this competitive pricing comparison:

**Brand**: {brand_filter}
**Dimension**: {dimension.replace('_', ' ').title()}
**Overall Position**: {weighted_premium:+.1f}% avg premium vs competition
**Volume Share**: {volume_share:.1f}%

**Premium Positioned** (priced above competition):
{chr(10).join([f"- {row[dimension]}: {brand_filter} ${row['avg_price_target']:.2f} vs Competition ${row['avg_price_comp']:.2f} ({row['price_premium_pct']:+.1f}%)" for _, row in premium_items.iterrows()]) if len(premium_items) > 0 else "None"}

**Value Positioned** (priced below competition):
{chr(10).join([f"- {row[dimension]}: {brand_filter} ${row['avg_price_target']:.2f} vs Competition ${row['avg_price_comp']:.2f} ({row['price_premium_pct']:+.1f}%)" for _, row in discount_items.iterrows()]) if len(discount_items) > 0 else "None"}

Provide strategic analysis:
1. **Competitive Positioning**: What does the pricing pattern reveal about {brand_filter}'s strategy?
2. **Opportunities**: Where can {brand_filter} raise prices to match or exceed competition?
3. **Risks**: Where is {brand_filter} overpriced and at risk of losing share?
4. **Recommendations**: Specific actions for pricing optimization.

Use markdown formatting. **Limit response to 250 words maximum.**"""

    try:
        detailed_narrative = ar_utils.get_llm_response(insight_prompt)
        if not detailed_narrative:
            detailed_narrative = f"""## Competitive Positioning Analysis

{brand_filter} is positioned at {weighted_premium:+.1f}% vs competition on average, holding {volume_share:.1f}% volume share.

**Key Findings:**
- Price leadership in {price_leaders} of {num_skus} {dimension.replace('_', ' ')} values analyzed
- Strategic mix of premium and value positioning across portfolio
"""
    except Exception as e:
        print(f"DEBUG: LLM insight generation failed: {e}")
        detailed_narrative = f"## Competitive Positioning\n\n{brand_filter} competitive analysis complete."

    brief_summary = f"{brand_display} positioned at {weighted_premium:+.1f}% vs competition with {volume_share:.1f}% volume share."

    # Create pills
    param_pills = [
        ParameterDisplayDescription(key="brand", value=f"Brand: {brand_display}"),
        ParameterDisplayDescription(key="time_period", value=f"Period: {time_period}"),
        ParameterDisplayDescription(key="dimension", value=f"Dimension: {dimension.replace('_', ' ').title()}"),
        ParameterDisplayDescription(key="skus", value=f"SKUs: {num_skus}"),
        ParameterDisplayDescription(key="premium", value=f"Avg Premium: {weighted_premium:+.1f}%"),
    ]

    return SkillOutput(
        final_prompt=brief_summary,
        narrative=detailed_narrative,
        visualizations=[SkillVisualization(title="Competitive Comparison", layout=html)],
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
                            "tooltip": {"valuePrefix": "$", "valueDecimals": 2},
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
                            "tooltip": {"valueSuffix": "M", "valuePrefix": "$"},
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
