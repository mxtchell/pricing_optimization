"""Explore pasta dataset for pricing optimization"""
from dotenv import load_dotenv
load_dotenv()

from answer_rocket import AnswerRocketClient
import os

DATABASE_ID = os.getenv('DATABASE_ID')
AR_URL = os.getenv('AR_URL')
AR_TOKEN = os.getenv('AR_TOKEN')

print(f"Using DATABASE_ID: {DATABASE_ID}")
print(f"Using AR_URL: {AR_URL}")
print(f"Using AR_TOKEN: {AR_TOKEN[:20]}..." if AR_TOKEN else "No token")

client = AnswerRocketClient(url=AR_URL, token=AR_TOKEN)

# Query to see all columns and sample data
query = """
SELECT *
FROM read_csv('pasta_2025.csv')
LIMIT 5
"""

print(f"\nExecuting query:\n{query}")

result = client.data.execute_sql_query(DATABASE_ID, query, row_limit=10)

if result.success and hasattr(result, 'df'):
    df = result.df
    print(f"\n✓ Dataset columns: {list(df.columns)}")
    print(f"\n✓ Sample data:")
    print(df)
    print(f"\n✓ Data types:")
    print(df.dtypes)
else:
    print(f"Failed to query dataset")
    if hasattr(result, 'error'):
        print(f"Error: {result.error}")
