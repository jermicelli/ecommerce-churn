import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

path = "data/2019-Oct.csv"  # default path

spark = (
    SparkSession.builder
    .appName("churn-process")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.session.timeZone", "UTC")  # the data's times are UTC
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

schema = """
    event_time TIMESTAMP, event_type STRING, product_id LONG, category_id LONG,
    category_code STRING, brand STRING, price DOUBLE, user_id LONG, user_session STRING
"""
df = (
    spark.read
    .option("header", True)
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss 'UTC'")
    .schema(schema)
    .csv(path)
)

df.createOrReplaceTempView("events")  # make a SQL table called "events"
# print("Duplicate rows")
# spark.sql("""SELECT
#     (SELECT COUNT(*) FROM events) AS total_rows,
#     (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM events)) AS distinct_rows,
#     total_rows - distinct_rows AS duplicate_rows
# """).show()

# print("Bad prices")
# spark.sql("SELECT event_type, COUNT(*) AS count FROM events WHERE price < 0 GROUP BY event_type").show()

# print("Distinct event types")
# spark.sql("SELECT DISTINCT event_type FROM events").show()

spark.sql("""CREATE OR REPLACE TEMP VIEW events_clean AS
    SELECT DISTINCT *
    FROM events  
""")

spark.sql("DESCRIBE events_clean").show(truncate=False)

# Building a feature table of users and their events
# Features: days_since_last_purchase, days_since_last_visit, n_purchases,
# n_views, n_carts, n_sessions, n_active_days, avg_price, total_spent,
# n_unique_products, n_unique_categories, n_unique_brands
# conversion_rate = n_purchases / n_views
# activity_last_7_days = n_active_days in the last 7 days

spark.sql("""
    CREATE OR REPLACE TEMP VIEW oct_buyers AS
    SELECT DISTINCT user_id
    FROM events_clean
    WHERE event_type = 'purchase'
""")

spark.sql("""
    CREATE OR REPLACE TEMP VIEW buyer_events AS
    SELECT e.*
    FROM events_clean e
    LEFT SEMI JOIN oct_buyers b ON e.user_id = b.user_id
""")


spark.sql("""
    CREATE OR REPLACE TEMP VIEW features AS
    WITH base AS (
        SELECT
            user_id,

            -- Recency (days back from Nov 1)
            DATEDIFF(DATE '2019-11-01',
                     DATE(MAX(CASE WHEN event_type = 'purchase' THEN event_time END))) AS days_since_last_purchase,
            DATEDIFF(DATE '2019-11-01', DATE(MAX(event_time)))                        AS days_since_last_visit,

            -- Frequency
            SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS n_purchases,
            COUNT(DISTINCT user_session)                              AS n_sessions,
            COUNT(DISTINCT DATE(event_time))                          AS n_active_days,

            -- Monetary
            SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS total_spent,
            AVG(CASE WHEN event_type = 'purchase' THEN price END)        AS avg_purchase_price,

            -- Interest
            SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS n_views,
            COUNT(DISTINCT product_id)                            AS n_unique_products,
            SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS n_cart_adds,

            -- Behaviour
            COUNT(DISTINCT brand)       AS n_unique_brands,
            COUNT(DISTINCT category_id) AS n_unique_categories,

            -- Trend
            SUM(CASE WHEN event_time >= TIMESTAMP '2019-10-25 00:00:00' THEN 1 ELSE 0 END) AS events_last_7_days

        FROM buyer_events
        GROUP BY user_id
    )
    SELECT *,
           n_purchases / GREATEST(n_views, 1) AS conversion_rate
    FROM base
""")

spark.sql("CACHE TABLE features")

spark.sql("SELECT COUNT(*) AS n_users FROM features").show()
spark.sql("SELECT * FROM features LIMIT 10").show()
spark.sql("SELECT * FROM features").describe().show()

future = (
    spark.read
    .option("header", True)
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss 'UTC'")
    .schema(schema)
    .csv(["data/2019-Nov.csv", "data/2019-Dec.csv", "data/2020-Jan.csv"])
)
future.createOrReplaceTempView("future_events")

spark.sql("""
    CREATE OR REPLACE TEMP VIEW future_buyers AS
    SELECT DISTINCT user_id FROM future_events WHERE event_type = 'purchase'
""")

spark.sql("CACHE TABLE future_buyers")

spark.sql("""
    CREATE OR REPLACE TEMP VIEW training_table AS
    SELECT f.*,
           CASE WHEN fb.user_id IS NULL THEN 1 ELSE 0 END AS churned
    FROM features f
    LEFT JOIN future_buyers fb ON f.user_id = fb.user_id
""")

spark.sql("""
    SELECT COUNT(*) AS n_users,
           SUM(churned) AS n_churned,
           AVG(churned) AS churn_rate
    FROM training_table
""").show()

spark.table("training_table").toPandas().to_parquet("data/training_table.parquet", index=False)
print("saved data/training_table.parquet")

spark.stop()