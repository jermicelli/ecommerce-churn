"""First look at the e-commerce event data with Spark.

Run:  python src/explore.py               (October, default)
      python src/explore.py data/2019-Nov.csv
"""
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# On Windows, make Spark's Python workers use this same Python, not a Store alias.
os.environ["PYSPARK_PYTHON"] = sys.executable

path = sys.argv[1] if len(sys.argv) > 1 else "data/2019-Oct.csv"

# 1. Start Spark. This is the "instantiate" step: one SparkSession per program.
#    local[*]  = run on this laptop, using every CPU core.
#    4g        = memory for Spark; this laptop has ~8 GB, leave the rest for Windows.
spark = (
    SparkSession.builder
    .appName("churn-explore")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.session.timeZone", "UTC")  # the data's times are UTC
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")  # hide Spark's chatty INFO logs

# 2. Read the CSV. We give the column types ourselves: letting Spark guess
#    (inferSchema) would read all 5.7 GB an extra time just to guess.
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

# Nothing has been read yet: Spark is "lazy" and only does work when you ask
# for a result (show, count, ...). Each step below scans the file, ~1-2 min each.

# print("\n== Columns and types")
# df.printSchema()

# print("== First 5 rows")
# df.show(5, truncate=False)

# print("== Row count:", f"{df.count():,}")

# print("\n== Events by type (view / cart / purchase)")
# df.groupBy("event_type").count().orderBy(F.desc("count")).show()

# print("== Empty (null) values per column")
# df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()

# 3. You can also query a DataFrame with plain SQL by giving it a table name.
df.createOrReplaceTempView("events")
print("== Date range and unique users (via SQL)")
spark.sql("""
    SELECT MIN(event_time) AS first_event,
           MAX(event_time) AS last_event,
           COUNT(DISTINCT user_id) AS users
    FROM events
""").show(truncate=False)

# Filtering users who made a purchase
print("== Users who made a purchase")
spark.sql("""
    SELECT user_id
    FROM events
    WHERE event_type = 'purchase'
""").show(truncate=False)

# SQL query to show all events for users who made a purchase
print("== All events for users who made a purchase")
spark.sql("""
    SELECT *
    FROM events
    WHERE user_id IN (
        SELECT user_id
        FROM events
        WHERE event_type = 'purchase'
    )""").show(truncate=False)

# Nov-Jan: read the three "future" months as one table, same columns as October.
future = (
    spark.read
    .option("header", True)
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss 'UTC'")
    .schema(schema)
    .csv(["data/2019-Nov.csv", "data/2019-Dec.csv", "data/2020-Jan.csv"])
)
future.createOrReplaceTempView("future_events")

# Save each buyer list as its own small table. CACHE keeps it in memory, so the
# queries below don't re-read the big files every time.
spark.sql("""
    CREATE OR REPLACE TEMP VIEW oct_buyers AS
    SELECT DISTINCT user_id FROM events WHERE event_type = 'purchase'
""")
spark.sql("""
    CREATE OR REPLACE TEMP VIEW future_buyers AS
    SELECT DISTINCT user_id FROM future_events WHERE event_type = 'purchase'
""")
spark.sql("CACHE TABLE oct_buyers")
spark.sql("CACHE TABLE future_buyers")

# 1. Users who purchased in Nov-Jan
print("== Users who purchased in Nov-Jan")
spark.sql("SELECT user_id FROM future_buyers").show()

# 2. Bought in October, did NOT buy in Nov-Jan (= churned).
#    LEFT ANTI JOIN keeps rows from the left table that have NO match on the right.
print("== Bought in October, did not buy in Nov-Jan (churned)")
churned = spark.sql("""
    SELECT o.user_id
    FROM oct_buyers o
    LEFT ANTI JOIN future_buyers f ON o.user_id = f.user_id
""")
churned.show()

# 3. Bought in October AND again in Nov-Jan (= stayed).
#    LEFT SEMI JOIN keeps rows from the left table that DO have a match on the right.
print("== Bought in October and again in Nov-Jan (stayed)")
stayed = spark.sql("""
    SELECT o.user_id
    FROM oct_buyers o
    LEFT SEMI JOIN future_buyers f ON o.user_id = f.user_id
""")
stayed.show()

n_oct = spark.table("oct_buyers").count()
n_future = spark.table("future_buyers").count()
n_churned, n_stayed = churned.count(), stayed.count()
print(f"October buyers:      {n_oct:,}")
print(f"Nov-Jan buyers:      {n_future:,}")
print(f"Churned:             {n_churned:,} ({n_churned / n_oct:.1%})")
print(f"Stayed:              {n_stayed:,} ({n_stayed / n_oct:.1%})")
assert n_churned + n_stayed == n_oct  # every October buyer is in exactly one group

spark.stop()
