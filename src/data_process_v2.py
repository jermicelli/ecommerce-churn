"""Feature table v2: the 15 original features + 9 new ones, one row per October buyer.

Run:  python src/data_process_v2.py
Output: data/training_table_v2.parquet
"""
from pyspark.sql import SparkSession

# A category/brand needs at least this many October buyers to get its own repeat
# rate; smaller ones use the overall rate (a rate from 2 people is just luck).
MIN_BUYERS = 50


def build_training_table(spark, min_buyers=MIN_BUYERS):
    """Expects temp views `events` (October) and `future_events` (Nov-Jan).
    Returns the training table as a Spark DataFrame."""

    # ---------- Cleaning + buyers (same as v1) ----------
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW events_clean AS
        SELECT DISTINCT * FROM events
    """)

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW oct_buyers AS
        SELECT DISTINCT user_id FROM events_clean WHERE event_type = 'purchase'
    """)

    # Buyers' events, plus two helper columns for the new features:
    #   top_category: first word of category_code ('electronics.smartphone' -> 'electronics')
    #   brand_key:    brand, with missing brands grouped as 'unknown'
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW buyer_events AS
        SELECT e.*,
               COALESCE(SUBSTRING_INDEX(e.category_code, '.', 1), 'unknown') AS top_category,
               COALESCE(e.brand, 'unknown')                                  AS brand_key
        FROM events_clean e
        LEFT SEMI JOIN oct_buyers b ON e.user_id = b.user_id
    """)
    spark.sql("CACHE TABLE buyer_events")  # read by every query below

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW purchases AS
        SELECT * FROM buyer_events WHERE event_type = 'purchase'
    """)

    # ---------- The 15 original features (unchanged from v1) ----------
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW features_v1 AS
        WITH base AS (
            SELECT
                user_id,
                DATEDIFF(DATE '2019-11-01',
                         DATE(MAX(CASE WHEN event_type = 'purchase' THEN event_time END))) AS days_since_last_purchase,
                DATEDIFF(DATE '2019-11-01', DATE(MAX(event_time)))                        AS days_since_last_visit,
                SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS n_purchases,
                COUNT(DISTINCT user_session)                              AS n_sessions,
                COUNT(DISTINCT DATE(event_time))                          AS n_active_days,
                SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS total_spent,
                AVG(CASE WHEN event_type = 'purchase' THEN price END)        AS avg_purchase_price,
                SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS n_views,
                COUNT(DISTINCT product_id)                            AS n_unique_products,
                SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS n_cart_adds,
                COUNT(DISTINCT brand)       AS n_unique_brands,
                COUNT(DISTINCT category_id) AS n_unique_categories,
                SUM(CASE WHEN event_time >= TIMESTAMP '2019-10-25 00:00:00' THEN 1 ELSE 0 END) AS events_last_7_days
            FROM buyer_events
            GROUP BY user_id
        )
        SELECT *, n_purchases / GREATEST(n_views, 1) AS conversion_rate
        FROM base
    """)

    # ---------- New features ----------

    # Key moments per buyer: first/last event and first/last purchase.
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW user_times AS
        SELECT user_id,
               MIN(event_time)                                            AS first_event,
               MAX(event_time)                                            AS last_event,
               MIN(CASE WHEN event_type = 'purchase' THEN event_time END) AS first_purchase,
               MAX(CASE WHEN event_type = 'purchase' THEN event_time END) AS last_purchase
        FROM buyer_events
        GROUP BY user_id
    """)

    # Views before the first purchase / after the last purchase.
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW around_purchase AS
        SELECT e.user_id,
               SUM(CASE WHEN e.event_type = 'view' AND e.event_time < t.first_purchase THEN 1 ELSE 0 END) AS views_before_first_purchase,
               SUM(CASE WHEN e.event_type = 'view' AND e.event_time > t.last_purchase  THEN 1 ELSE 0 END) AS views_after_last_purchase
        FROM buyer_events e
        JOIN user_times t ON e.user_id = t.user_id
        GROUP BY e.user_id
    """)

    # Views after the last purchase, only in categories the buyer has bought from.
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW same_category_after AS
        SELECT e.user_id, COUNT(*) AS same_category_views_after_purchase
        FROM buyer_events e
        JOIN user_times t ON e.user_id = t.user_id
        LEFT SEMI JOIN (SELECT DISTINCT user_id, category_id FROM purchases) pc
            ON e.user_id = pc.user_id AND e.category_id = pc.category_id
        WHERE e.event_type = 'view' AND e.event_time > t.last_purchase
        GROUP BY e.user_id
    """)

    # Products added to the cart but never bought (in October).
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW carted_not_bought AS
        SELECT c.user_id, COUNT(*) AS n_carted_not_bought
        FROM (SELECT DISTINCT user_id, product_id FROM buyer_events WHERE event_type = 'cart') c
        LEFT ANTI JOIN (SELECT DISTINCT user_id, product_id FROM purchases) p
            ON c.user_id = p.user_id AND c.product_id = p.product_id
        GROUP BY c.user_id
    """)

    # Top-level category of the most recent purchase (ties broken by name, so it's repeatable).
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW top_category AS
        SELECT user_id,
               MAX(NAMED_STRUCT('t', event_time, 'c', top_category)).c AS purchase_top_category
        FROM purchases
        GROUP BY user_id
    """)

    # Repeat buyer = purchases in 2+ different sessions in October.
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW user_repeat AS
        SELECT user_id,
               CASE WHEN COUNT(DISTINCT user_session) >= 2 THEN 1 ELSE 0 END AS is_repeat
        FROM purchases
        GROUP BY user_id
    """)

    # Repeat rate per category / brand, then averaged over what each buyer bought.
    # Groups with fewer than min_buyers buyers fall back to the overall repeat rate.
    for key, feature in [("category_id", "category_repeat_rate"), ("brand_key", "brand_repeat_rate")]:
        spark.sql(f"""
            CREATE OR REPLACE TEMP VIEW {feature} AS
            WITH bought AS (
                SELECT DISTINCT user_id, {key} FROM purchases
            ),
            group_rate AS (
                SELECT b.{key},
                       CASE WHEN COUNT(*) >= {min_buyers} THEN AVG(r.is_repeat)
                            ELSE (SELECT AVG(is_repeat) FROM user_repeat) END AS rate
                FROM bought b
                JOIN user_repeat r ON b.user_id = r.user_id
                GROUP BY b.{key}
            )
            SELECT b.user_id, AVG(g.rate) AS {feature}
            FROM bought b
            JOIN group_rate g ON b.{key} = g.{key}
            GROUP BY b.user_id
        """)

    # ---------- Put it all together: 15 + 9 features ----------
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW features_v2 AS
        SELECT f.*,
               tc.purchase_top_category,
               cr.category_repeat_rate,
               br.brand_repeat_rate,
               COALESCE(cn.n_carted_not_bought, 0)                 AS n_carted_not_bought,
               ap.views_after_last_purchase,
               COALESCE(sc.same_category_views_after_purchase, 0)  AS same_category_views_after_purchase,
               ap.views_before_first_purchase,
               DAY(t.first_event)                                  AS first_seen_day,
               DATEDIFF(DATE(t.last_event), DATE(t.first_event))   AS days_first_to_last_visit
        FROM features_v1 f
        JOIN user_times t            ON f.user_id = t.user_id
        JOIN around_purchase ap      ON f.user_id = ap.user_id
        JOIN top_category tc         ON f.user_id = tc.user_id
        JOIN category_repeat_rate cr ON f.user_id = cr.user_id
        JOIN brand_repeat_rate br    ON f.user_id = br.user_id
        LEFT JOIN carted_not_bought cn   ON f.user_id = cn.user_id
        LEFT JOIN same_category_after sc ON f.user_id = sc.user_id
    """)

    # ---------- Label (same as v1) ----------
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW future_buyers AS
        SELECT DISTINCT user_id FROM future_events WHERE event_type = 'purchase'
    """)

    return spark.sql("""
        SELECT f.*,
               CASE WHEN fb.user_id IS NULL THEN 1 ELSE 0 END AS churned
        FROM features_v2 f
        LEFT JOIN future_buyers fb ON f.user_id = fb.user_id
    """)


def check(table, n_buyers, v1_path=None):
    """Sanity checks on the finished pandas table. Stops the script if anything is off."""
    import pandas as pd

    assert len(table) == n_buyers, f"expected {n_buyers} rows, got {len(table)}"
    assert table["user_id"].is_unique, "a user appears more than once"
    nulls = table.isna().sum()
    assert nulls.sum() == 0, f"empty values found:\n{nulls[nulls > 0]}"

    assert table["category_repeat_rate"].between(0, 1).all()
    assert table["brand_repeat_rate"].between(0, 1).all()
    assert table["first_seen_day"].between(1, 31).all()
    assert table["days_first_to_last_visit"].between(0, 30).all()
    assert (table["views_before_first_purchase"] + table["views_after_last_purchase"] <= table["n_views"]).all()
    assert (table["same_category_views_after_purchase"] <= table["views_after_last_purchase"]).all()
    assert (table["n_carted_not_bought"] <= table["n_cart_adds"]).all()

    # The 15 old features and the label must match v1 exactly.
    if v1_path:
        v1 = pd.read_parquet(v1_path)
        merged = table.merge(v1, on="user_id", suffixes=("", "_v1"))
        assert len(merged) == len(v1) == len(table), "v1 and v2 have different users"
        for col in [c for c in v1.columns if c != "user_id"]:
            a, b = merged[col], merged[f"{col}_v1"]
            same = (a - b).abs().le(1e-6).all() if a.dtype.kind == "f" else (a == b).all()
            assert same, f"column {col} differs from v1"
    print("all checks passed")


if __name__ == "__main__":
    import os

    spark = (
        SparkSession.builder
        .appName("churn-process-v2")
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

    def read(paths):
        return (
            spark.read
            .option("header", True)
            .option("timestampFormat", "yyyy-MM-dd HH:mm:ss 'UTC'")
            .schema(schema)
            .csv(paths)
        )

    read("data/2019-Oct.csv").createOrReplaceTempView("events")
    read(["data/2019-Nov.csv", "data/2019-Dec.csv", "data/2020-Jan.csv"]).createOrReplaceTempView("future_events")

    training_table = build_training_table(spark)
    training_table.createOrReplaceTempView("training_table")
    spark.sql("CACHE TABLE training_table")

    spark.sql("""
        SELECT COUNT(*) AS n_users, SUM(churned) AS n_churned, AVG(churned) AS churn_rate
        FROM training_table
    """).show()
    spark.sql("SELECT * FROM training_table").describe().show(vertical=True)

    n_buyers = spark.table("oct_buyers").count()
    table = training_table.toPandas()
    v1_path = "data/training_table.parquet"
    check(table, n_buyers, v1_path if os.path.exists(v1_path) else None)

    table.to_parquet("data/training_table_v2.parquet", index=False)
    print(f"saved data/training_table_v2.parquet: {table.shape[0]:,} rows x {table.shape[1]} columns")

    spark.stop()
