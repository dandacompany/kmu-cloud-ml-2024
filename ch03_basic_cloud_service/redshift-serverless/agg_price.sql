with min_max_price as (
    select min(price) as min_price, max(price) as max_price
    from datamart.houseprice
),
price_ranges AS (
    SELECT 
        min_price,
        max_price,
        (max_price - min_price) / 10 AS bin_width
    FROM min_max_price
),
binned_prices AS (
    SELECT 
        FLOOR((price - min_price) / bin_width) + 1 AS bucket,
        COUNT(*) AS count,
        MIN(price) AS min_price_in_bucket,
        MAX(price) AS max_price_in_bucket
    FROM datamart.houseprice, price_ranges
    GROUP BY FLOOR((price - min_price) / bin_width) + 1
)
SELECT 
    bucket as "구간",
    count as "카운트",
    TO_CHAR(min_price_in_bucket, '999,999,999.99') AS "최소 가격",
    TO_CHAR(max_price_in_bucket, '999,999,999.99') AS "최대 가격",
    TO_CHAR(min_price_in_bucket, '999,999,999.99') || ' - ' || TO_CHAR(max_price_in_bucket, '999,999,999.99') AS "가격대"
FROM binned_prices
ORDER BY bucket;
