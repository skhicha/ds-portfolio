-- =============================================================================
-- segment_portfolio.sql
-- Portfolio segmentation by product / geography / vintage as of a given date.
-- Filters are optional: pass NULL for any dimension you don't want to
-- restrict on (bound as a parameter, not string-formatted).
--
-- Bound parameters:
--   :as_of_date   ISO date string
--   :product      product code or NULL
--   :geography    geography code or NULL
--   :vintage      vintage_quarter code or NULL
-- =============================================================================

WITH ranked_snapshots AS (
    SELECT
        s.*,
        ROW_NUMBER() OVER (
            PARTITION BY s.loan_id
            ORDER BY s.snapshot_date DESC
        ) AS rn
    FROM loan_snapshots s
    WHERE s.snapshot_date <= :as_of_date
),
point_in_time AS (
    SELECT
        r.loan_id,
        r.outstanding_balance,
        r.dpd,
        r.delinquency_bucket,
        r.stage,
        l.product,
        l.geography,
        l.vintage_quarter,
        l.principal,
        l.interest_rate
    FROM ranked_snapshots r
    INNER JOIN loans l ON l.loan_id = r.loan_id
    WHERE r.rn = 1
)
SELECT
    product,
    geography,
    vintage_quarter,
    COUNT(*)                                                   AS loan_count,
    SUM(outstanding_balance)                                   AS total_outstanding,
    SUM(principal)                                              AS total_principal,
    AVG(interest_rate)                                          AS avg_interest_rate,
    SUM(CASE WHEN delinquency_bucket <> 'Current' THEN 1 ELSE 0 END) AS delinquent_loan_count,
    SUM(CASE WHEN delinquency_bucket <> 'Current' THEN outstanding_balance ELSE 0 END) AS delinquent_balance,
    SUM(CASE WHEN stage = 3 THEN 1 ELSE 0 END)                 AS stage3_loan_count,
    SUM(CASE WHEN stage = 3 THEN outstanding_balance ELSE 0 END) AS stage3_balance
FROM point_in_time
WHERE (:product   IS NULL OR product   = :product)
  AND (:geography IS NULL OR geography = :geography)
  AND (:vintage   IS NULL OR vintage_quarter = :vintage)
GROUP BY product, geography, vintage_quarter
ORDER BY product, geography, vintage_quarter;
