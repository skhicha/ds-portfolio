-- =============================================================================
-- view_delinquency_buckets.sql
-- Aggregation view: outstanding balance and loan counts by delinquency bucket
-- for every snapshot date in the panel. Downstream code filters this view
-- to a single as_of_date with a bound parameter rather than baking the date
-- into the view itself.
-- =============================================================================

DROP VIEW IF EXISTS vw_delinquency_bucket_summary;

CREATE VIEW vw_delinquency_bucket_summary AS
SELECT
    s.snapshot_date,
    l.product,
    l.geography,
    l.vintage_quarter,
    s.delinquency_bucket,
    s.stage,
    COUNT(DISTINCT s.loan_id)          AS loan_count,
    SUM(s.outstanding_balance)         AS total_outstanding,
    AVG(s.dpd)                         AS avg_dpd
FROM loan_snapshots s
INNER JOIN loans l
    ON l.loan_id = s.loan_id
GROUP BY
    s.snapshot_date,
    l.product,
    l.geography,
    l.vintage_quarter,
    s.delinquency_bucket,
    s.stage;
