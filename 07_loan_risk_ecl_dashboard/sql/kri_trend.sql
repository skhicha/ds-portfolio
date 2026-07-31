-- =============================================================================
-- kri_trend.sql
-- Monthly time series of key risk indicators (delinquency %, Stage mix,
-- balances) across the whole snapshot history, optionally filtered by
-- product / geography. Used to plot KRI trend lines on the dashboard.
--
-- Bound parameters:
--   :product      product code or NULL (no filter)
--   :geography    geography code or NULL (no filter)
-- =============================================================================

SELECT
    s.snapshot_date,
    COUNT(DISTINCT s.loan_id)                                              AS loan_count,
    SUM(s.outstanding_balance)                                              AS total_outstanding,
    SUM(CASE WHEN s.delinquency_bucket <> 'Current' THEN s.outstanding_balance ELSE 0 END)
        / NULLIF(SUM(s.outstanding_balance), 0)                            AS delinquent_balance_pct,
    SUM(CASE WHEN s.stage = 1 THEN s.outstanding_balance ELSE 0 END)       AS stage1_balance,
    SUM(CASE WHEN s.stage = 2 THEN s.outstanding_balance ELSE 0 END)       AS stage2_balance,
    SUM(CASE WHEN s.stage = 3 THEN s.outstanding_balance ELSE 0 END)       AS stage3_balance
FROM loan_snapshots s
INNER JOIN loans l
    ON l.loan_id = s.loan_id
WHERE (:product   IS NULL OR l.product   = :product)
  AND (:geography IS NULL OR l.geography = :geography)
GROUP BY s.snapshot_date
ORDER BY s.snapshot_date;
