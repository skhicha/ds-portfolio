-- =============================================================================
-- latest_snapshot_per_loan.sql
-- For a given as_of_date (bound parameter :as_of_date), return the most
-- recent snapshot on or before that date for every loan, joined to the
-- loan's static attributes. This is the "point in time portfolio view"
-- used for KRI / ECL / roll-rate dashboards.
--
-- Bound parameters:
--   :as_of_date   ISO date string, e.g. '2026-06-30'
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
)
SELECT
    r.loan_id,
    r.snapshot_date,
    r.months_on_book,
    r.outstanding_balance,
    r.dpd,
    r.delinquency_bucket,
    r.stage,
    l.product,
    l.geography,
    l.vintage_quarter,
    l.origination_date,
    l.term_months,
    l.principal,
    l.interest_rate,
    l.credit_score,
    l.borrower_income,
    l.channel,
    l.closed_flag,
    l.default_flag,
    l.default_date
FROM ranked_snapshots r
INNER JOIN loans l
    ON l.loan_id = r.loan_id
WHERE r.rn = 1;
