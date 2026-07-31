-- =============================================================================
-- roll_rate_pairs.sql
-- Builds the (bucket at month t) -> (bucket at month t+1) pairs used to
-- estimate the empirical monthly roll-rate / transition matrix. Self-joins
-- the snapshot panel on the same loan at consecutive months_on_book values.
-- Optional filters (bound parameters) allow computing the matrix for a
-- specific product / geography slice, e.g. for stress testing a segment.
--
-- Bound parameters:
--   :product      product code or NULL (no filter)
--   :geography    geography code or NULL (no filter)
-- =============================================================================

SELECT
    l.product,
    l.geography,
    cur.delinquency_bucket AS bucket_from,
    nxt.delinquency_bucket AS bucket_to
FROM loan_snapshots cur
INNER JOIN loan_snapshots nxt
    ON nxt.loan_id = cur.loan_id
   AND nxt.months_on_book = cur.months_on_book + 1
INNER JOIN loans l
    ON l.loan_id = cur.loan_id
WHERE (:product   IS NULL OR l.product   = :product)
  AND (:geography IS NULL OR l.geography = :geography);
