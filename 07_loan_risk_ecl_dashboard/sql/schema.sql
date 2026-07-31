-- =============================================================================
-- schema.sql
-- Core schema for the Loan Portfolio Risk & ECL Analytics Dashboard.
-- Written in portable ANSI-ish SQL (runs on SQLite; mirrors syntax you would
-- use on SQL Server / PostgreSQL with only minor type-name changes).
-- =============================================================================

DROP TABLE IF EXISTS loan_snapshots;
DROP TABLE IF EXISTS loans;

-- One row per originated loan (static / slowly-changing attributes).
CREATE TABLE loans (
    loan_id             TEXT PRIMARY KEY,
    product             TEXT    NOT NULL,   -- personal_loan | auto_loan | mortgage | credit_card
    geography           TEXT    NOT NULL,   -- North | South | East | West | Central
    origination_date    TEXT    NOT NULL,   -- ISO date (YYYY-MM-DD)
    vintage_quarter      TEXT    NOT NULL,   -- e.g. 2023Q1, derived from origination_date
    term_months         INTEGER NOT NULL,
    principal           REAL    NOT NULL,
    interest_rate       REAL    NOT NULL,   -- annual nominal rate, e.g. 0.115 = 11.5%
    credit_score        INTEGER NOT NULL,   -- bureau-style score at origination, 300-850
    borrower_income     REAL    NOT NULL,
    channel             TEXT    NOT NULL,   -- branch | digital | partner
    closed_flag         INTEGER NOT NULL DEFAULT 0,  -- 1 = fully amortized / paid off before as_of_date
    default_flag        INTEGER NOT NULL DEFAULT 0,  -- 1 = loan reached 90+ DPD (Stage 3) at any point
    default_date        TEXT                          -- date first observed at 90+ DPD, NULL if never
);

CREATE INDEX idx_loans_product   ON loans(product);
CREATE INDEX idx_loans_geography ON loans(geography);
CREATE INDEX idx_loans_vintage   ON loans(vintage_quarter);

-- One row per loan per monthly observation ("snapshot").
-- This is the panel data used for delinquency bucketing, roll-rate
-- computation and ECL staging.
CREATE TABLE loan_snapshots (
    snapshot_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_id             TEXT    NOT NULL,
    snapshot_date       TEXT    NOT NULL,   -- ISO date, first-of-month
    months_on_book      INTEGER NOT NULL,   -- months since origination_date
    outstanding_balance REAL    NOT NULL,
    dpd                 INTEGER NOT NULL,   -- days past due at this snapshot
    delinquency_bucket  TEXT    NOT NULL,   -- Current | 1-29 | 30-59 | 60-89 | 90+
    stage                INTEGER NOT NULL,   -- IFRS 9 / Ind AS 109 stage: 1, 2 or 3
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id)
);

CREATE INDEX idx_snap_loan   ON loan_snapshots(loan_id);
CREATE INDEX idx_snap_date   ON loan_snapshots(snapshot_date);
CREATE INDEX idx_snap_bucket ON loan_snapshots(delinquency_bucket);
