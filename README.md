# Amazon Sales Analysis

Statistical analysis of ~114,000 Amazon marketplace orders from an Indian
apparel seller, plus a Power BI dashboard over the same data.

Two deliverables:

- **`src/`** — five hypothesis tests in Python, each with a saved figure
- **`Online Sales.pbix`** — a four-page Power BI dashboard

## Dashboard

`Online Sales.pbix` (Power BI Desktop). Headline figures across the reporting
period:

| | |
|---|---|
| Total revenue | **75M** |
| Total orders | **114K** |
| Average order value | **663.16** |
| B2B revenue share | **0.77%** |
| Revenue lost to cancellations | **3.73M** |

**Page 1 — Overview.** Revenue and order KPIs, revenue trend by week,
revenue by category, and an order-size split showing that single-item orders
dominate at 99.67% of volume.

**Page 2 — Product.** Revenue by category × size, top 10 SKUs, revenue and
order count by style, and size distribution within each category. Set and
Kurta together account for the majority of the 75M.

**Page 3 — B2B vs B2C.** Revenue and order comparison, monthly trend by
segment, average order value per category split by segment, and order-size
distribution. B2B is a very small share of the business (0.77% of revenue).

**Page 4 — Cancellations.** Cancellation rate by fulfilment method, order
status breakdown per category, cancellation trend over time, and revenue lost
by category (3.73M total, concentrated in Set and Kurta).

## Hypothesis tests

```bash
pip install -r requirements.txt
python -m src.hypothesis_tests
```

| # | Question | Test |
|---|---|---|
| 1 | Does order quantity track sale amount, B2B vs B2C? | Spearman correlation |
| 2 | Do B2B orders earn more per unit than B2C? | Mann-Whitney U + rank-biserial |
| 3 | Do sale amounts differ across product categories? | Kruskal-Wallis + Bonferroni pairwise |
| 4 | Is the spread of amounts wider for larger orders? | Bootstrap CI on IQR difference |
| 5 | Is cancellation associated with fulfilment method? | Chi-square + Cramér's V |

Each test writes a figure to `figures/`.

### Notes on test choice

**Effect sizes are reported alongside p-values.** At n ≈ 114,000 almost any
difference reaches p < 0.05, so significance alone says little. Test 2 reports
the rank-biserial correlation and test 5 reports Cramér's V; where those are
small, the script says so explicitly rather than presenting a significant
p-value as a finding.

**Test 4 tests IQR, not variance.** The hypothesis is about spread as measured
by IQR, so it is tested by bootstrapping the difference in IQR. Levene's test
compares variance, which on this heavily right-skewed distribution is driven by
a handful of extreme orders and answers a different question.

**Test 5 crosses fulfilment against cancellation specifically** — a 2×2 table —
rather than against the full multi-valued `Status` column, which would test
whether *any* status differs by fulfilment method.

**Test 3 corrects for multiple comparisons.** Nine categories give 36 pairwise
tests; at α = 0.05 roughly two would appear significant by chance, so the
pairwise p-values are Bonferroni-corrected.

## Data

The raw export is **not committed** — it is large and redistributable only from
its source. Download `Amazon Sale Report.csv` from
[Kaggle](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data)
and place it in the repository root. The scripts fail with that instruction if
it is missing.

`amazon_sales_clean_sample.csv` holds the first 50 cleaned rows, committed so
the output schema is readable without the download.

### Columns used

| Column | Meaning |
|---|---|
| `Date` | Sale date (day-first format) |
| `Status` | Order status; `Cancelled` drives the cancellation flag |
| `Fulfilment` | `Amazon` or `Merchant` |
| `Category`, `Size`, `Style`, `SKU` | Product attributes |
| `Courier Status` | Delivery state |
| `Qty` | Units in the order |
| `Amount` | Sale value |
| `B2B` | Business-to-business flag |

### Cleaning

Handled in `src/preprocessing.py`:

- Rows missing `Amount` or `Qty` are dropped — every test depends on both
- Non-positive amounts and quantities are removed, along with quantities above
  100 as data-entry errors
- Categorical columns are trimmed and title-cased so `" kurta "` and `"Kurta"`
  are one category
- `B2B` is normalised from whichever of `True`/`1`/`yes` the export used
- Missing `Courier Status`, `Size` and `Style` become explicit labels rather
  than being dropped

Derived columns: `RevenuePerUnit`, `OrderSizeBucket`, `IsCancelled`,
`IsDelivered`, `SizeRank`, and calendar parts of `Date`.

## Structure

```
src/preprocessing.py      loading, cleaning, derived columns
src/hypothesis_tests.py   the five tests and their figures
figures/                  generated output
Online Sales.pbix         Power BI dashboard
amazon_sales_clean_sample.csv   50 cleaned rows, for reference
requirements.txt          pinned versions
```

## Limitations

- **One seller, one period.** Findings describe this account's apparel sales,
  not the Amazon marketplace generally.
- **B2B is a very small sample** (0.77% of revenue), so B2B-versus-B2C
  comparisons rest on far fewer orders on the B2B side.
- **Cancellation is inferred from `Status`** containing "cancel"; the data does
  not record who cancelled or why.
- **No cost data**, so "revenue lost" is lost sale value, not lost margin.
- **Association, not causation.** Test 5 shows cancellation and fulfilment
  method are related; it cannot show that one causes the other.
