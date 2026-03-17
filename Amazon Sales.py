import pandas as pd
import numpy as np

## 1.Parse and extract date features
df = pd.read_csv('Amazon Sale Report.csv')

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['MonthName'] = df['Date'].dt.strftime('%B')
df['WeekNumber'] = df['Date'].dt.isocalendar().week.astype(int)
df['Quarter'] = df['Date'].dt.quarter
df['DayOfWeek'] = df['Date'].dt.day_name

## 2.Standardise string columns
# Strip whitespace and fix inconsistent casing
str_cols = ['Category', 'Size', 'Status', 'Fulfilment',
            'Style', 'Courier Status', 'currency']

for col in str_cols:
    df[col] = df[col].str.strip().str.title()

# Check unique values for unexpected entries
for col in str_cols:
    print(f"\n{col}: {df[col].unique()}")

## 3. Handle missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Amount and Qty are critical - drop rows where both are missing
df = df.dropna(subset=['Amount', 'Qty'])

# Fill missing Courier Status with 'Unknown'
df['Courier Status'] = df['Courier Status'].fillna('Unknown')

# Fill missing Size and Style with 'Unspecified'
df['Size'] = df['Size'].fillna('Unspecified')
df['Style'] = df['Style'].fillna('Unspecified')

## 4. Validate numeric ranges
print("\nNegative quantities:", (df['Qty'] < 0).sum())
print("Negative amounts:", (df['Amount'] < 0).sum())
print("Zero amounts:", (df['Amount'] == 0).sum())

# Keep only valid rows
df = df[df['Qty'] > 0]
df = df[df['Amount'] > 0]

## 5.Standardise B2B column
# B2B might come in as True/False, TRUE/FALSE, 1/0, or Yes/No
df['B2B'] = df['B2B'].astype(str).str.strip().str.lower()
df['B2B'] = df['B2B'].map({
    'true': True, '1': True, 'yes': True,
    'false': False, '0': False, 'no': False
})
df['B2B'] = df['B2B'].astype(bool)
df['B2BLabel'] = df['B2B'].map({True: 'B2B', False: 'B2C'})

## 6. Create derived columns
# Revenue per unit
df['RevenuePerUnit'] = df['Amount'] / df['Qty']

# Order size bucket
df['OrderSizeBucket'] = pd.cut(
    df['Qty'],
    bins=[0, 1, 5, 20, 9999],
    labels=['Single (1)', 'Small (2-5)', 'Medium (6-20)', 'Bulk (20+)']
)

# Cancelled flag
df['IsCancelled'] = (df['Status'].str.lower().str.contains('cancel')).astype(int)

# Fulfilment speed tier (if Courier Status has values like Shipped, Delivered etc.)
df['IsDelivered'] = (df['Courier Status'].str.lower() == 'delivered').astype(int)

# Size rank for ordering in visuals
size_order = {'XS': 1, 'S': 2, 'M': 3, 'L': 4, 'XL': 5, 'XXL': 6, 'Unspecified': 7}
df['SizeRank'] = df['Size'].map(size_order).fillna(99)

print(f"\nClean dataset: {len(df):,} rows")
#df.to_csv('amazon_sales_clean.csv', index=False)

print("\n------------------")
print("Testing Hypothesis")
print("------------------\n")
### Statistical Tests
"""Test 1 — Spearman Correlation.
   Does order quantity correlate with sale amount — and does this relationship differ for B2B vs B2C?"""
from scipy import stats
import matplotlib.pyplot as plt

# Overall correlation
corr, p = stats.spearmanr(df['Qty'], df['Amount'])
print(f"Overall Spearman r={corr:.3f}, p={p:.4f}")

# Split by B2B vs B2C
for label in ['B2B', 'B2C']:
    subset = df[df['B2BLabel'] == label]
    c, pv  = stats.spearmanr(subset['Qty'], subset['Amount'])
    print(f"{label:4s} — r={c:.3f}, p={pv:.4f}  (n={len(subset):,})")

# Scatter plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, label, color in zip(axes, ['B2B', 'B2C'], ['steelblue', 'coral']):
    subset = df[df['B2BLabel'] == label]
    ax.scatter(subset['Qty'], subset['Amount'],
               alpha=0.3, s=15, color=color)
    m, b = np.polyfit(subset['Qty'], subset['Amount'], 1)
    x_line = np.linspace(subset['Qty'].min(), subset['Qty'].max(), 100)
    ax.plot(x_line, m * x_line + b, color='black', linewidth=2)
    c, _ = stats.spearmanr(subset['Qty'], subset['Amount'])
    ax.set_title(f"{label}  (r={c:.3f})")
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Amount')

#plt.suptitle('Qty vs Amount — B2B vs B2C', fontsize=13)
#plt.tight_layout()
#plt.savefig('test1_spearman.png', dpi=150)
#plt.show()


"""Test 2 — Mann-Whitney U.
   Do B2B orders generate significantly higher revenue per unit than B2C orders?"""
b2b_rev = df[df['B2BLabel'] == 'B2B']['RevenuePerUnit'].dropna()
b2c_rev = df[df['B2BLabel'] == 'B2C']['RevenuePerUnit'].dropna()

stat, p = stats.mannwhitneyu(b2b_rev, b2c_rev, alternative='two-sided')

print(f"B2B median revenue/unit: {b2b_rev.median():.2f}")
print(f"B2C median revenue/unit: {b2c_rev.median():.2f}")
diff = (b2b_rev.median() - b2c_rev.median()) / b2c_rev.median() * 100
print(f"Difference: {diff:+.1f}%")
print(f"p={p:.4f} → Significant: {'YES ✅' if p < 0.05 else 'NO ❌'}")

# Visualize
plt.figure(figsize=(7, 5))
plt.boxplot(
    [b2b_rev.clip(upper=b2b_rev.quantile(0.95)),
     b2c_rev.clip(upper=b2c_rev.quantile(0.95))],
    labels=['B2B', 'B2C'], patch_artist=True,
    boxprops=dict(facecolor='mediumseagreen', alpha=0.6)
)
#plt.ylabel('Revenue per Unit')
#plt.title(f'B2B vs B2C Revenue per Unit  (p={p:.4f})')
#plt.tight_layout()
#plt.savefig('test2_mannwhitney.png', dpi=150)
#plt.show()


"""Test 3 — Kruskal-Wallis + Pairwise Mann-Whitney.
   Does sale Amount differ significantly across product Categories?"""
from itertools import combinations

groups = {
    name: grp['Amount'].dropna().values
    for name, grp in df.groupby('Category')
    if len(grp) >= 10
}

stat, p = stats.kruskal(*groups.values())
print(f"Kruskal-Wallis p={p:.4f} → Significant: {'YES ✅' if p < 0.05 else 'NO ❌'}\n")

print("Median Amount per Category:")
for name, vals in sorted(groups.items(),
                          key=lambda x: -pd.Series(x[1]).median()):
    print(f"  {name:25s}: {pd.Series(vals).median():,.2f}  (n={len(vals)})")

print("\nPairwise comparisons:")
for a, b in combinations(groups.keys(), 2):
    _, p_pair = stats.mannwhitneyu(
        groups[a], groups[b], alternative='two-sided')
    sig = '✅ different' if p_pair < 0.05 else '❌ similar'
    print(f"  {a:20s} vs {b:20s} → p={p_pair:.4f}  {sig}")


"""Test 4 — Levene's Test.
   Is the spread of sale amounts wider for Bulk orders than Single or Small orders?"""
bucket_groups = [
    df[df['OrderSizeBucket'] == bucket]['Amount'].dropna()
    for bucket in ['Single (1)', 'Small (2-5)',
                   'Medium (6-20)', 'Bulk (20+)']
    if len(df[df['OrderSizeBucket'] == bucket]) >= 10
]

stat, p = stats.levene(*bucket_groups)
print(f"Levene's test p={p:.4f}")
print(f"Amount spread differs by order size: {'YES ✅' if p < 0.05 else 'NO ❌'}\n")

for bucket in ['Single (1)', 'Small (2-5)', 'Medium (6-20)', 'Bulk (20+)']:
    subset = df[df['OrderSizeBucket'] == bucket]['Amount']
    if len(subset) >= 10:
        iqr = subset.quantile(0.75) - subset.quantile(0.25)
        print(f"  {bucket:15s} → IQR: {iqr:,.2f}  Median: {subset.median():,.2f}")


"""Test 5 — Chi-Square Test of Independence.
Is cancellation rate associated with fulfilment method?"""
contingency = pd.crosstab(df['Fulfilment'], df['Status'])
print(contingency)

chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square p={p:.4f}")
print(f"Cancellation linked to fulfilment: {'YES ✅' if p < 0.05 else 'NO ❌'}")

# Cancellation rate per fulfilment method
cancel_rate = df.groupby('Fulfilment')['IsCancelled'].mean().sort_values(ascending=False)
print("\nCancellation rate by fulfilment method:")
print((cancel_rate * 100).round(1).astype(str) + '%')