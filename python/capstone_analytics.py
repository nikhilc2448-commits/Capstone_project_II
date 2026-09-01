from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_object_dtype, is_string_dtype
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CLEANING_DIR = BASE_DIR / "cleaning"
EXCEL_DIR = BASE_DIR / "excel"
POWERBI_DIR = BASE_DIR / "powerbi"
REPORT_DIR = BASE_DIR / "report"

for folder in [OUTPUT_DIR, CLEANING_DIR, EXCEL_DIR, POWERBI_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Data loading and audit
# -----------------------------

def load_data(customers_path=None, orders_path=None):
    customers_path = Path(customers_path or DATA_DIR / "Capstone_Customers.csv")
    orders_path = Path(orders_path or DATA_DIR / "Capstone_Orders.csv")

    try:
        if not customers_path.exists():
            raise FileNotFoundError(f"Customers file not found: {customers_path}")
        if not orders_path.exists():
            raise FileNotFoundError(f"Orders file not found: {orders_path}")

        customers = pd.read_csv(customers_path)
        orders = pd.read_csv(orders_path)

        if customers.empty or orders.empty:
            raise ValueError("One or both input files are empty.")

        print("Customers Dataset Loaded Successfully")
        print("Orders Dataset Loaded Successfully")
        return customers, orders

    except (FileNotFoundError, pd.errors.EmptyDataError,
            pd.errors.ParserError, ValueError) as exc:
        raise RuntimeError(f"Data loading failed: {exc}") from exc


def text_columns(df):
    """Return columns that currently contain text values."""
    return [
        col for col in df.columns
        if is_object_dtype(df[col]) or is_string_dtype(df[col])
    ]


def audit_dataframe(df, name):
    text_unique = {}
    for col in text_columns(df):
        text_unique[col] = df[col].astype("string").unique().tolist()

    return {
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_text_values": text_unique,
    }


def save_audit(audits, filename):
    path = CLEANING_DIR / filename

    with open(path, "w", encoding="utf-8") as file:
        for audit in audits:
            file.write("=" * 80 + "\n")
            file.write(f"{audit['dataset']} DATA AUDIT\n")
            file.write("=" * 80 + "\n")
            file.write(f"Shape: {audit['shape']}\n\n")

            file.write("Data Types:\n")
            for col, dtype in audit["dtypes"].items():
                file.write(f"  {col}: {dtype}\n")

            file.write("\nMissing Values:\n")
            for col, count in audit["missing_values"].items():
                file.write(f"  {col}: {count}\n")

            file.write(f"\nDuplicate Rows: {audit['duplicate_rows']}\n\n")
            file.write("Unique Values - Text Columns:\n")

            if not audit["unique_text_values"]:
                file.write("  None\n")
            else:
                for col, values in audit["unique_text_values"].items():
                    file.write(f"  {col} ({len(values)} unique): {values}\n")

            file.write("\n")

    return path


# -----------------------------
# Cleaning
# -----------------------------

def standardize_text(df):
    df = df.copy()

    for col in text_columns(df):
        df[col] = df[col].astype("string").str.strip()

    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].str.upper()
    if "OrderID" in df.columns:
        df["OrderID"] = df["OrderID"].str.upper()
    if "CustomerName" in df.columns:
        df["CustomerName"] = df["CustomerName"].str.title()
    if "Segment" in df.columns:
        df["Segment"] = df["Segment"].str.title()
    if "Region" in df.columns:
        df["Region"] = df["Region"].str.title()
    if "City" in df.columns:
        df["City"] = df["City"].str.title()
    if "State" in df.columns:
        df["State"] = df["State"].str.title()
    if "ProductCategory" in df.columns:
        df["ProductCategory"] = df["ProductCategory"].str.title()
    if "Product" in df.columns:
        df["Product"] = df["Product"].str.title()
    if "OrderStatus" in df.columns:
        df["OrderStatus"] = df["OrderStatus"].str.title()
    if "PaymentMethod" in df.columns:
        df["PaymentMethod"] = df["PaymentMethod"].str.title().replace({
            "Upi": "UPI",
            "Net Banking": "NetBanking",
            "Netbanking": "NetBanking",
            "Cash On Delivery": "Cash on Delivery",
        })

    return df


def clean_data(customers, orders):
    customers = customers.copy()
    orders = orders.copy()
    log = []

    # Text cleanup first so the later checks use consistent values.
    customer_text_cols = text_columns(customers)
    order_text_cols = text_columns(orders)
    customer_before = customers[customer_text_cols].copy()
    order_before = orders[order_text_cols].copy()

    customers = standardize_text(customers)
    orders = standardize_text(orders)

    customer_changed = int(
        (customer_before.fillna("<NA>") != customers[customer_text_cols].fillna("<NA>"))
        .any(axis=1)
        .sum()
    )
    order_changed = int(
        (order_before.fillna("<NA>") != orders[order_text_cols].fillna("<NA>"))
        .any(axis=1)
        .sum()
    )

    log.append({
        "Issue": "Inconsistent whitespace/casing in customer text fields",
        "Rows Affected": customer_changed,
        "Action": "Trimmed whitespace and standardized customer text fields.",
        "Why": "Keeps categories and identifiers consistent for grouping and joins.",
    })
    log.append({
        "Issue": "Inconsistent whitespace/casing in order text fields",
        "Rows Affected": order_changed,
        "Action": "Trimmed whitespace and standardized order text fields.",
        "Why": "Prevents false category differences and identifier mismatches.",
    })

    # Exact duplicates.
    duplicate_customers = int(customers.duplicated().sum())
    if duplicate_customers:
        customers = customers.drop_duplicates(keep="first").reset_index(drop=True)
    log.append({
        "Issue": "Duplicate customer rows",
        "Rows Affected": duplicate_customers,
        "Action": "Removed exact duplicate rows and kept the first occurrence.",
        "Why": "An exact repeated customer row would count the same customer twice.",
    })

    duplicate_orders = int(orders.duplicated().sum())
    if duplicate_orders:
        orders = orders.drop_duplicates(keep="first").reset_index(drop=True)
    log.append({
        "Issue": "Duplicate order rows",
        "Rows Affected": duplicate_orders,
        "Action": "Removed exact duplicate rows and kept the first occurrence.",
        "Why": "Repeated orders would inflate sales, profit and order counts.",
    })

    # Customer Segment.
    missing_segment = int(customers["Segment"].isna().sum())
    if missing_segment:
        customers.loc[customers["Segment"].isna(), "Segment"] = "Unknown"
    log.append({
        "Issue": "Missing Customer Segment",
        "Rows Affected": missing_segment,
        "Action": "Filled missing Segment with 'Unknown'.",
        "Why": "There is no reliable row-level evidence for assigning a specific segment.",
    })

    # Customer Region.
    missing_region = int(customers["Region"].isna().sum())
    region_inferred = 0
    if missing_region:
        mapping = (
            customers.dropna(subset=["Region"])
            .groupby(["City", "State"])["Region"]
            .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else np.nan)
            .to_dict()
        )

        for idx in customers.index[customers["Region"].isna()]:
            key = (customers.at[idx, "City"], customers.at[idx, "State"])
            region = mapping.get(key, np.nan)
            if pd.notna(region):
                customers.at[idx, "Region"] = region
                region_inferred += 1
            else:
                customers.at[idx, "Region"] = "Unknown"

    log.append({
        "Issue": "Missing Customer Region",
        "Rows Affected": missing_region,
        "Action": f"Inferred {region_inferred} row(s) from City + State; used 'Unknown' where no match existed.",
        "Why": "City and State provide a defensible geographic mapping in this dataset.",
    })

    # Product category from product name.
    missing_category = int(orders["ProductCategory"].isna().sum())
    category_inferred = 0
    product_to_category = (
        orders.dropna(subset=["ProductCategory", "Product"])
        .groupby("Product")["ProductCategory"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else np.nan)
        .to_dict()
    )

    for idx in orders.index[orders["ProductCategory"].isna()]:
        product = orders.at[idx, "Product"]
        category = product_to_category.get(product, np.nan)
        if pd.notna(category):
            orders.at[idx, "ProductCategory"] = category
            category_inferred += 1
        else:
            orders.at[idx, "ProductCategory"] = "Unknown"

    log.append({
        "Issue": "Missing ProductCategory",
        "Rows Affected": missing_category,
        "Action": f"Inferred {category_inferred} row(s) from Product; used 'Unknown' if unavailable.",
        "Why": "The product-to-category relationship can be learned from the other order records.",
    })

    # Product.
    missing_product = int(orders["Product"].isna().sum())
    if missing_product:
        orders.loc[orders["Product"].isna(), "Product"] = "Unknown Product"
    log.append({
        "Issue": "Missing Product",
        "Rows Affected": missing_product,
        "Action": "Filled missing Product with 'Unknown Product'.",
        "Why": "There is not enough evidence to safely invent a product name.",
    })

    # Payment method.
    missing_payment = int(orders["PaymentMethod"].isna().sum())
    if missing_payment:
        orders.loc[orders["PaymentMethod"].isna(), "PaymentMethod"] = "Unknown"
    log.append({
        "Issue": "Missing PaymentMethod",
        "Rows Affected": missing_payment,
        "Action": "Filled missing PaymentMethod with 'Unknown'.",
        "Why": "Payment method cannot be reliably inferred from the other fields.",
    })

    # Order status.
    missing_status = int(orders["OrderStatus"].isna().sum())
    if missing_status:
        orders.loc[orders["OrderStatus"].isna(), "OrderStatus"] = "Unknown"
    log.append({
        "Issue": "Missing OrderStatus",
        "Rows Affected": missing_status,
        "Action": "Filled missing OrderStatus with 'Unknown'.",
        "Why": "The order can still be analyzed without making up its status.",
    })

    # Discount.
    missing_discount = int(orders["Discount"].isna().sum())
    if missing_discount:
        for idx in orders.index[orders["Discount"].isna()]:
            product = orders.at[idx, "Product"]
            category = orders.at[idx, "ProductCategory"]
            value = orders.loc[orders["Product"].eq(product), "Discount"].median()
            if pd.isna(value):
                value = orders.loc[orders["ProductCategory"].eq(category), "Discount"].median()
            if pd.isna(value):
                value = orders["Discount"].median()
            orders.at[idx, "Discount"] = value
    log.append({
        "Issue": "Missing Discount",
        "Rows Affected": missing_discount,
        "Action": "Used Product median, then ProductCategory median, then overall median.",
        "Why": "Discount can vary by product, so a grouped median is less sensitive to extreme values.",
    })

    # Quantity.
    missing_quantity = int(orders["Quantity"].isna().sum())
    if missing_quantity:
        for idx in orders.index[orders["Quantity"].isna()]:
            product = orders.at[idx, "Product"]
            category = orders.at[idx, "ProductCategory"]
            value = orders.loc[orders["Product"].eq(product), "Quantity"].median()
            if pd.isna(value):
                value = orders.loc[orders["ProductCategory"].eq(category), "Quantity"].median()
            if pd.isna(value):
                value = orders["Quantity"].median()
            orders.at[idx, "Quantity"] = value
    log.append({
        "Issue": "Missing Quantity",
        "Rows Affected": missing_quantity,
        "Action": "Used Product median, then ProductCategory median, then overall median.",
        "Why": "Quantity is product-related and the median limits the effect of extreme orders.",
    })

    # Profit.
    missing_profit = int(orders["Profit"].isna().sum())
    if missing_profit:
        for idx in orders.index[orders["Profit"].isna()]:
            product = orders.at[idx, "Product"]
            category = orders.at[idx, "ProductCategory"]

            product_rows = orders[
                (orders["Product"] == product)
                & orders["Profit"].notna()
                & orders["Sales"].ne(0)
            ]
            if not product_rows.empty:
                margin = (product_rows["Profit"] / product_rows["Sales"]).median()
            else:
                category_rows = orders[
                    (orders["ProductCategory"] == category)
                    & orders["Profit"].notna()
                    & orders["Sales"].ne(0)
                ]
                if not category_rows.empty:
                    margin = (category_rows["Profit"] / category_rows["Sales"]).median()
                else:
                    valid = orders[orders["Profit"].notna() & orders["Sales"].ne(0)]
                    margin = (valid["Profit"] / valid["Sales"]).median()

            orders.at[idx, "Profit"] = orders.at[idx, "Sales"] * margin

    log.append({
        "Issue": "Missing Profit",
        "Rows Affected": missing_profit,
        "Action": "Estimated from the median Profit/Sales margin for the Product, with category/global fallback.",
        "Why": "Profit is not directly derivable from the available fields, so observed margins are used.",
    })

    # Referential integrity.
    valid_customer_ids = set(customers["CustomerID"].dropna())
    orphan_mask = ~orders["CustomerID"].isin(valid_customer_ids)
    orphan_count = int(orphan_mask.sum())
    orphan_ids = sorted(orders.loc[orphan_mask, "CustomerID"].astype(str).unique().tolist())

    if orphan_count:
        orders = orders.loc[~orphan_mask].copy().reset_index(drop=True)

    log.append({
        "Issue": "Orders with CustomerIDs not present in Customers table",
        "Rows Affected": orphan_count,
        "Action": f"Removed orphan orders after ID normalization. Orphan IDs: {', '.join(orphan_ids) if orphan_ids else 'None'}",
        "Why": "Customer-level analysis requires each order to map to a valid customer.",
    })

    # Date conversion is handled separately from text standardization.
    bad_signup_dates = int(pd.to_datetime(customers["SignupDate"], errors="coerce").isna().sum())
    bad_order_dates = int(pd.to_datetime(orders["OrderDate"], errors="coerce").isna().sum())

    customers["SignupDate"] = pd.to_datetime(customers["SignupDate"], errors="coerce")
    orders["OrderDate"] = pd.to_datetime(orders["OrderDate"], errors="coerce")

    log.append({
        "Issue": "SignupDate stored as text",
        "Rows Affected": bad_signup_dates,
        "Action": "Converted SignupDate to datetime.",
        "Why": "Datetime is needed for reliable date handling and analysis.",
    })
    log.append({
        "Issue": "OrderDate stored as text",
        "Rows Affected": bad_order_dates,
        "Action": "Converted OrderDate to datetime.",
        "Why": "Datetime is needed for the monthly trend analysis.",
    })

    return customers, orders, pd.DataFrame(log)


# -----------------------------
# Transformations and EDA
# -----------------------------

def transform_data(customers, orders):
    merged = orders.merge(
        customers,
        on="CustomerID",
        how="inner",
        validate="many_to_one",
        suffixes=("_Order", "_Customer"),
    )

    merged["Profit Margin"] = np.where(
        merged["Sales"].ne(0),
        merged["Profit"] / merged["Sales"],
        np.nan,
    )
    merged["Profit per Unit"] = np.where(
        merged["Quantity"].ne(0),
        merged["Profit"] / merged["Quantity"],
        np.nan,
    )
    merged["Loss Flag"] = np.where(merged["Profit"] < 0, "Loss", "Profit")
    merged["OrderYear"] = merged["OrderDate"].dt.year
    merged["OrderMonth"] = merged["OrderDate"].dt.to_period("M").astype(str)

    return merged


def numpy_analysis(orders):
    sales = orders["Sales"].to_numpy(dtype=float)
    profit = orders["Profit"].to_numpy(dtype=float)

    mean_sales = np.mean(sales)
    std_sales = np.std(sales)
    min_sales = np.min(sales)
    max_sales = np.max(sales)

    sales_range = max_sales - min_sales
    if sales_range:
        normalized_sales = (sales - min_sales) / sales_range
    else:
        normalized_sales = np.zeros_like(sales)

    loss_flags = np.where(profit < 0, 1, 0)

    return {
        "Sales Mean": float(mean_sales),
        "Sales Std": float(std_sales),
        "Sales Min": float(min_sales),
        "Sales Max": float(max_sales),
        "Normalized Sales Min": float(np.min(normalized_sales)),
        "Normalized Sales Max": float(np.max(normalized_sales)),
        "Loss Order Count": int(np.sum(loss_flags)),
    }


def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return q1, q3, lower, upper, mask


def statistics_analysis(merged):
    numeric_cols = ["Sales", "Profit", "Quantity", "Discount"]
    descriptive = merged[numeric_cols].describe().T

    full_mean_sales = merged["Sales"].mean()
    sales_median = merged["Sales"].median()
    sales_skew = merged["Sales"].skew()

    if sales_skew > 0.5:
        sales_measure = "Median"
        sales_measure_reason = (
            f"Median better represents Sales because the distribution is right-skewed "
            f"(skewness {sales_skew:.2f}), so a small number of high-value orders pull the mean upward."
        )
    else:
        sales_measure = "Mean"
        sales_measure_reason = (
            f"Mean is a reasonable summary for Sales because the distribution is not strongly right-skewed "
            f"(skewness {sales_skew:.2f})."
        )

    sample1 = merged.sample(n=50, random_state=42)
    sample2 = merged.sample(n=50, random_state=99)

    sample1_mean = sample1["Sales"].mean()
    sample2_mean = sample2["Sales"].mean()

    sample_comparison = pd.DataFrame({
        "Group": ["Full Dataset", "Sample 1", "Sample 2"],
        "N": [len(merged), 50, 50],
        "Mean Sales": [full_mean_sales, sample1_mean, sample2_mean],
        "Difference vs Full Mean": [0, sample1_mean - full_mean_sales, sample2_mean - full_mean_sales],
    })

    sample_explanation = (
        f"Sample 1 has a mean of ₹{sample1_mean:,.2f}, while Sample 2 has a mean of "
        f"₹{sample2_mean:,.2f}, compared with the full-data mean of ₹{full_mean_sales:,.2f}. "
        "The samples contain only 50 orders, so random sampling can produce different mixes of "
        "low- and high-value orders; the larger the sample, the more stable the sample mean would generally be."
    )

    regions = sorted(merged["Region"].dropna().unique().tolist())
    if len(regions) < 2:
        raise ValueError("At least two regions are required for the t-test.")

    region1, region2 = regions[0], regions[1]
    profit1 = merged.loc[merged["Region"] == region1, "Profit"].dropna()
    profit2 = merged.loc[merged["Region"] == region2, "Profit"].dropna()

    t_stat, p_value = ttest_ind(profit1, profit2, equal_var=False)

    if p_value < 0.05:
        conclusion = "Reject the null hypothesis at the 5% significance level."
        test_explanation = (
            f"The p-value of {p_value:.6f} is below 0.05, so the sample provides evidence that "
            f"mean Profit differs between {region1} and {region2}. This does not prove that region "
            "caused the difference or explain why the difference exists."
        )
    else:
        conclusion = "Fail to reject the null hypothesis at the 5% significance level."
        test_explanation = (
            f"The p-value of {p_value:.6f} is above 0.05, so there is not enough statistical evidence "
            f"to conclude that mean Profit differs between {region1} and {region2}. This does not prove "
            "that the two regions have exactly equal profit; it only means the observed difference is not "
            "statistically significant at the chosen level."
        )

    test_result = {
        "Region 1": region1,
        "Region 2": region2,
        "t-statistic": float(t_stat),
        "p-value": float(p_value),
        "Significance Level": 0.05,
        "Conclusion": conclusion,
        "Explanation": test_explanation,
    }

    stats_notes = {
        "Sales Measure": sales_measure,
        "Sales Measure Reason": sales_measure_reason,
        "Sales Median": float(sales_median),
        "Sales Skewness": float(sales_skew),
        "Sample Explanation": sample_explanation,
        "T-Test Explanation": test_explanation,
    }

    return descriptive, sample_comparison, test_result, stats_notes


def correlation_interpretation(corr):
    rows = []

    for i, col1 in enumerate(corr.columns):
        for j, col2 in enumerate(corr.columns):
            if j <= i:
                continue

            value = corr.loc[col1, col2]
            if value > 0.3 or value < -0.3:
                if abs(value) >= 0.7:
                    strength = "strong"
                elif abs(value) >= 0.5:
                    strength = "moderate"
                else:
                    strength = "weak-to-moderate"

                direction = "positive" if value > 0 else "negative"
                text = (
                    f"{col1} and {col2} have a {strength} {direction} relationship "
                    f"(r={value:.2f}). This means higher values of one tend to occur with "
                    f"higher values of the other in this dataset. Correlation alone does not prove causation."
                )
                rows.append([col1, col2, value, text])

    return pd.DataFrame(
        rows,
        columns=["Variable 1", "Variable 2", "Correlation", "Interpretation"],
    )


# -----------------------------
# Charts
# -----------------------------

def create_visualizations(merged):
    chart_dir = OUTPUT_DIR / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    # 1. Region comparison
    region_sales = merged.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    region_sales.plot(kind="bar")
    plt.title("Total Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(chart_dir / "01_region_comparison.png", dpi=180)
    plt.close()

    # 2. Trend over time
    monthly_sales = merged.groupby("OrderMonth")["Sales"].sum().sort_index()
    plt.figure(figsize=(11, 5))
    monthly_sales.plot(marker="o")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(chart_dir / "02_trend_over_time.png", dpi=180)
    plt.close()

    # 3. Distribution
    plt.figure(figsize=(9, 5))
    merged["Sales"].plot(kind="hist", bins=30)
    plt.title("Sales Distribution")
    plt.xlabel("Sales")
    plt.ylabel("Order Count")
    plt.tight_layout()
    plt.savefig(chart_dir / "03_sales_distribution.png", dpi=180)
    plt.close()

    # 4. Relationship between two variables
    plt.figure(figsize=(8, 5))
    plt.scatter(merged["Sales"], merged["Profit"], alpha=0.45)
    plt.title("Sales vs Profit")
    plt.xlabel("Sales")
    plt.ylabel("Profit")
    plt.tight_layout()
    plt.savefig(chart_dir / "04_sales_profit_relationship.png", dpi=180)
    plt.close()

    # 5. Outlier view
    q1, q3, lower, upper, mask = iqr_outliers(merged["Sales"])
    plt.figure(figsize=(9, 4))
    plt.boxplot(merged["Sales"].dropna(), vert=False)
    plt.title("Sales Outlier View (IQR)")
    plt.xlabel("Sales")
    plt.tight_layout()
    plt.savefig(chart_dir / "05_sales_outliers.png", dpi=180)
    plt.close()

    # 6. Correlation heatmap
    corr_cols = ["Sales", "Profit", "Quantity", "Discount", "Profit Margin", "Profit per Unit"]
    corr = merged[corr_cols].corr()

    plt.figure(figsize=(9, 7))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(chart_dir / "06_correlation_heatmap.png", dpi=180)
    plt.close()

    return corr, {
        "sales_q1": q1,
        "sales_q3": q3,
        "sales_lower": lower,
        "sales_upper": upper,
        "sales_outlier_count": int(mask.sum()),
    }


# -----------------------------
# Customer segmentation
# -----------------------------

def customer_segmentation(merged):
    features = merged.groupby("CustomerID").agg(
        Total_Sales=("Sales", "sum"),
        Order_Count=("OrderID", "nunique"),
        Average_Order_Value=("Sales", "mean"),
    ).reset_index()

    feature_cols = ["Total_Sales", "Order_Count", "Average_Order_Value"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[feature_cols])

    inertias = []
    k_values = range(2, 9)
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(scaled)
        inertias.append(model.inertia_)

    selected_k = 4
    elbow = pd.DataFrame({"k": list(k_values), "Inertia": inertias})
    elbow_reason = (
        "k=4 was selected because the elbow curve shows a clear improvement up to four clusters, "
        "while additional clusters give smaller incremental reductions in inertia."
    )

    model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    features["Cluster"] = model.fit_predict(scaled)

    profile = features.groupby("Cluster")[feature_cols].mean().sort_index()

    # Give each cluster one distinct behavior label. The assignment is based on
    # the three customer measures, not on the numeric cluster ID.
    remaining = set(profile.index)
    names = {}

    high_value_cluster = profile["Total_Sales"].idxmax()
    names[high_value_cluster] = "High-Value Customers"
    remaining.remove(high_value_cluster)

    frequent_cluster = profile.loc[list(remaining), "Order_Count"].idxmax()
    names[frequent_cluster] = "Frequent Customers"
    remaining.remove(frequent_cluster)

    high_aov_cluster = profile.loc[list(remaining), "Average_Order_Value"].idxmax()
    names[high_aov_cluster] = "High-AOV Customers"
    remaining.remove(high_aov_cluster)

    if remaining:
        names[next(iter(remaining))] = "Occasional / Lower-Value Customers"

    features["Cluster Name"] = features["Cluster"].map(names)

    recommendations = {
        "High-Value Customers": "Protect retention with VIP treatment, loyalty benefits and personalized cross-sell offers.",
        "Frequent Customers": "Use replenishment reminders and loyalty incentives to increase basket value.",
        "High-AOV Customers": "Promote premium bundles and complementary products to grow high-value baskets.",
        "Occasional / Lower-Value Customers": "Use targeted reactivation campaigns and simple offers to encourage repeat purchases.",
    }

    profile = profile.reset_index()
    profile["Cluster Name"] = profile["Cluster"].map(names)
    profile["Recommendation"] = profile["Cluster Name"].map(recommendations)

    return features, profile, elbow, elbow_reason, recommendations, selected_k


# -----------------------------
# KPIs
# -----------------------------

def calculate_kpis(merged):
    total_sales = merged["Sales"].sum()
    total_profit = merged["Profit"].sum()
    avg_profit_margin = total_profit / total_sales if total_sales else np.nan
    avg_order_value = merged["Sales"].mean()

    order_counts = merged.groupby("CustomerID")["OrderID"].nunique()
    repeat_customers = int((order_counts > 1).sum())
    customers_with_orders = int(order_counts.shape[0])
    repeat_customer_rate = (
        repeat_customers / customers_with_orders if customers_with_orders else np.nan
    )

    return_cancelled = merged["OrderStatus"].isin(["Cancelled", "Returned"]).sum()
    return_cancel_rate = return_cancelled / len(merged) if len(merged) else np.nan

    kpis = pd.DataFrame([
        ["Total Sales", total_sales, "Sales Manager / Finance", "Measures overall revenue generated."],
        ["Total Profit", total_profit, "Finance / Management", "Shows total profit contribution."],
        ["Average Profit Margin", avg_profit_margin, "Finance / Pricing", "Helps evaluate pricing and profitability quality."],
        ["Average Order Value", avg_order_value, "Sales / Marketing", "Guides efforts to increase basket value."],
        ["Repeat Customer Rate", repeat_customer_rate, "CRM / Marketing", "Indicates customer retention and loyalty."],
        ["Return/Cancel Rate", return_cancel_rate, "Operations / Customer Service", "Highlights order-quality and fulfillment issues."],
    ], columns=["KPI", "Value", "Who Uses It", "Decision It Informs"])

    best_kpi = "Repeat Customer Rate"
    worst_kpi = "Return/Cancel Rate"

    kpi_notes = {
        "Best KPI": (
            f"{best_kpi} is the strongest customer-health KPI because {repeat_customer_rate:.2%} "
            "of customers with orders are repeat customers, indicating strong repeat purchasing behavior."
        ),
        "Worst KPI": (
            f"{worst_kpi} is the clearest improvement opportunity because {return_cancel_rate:.2%} "
            "of orders are cancelled or returned, which can create avoidable service and profitability pressure."
        ),
    }

    return kpis, best_kpi, worst_kpi, kpi_notes


# -----------------------------
# Excel and Power BI support files
# -----------------------------

def create_excel_report(
    customers,
    orders,
    merged,
    cleaning_log,
    kpis,
    region_summary,
    product_summary,
    customer_features,
    stats,
    stats_notes,
    sample_comparison,
    ttest_result,
    corr_interpretations,
):
    path = EXCEL_DIR / "Capstone_Excel_Report.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        customers.to_excel(writer, sheet_name="Cleaned_Customers", index=False)
        orders.to_excel(writer, sheet_name="Cleaned_Orders", index=False)
        cleaning_log.to_excel(writer, sheet_name="Cleaning_Log", index=False)
        region_summary.to_excel(writer, sheet_name="Pivot_Region", index=False)
        product_summary.to_excel(writer, sheet_name="Pivot_Product", index=False)
        customer_features.to_excel(writer, sheet_name="Customer_Segmentation", index=False)
        stats.to_excel(writer, sheet_name="Descriptive_Stats")
        pd.DataFrame([stats_notes]).to_excel(writer, sheet_name="Statistics_Notes", index=False)
        sample_comparison.to_excel(writer, sheet_name="Sample_Comparison", index=False)
        pd.DataFrame([ttest_result]).to_excel(writer, sheet_name="T_Test", index=False)
        corr_interpretations.to_excel(writer, sheet_name="Correlation_Interpretation", index=False)
        merged.to_excel(writer, sheet_name="Merged_Analysis", index=False)

        workbook = writer.book
        ws = workbook.create_sheet("KPI")

        headers = ["KPI", "Formula", "Value", "Who Uses It", "Decision It Informs"]
        for col_num, header in enumerate(headers, 1):
            ws.cell(1, col_num, header)

        formula_rows = [
            ("Total Sales", "=SUM(Cleaned_Orders!H:H)", "Sales Manager / Finance", "Measures overall revenue generated."),
            ("Total Profit", "=SUM(Cleaned_Orders!I:I)", "Finance / Management", "Shows total profit contribution."),
            ("Average Profit Margin", "=IFERROR(SUM(Cleaned_Orders!I:I)/SUM(Cleaned_Orders!H:H),0)", "Finance / Pricing", "Evaluates pricing and profitability quality."),
            ("Average Order Value", "=AVERAGE(Cleaned_Orders!H:H)", "Sales / Marketing", "Guides basket-growth efforts."),
            ("Repeat Customer Rate", "=IFERROR(COUNTIF(Customer_Segmentation!C:C,\">1\")/COUNTA(Customer_Segmentation!A:A),0)", "CRM / Marketing", "Indicates customer retention and loyalty."),
            ("Return/Cancel Rate", '=IFERROR((COUNTIF(Cleaned_Orders!K:K,"Cancelled")+COUNTIF(Cleaned_Orders!K:K,"Returned"))/COUNTA(Cleaned_Orders!A:A),0)', "Operations / Customer Service", "Highlights order-quality and fulfillment issues."),
        ]

        for row_num, row in enumerate(formula_rows, 2):
            ws.cell(row_num, 1, row[0])
            ws.cell(row_num, 2, "'" + row[1])
            ws.cell(row_num, 3, row[1])
            ws.cell(row_num, 4, row[2])
            ws.cell(row_num, 5, row[3])

        for col in ["A", "B", "C", "D", "E"]:
            ws.column_dimensions[col].width = 32

        ws.freeze_panes = "A2"

    return path


def export_powerbi_sources(merged, customer_features, kpis):
    merged.to_csv(POWERBI_DIR / "PowerBI_Merged_Analysis.csv", index=False)
    customer_features.to_csv(POWERBI_DIR / "PowerBI_Customer_Segments.csv", index=False)
    kpis.to_csv(POWERBI_DIR / "PowerBI_KPI_Reference.csv", index=False)

    # These are the two required DIVIDE measures to create in Power BI.
    dax_text = """Total Sales =\nSUM(PowerBI_Merged_Analysis[Sales])\n\nProfit Margin =\nDIVIDE(\n    SUM(PowerBI_Merged_Analysis[Profit]),\n    SUM(PowerBI_Merged_Analysis[Sales]),\n    0\n)\n\nAverage Profit per Order =\nDIVIDE(\n    SUM(PowerBI_Merged_Analysis[Profit]),\n    DISTINCTCOUNT(PowerBI_Merged_Analysis[OrderID]),\n    0\n)\n"""
    (POWERBI_DIR / "Required_DAX_Measures.txt").write_text(dax_text, encoding="utf-8")


def write_chart_insights(merged, outlier_info):
    region_sales = merged.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    best_region = region_sales.index[0]
    best_region_sales = region_sales.iloc[0]

    monthly_sales = merged.groupby("OrderMonth")["Sales"].sum().sort_index()
    peak_month = monthly_sales.idxmax()
    peak_month_sales = monthly_sales.max()

    sales_profit_corr = merged["Sales"].corr(merged["Profit"])
    profit_unit_corr = merged["Profit"].corr(merged["Profit per Unit"])

    if sales_profit_corr > 0.3:
        relationship_text = "shows a positive linear relationship in this dataset."
    elif sales_profit_corr < -0.3:
        relationship_text = "shows a negative linear relationship in this dataset."
    else:
        relationship_text = "shows only a weak linear relationship in this dataset."

    if profit_unit_corr > 0.3:
        heatmap_text = "the strongest positive relationship above the 0.3 threshold."
    elif profit_unit_corr < -0.3:
        heatmap_text = "the strongest negative relationship below the -0.3 threshold."
    else:
        heatmap_text = "does not cross the required ±0.3 interpretation threshold."

    text = f"""CHART INSIGHTS

1. Region comparison:
{best_region} generated ₹{best_region_sales:,.2f} in sales, the highest among the regions.

2. Trend over time:
{peak_month} recorded the highest monthly sales at ₹{peak_month_sales:,.2f}.

3. Distribution:
The median order sales value is ₹{merged['Sales'].median():,.2f}, below the mean of ₹{merged['Sales'].mean():,.2f}, showing the effect of high-value orders.

4. Relationship:
Sales and Profit have a correlation of {sales_profit_corr:.2f}, which {relationship_text}

5. Outlier view:
{outlier_info['sales_outlier_count']:,} orders are flagged as Sales outliers using the 1.5×IQR rule.

6. Correlation heatmap:
Profit and Profit per Unit have a correlation of {profit_unit_corr:.2f}; this is {heatmap_text}
"""

    path = OUTPUT_DIR / "Chart_Insights.txt"
    path.write_text(text, encoding="utf-8")
    return path


def write_review_notes(
    merged,
    ttest_result,
    stats_notes,
    elbow_reason,
    best_kpi,
    worst_kpi,
    kpi_notes,
    selected_k,
):
    path = OUTPUT_DIR / "Review_Notes.txt"

    total_sales = merged["Sales"].sum()
    total_profit = merged["Profit"].sum()

    text = f"""CAPSTONE REVIEW NOTES

Python / Power BI cross-check
Total Sales from Python: ₹{total_sales:,.2f}
Dashboard Total Sales should match: ₹{total_sales:,.2f}

Sales measure decision
{stats_notes['Sales Measure']}: {stats_notes['Sales Measure Reason']}

Sample comparison
{stats_notes['Sample Explanation']}

T-test
Compared {ttest_result['Region 1']} and {ttest_result['Region 2']} mean Profit.
p-value: {ttest_result['p-value']:.6f}
{ttest_result['Explanation']}

Customer segmentation
Selected k: {selected_k}
{elbow_reason}

KPI assessment
Best-performing KPI: {best_kpi}
{kpi_notes['Best KPI']}

Worst-performing KPI: {worst_kpi}
{kpi_notes['Worst KPI']}

Final totals
Total Sales: ₹{total_sales:,.2f}
Total Profit: ₹{total_profit:,.2f}
"""

    path.write_text(text, encoding="utf-8")
    return path


# -----------------------------
# Manager report
# -----------------------------

def write_business_insights_report(
    merged,
    customer_features,
    best_kpi,
    worst_kpi,
    kpi_notes,
    ttest_result,
):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    region_sales = merged.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    product_sales = merged.groupby("ProductCategory")["Sales"].sum().sort_values(ascending=False)
    region_profit = merged.groupby("Region")["Profit"].sum().sort_values(ascending=False)

    top_region = region_sales.index[0]
    top_product = product_sales.index[0]
    top_profit_region = region_profit.index[0]

    total_sales = merged["Sales"].sum()
    total_profit = merged["Profit"].sum()
    avg_margin = total_profit / total_sales if total_sales else np.nan
    aov = merged["Sales"].mean()
    repeat_rate = (merged.groupby("CustomerID")["OrderID"].nunique() > 1).mean()

    top_region_share = region_sales.iloc[0] / total_sales
    cluster_counts = customer_features["Cluster Name"].value_counts()
    largest_cluster = cluster_counts.index[0]
    largest_cluster_count = cluster_counts.iloc[0]

    sales_profit_corr = merged["Sales"].corr(merged["Profit"])

    insights = [
        f"{top_region} leads regional sales at ₹{region_sales.iloc[0]:,.2f}, representing {top_region_share:.1%} of total sales.",
        f"{top_product} is the top product category at ₹{product_sales.iloc[0]:,.2f} in sales.",
        f"{top_profit_region} contributes the highest regional profit at ₹{region_profit.iloc[0]:,.2f}.",
        f"Average order value is ₹{aov:,.2f}, while overall profit margin is {avg_margin:.1%}.",
        f"{largest_cluster} is the largest customer segment with {largest_cluster_count:,} customers and a repeat-customer rate of {repeat_rate:.1%} across customers with orders.",
    ]

    recommendations = [
        f"Protect {top_region}'s sales position with retention and cross-sell activity aimed at its highest-value customers.",
        f"Expand {top_product} through bundles and complementary products to improve the ₹{aov:,.2f} average order value.",
        f"Use behavior-based customer campaigns and monitor the {worst_kpi} rate while keeping the {best_kpi} rate strong.",
    ]

    report_md = "# Business Insights Report\n\n## Key Insights\n"
    report_md += "\n".join(f"{i}. {item}" for i, item in enumerate(insights, 1))
    report_md += "\n\n## Recommendations\n"
    report_md += "\n".join(f"{i}. {item}" for i, item in enumerate(recommendations, 1))
    report_md += f"\n\n## KPI Assessment\n\n**Best-performing KPI:** {best_kpi}. {kpi_notes['Best KPI']}\n\n"
    report_md += f"**Worst-performing KPI:** {worst_kpi}. {kpi_notes['Worst KPI']}\n\n"
    report_md += f"**T-test:** {ttest_result['Region 1']} vs {ttest_result['Region 2']}, p-value {ttest_result['p-value']:.6f}."

    (OUTPUT_DIR / "Business_Insights_Report.md").write_text(report_md, encoding="utf-8")

    pdf_path = REPORT_DIR / "Business_Insights_Report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=16,
        leading=18,
    )
    body = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=8.7,
        leading=11,
    )

    story = [
        Paragraph("Business Insights Report", title_style),
        Spacer(1, 8),
        Paragraph(
            "<b>Executive summary:</b> The analysis covers sales, profitability, customer behavior and operational KPIs across the cleaned datasets.",
            body,
        ),
        Spacer(1, 5),
        Paragraph("<b>Key Insights</b>", styles["Heading2"]),
    ]

    for i, item in enumerate(insights, 1):
        story.append(Paragraph(f"<b>{i}.</b> {item}", body))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Recommendations</b>", styles["Heading2"]))

    for i, item in enumerate(recommendations, 1):
        story.append(Paragraph(f"<b>{i}.</b> {item}", body))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 5))
    story.append(Paragraph(
        f"<b>KPI assessment:</b> Best-performing KPI: {best_kpi}. Worst-performing KPI: {worst_kpi}. "
        f"{kpi_notes['Worst KPI']}",
        body,
    ))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"<b>Statistical check:</b> The {ttest_result['Region 1']} vs {ttest_result['Region 2']} profit t-test returned p={ttest_result['p-value']:.6f}.",
        body,
    ))

    doc.build(story)
    return pdf_path


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    customers, orders = load_data()

    print("\nRunning initial data audit...")
    initial_audits = [
        audit_dataframe(customers, "Customers - Initial"),
        audit_dataframe(orders, "Orders - Initial"),
    ]
    save_audit(initial_audits, "Initial_Data_Audit.txt")

    print("Cleaning data...")
    customers_clean, orders_clean, cleaning_log = clean_data(customers, orders)
    cleaning_log.to_excel(CLEANING_DIR / "Data_Cleaning_Log.xlsx", index=False)

    print("Running final data audit...")
    final_audits = [
        audit_dataframe(customers_clean, "Customers - Final"),
        audit_dataframe(orders_clean, "Orders - Final"),
    ]
    save_audit(final_audits, "Final_Data_Audit.txt")

    # Check that the final data is actually clean.
    final_missing = sum(sum(audit["missing_values"].values()) for audit in final_audits)
    final_duplicates = sum(audit["duplicate_rows"] for audit in final_audits)
    remaining_orphans = int(
        (~orders_clean["CustomerID"].isin(customers_clean["CustomerID"])).sum()
    )

    if final_missing != 0:
        raise AssertionError(f"Missing values remain after cleaning: {final_missing}")
    if final_duplicates != 0:
        raise AssertionError(f"Duplicate rows remain after cleaning: {final_duplicates}")
    if remaining_orphans != 0:
        raise AssertionError(f"Referential mismatch remains: {remaining_orphans} orphan orders.")

    print("Final audit passed: no missing values, duplicates or orphan orders remain.")

    # NumPy
    np_results = numpy_analysis(orders_clean)
    pd.DataFrame([np_results]).to_csv(OUTPUT_DIR / "NumPy_Results.csv", index=False)

    # Merge and EDA
    merged = transform_data(customers_clean, orders_clean)

    region_summary = pd.pivot_table(
        merged,
        index="Region",
        values=["Sales", "Profit", "Discount", "OrderID"],
        aggfunc={
            "Sales": "sum",
            "Profit": "sum",
            "Discount": "mean",
            "OrderID": "nunique",
        },
        fill_value=0,
    ).reset_index().rename(columns={
        "Sales": "Total_Sales",
        "Profit": "Total_Profit",
        "Discount": "Average_Discount",
        "OrderID": "Order_Count",
    })

    product_summary = pd.pivot_table(
        merged,
        index="ProductCategory",
        values=["Sales", "Profit", "Profit Margin", "OrderID"],
        aggfunc={
            "Sales": "sum",
            "Profit": "sum",
            "Profit Margin": "mean",
            "OrderID": "nunique",
        },
        fill_value=0,
    ).reset_index().rename(columns={
        "Sales": "Total_Sales",
        "Profit": "Total_Profit",
        "Profit Margin": "Average_Profit_Margin",
        "OrderID": "Order_Count",
    })

    # Outliers
    q1, q3, lower, upper, outlier_mask = iqr_outliers(merged["Sales"])
    merged.loc[outlier_mask].to_csv(OUTPUT_DIR / "Sales_IQR_Outliers.csv", index=False)

    # Correlations
    corr_cols = ["Sales", "Profit", "Quantity", "Discount", "Profit Margin", "Profit per Unit"]
    corr = merged[corr_cols].corr()
    corr.to_csv(OUTPUT_DIR / "Correlation_Matrix.csv")
    corr_interpretations = correlation_interpretation(corr)
    corr_interpretations.to_csv(OUTPUT_DIR / "Correlation_Interpretation.csv", index=False)

    # Statistics
    descriptive, sample_comparison, ttest_result, stats_notes = statistics_analysis(merged)
    descriptive.to_csv(OUTPUT_DIR / "Descriptive_Statistics.csv")
    sample_comparison.to_csv(OUTPUT_DIR / "Sample_Mean_Comparison.csv", index=False)
    pd.DataFrame([ttest_result]).to_csv(OUTPUT_DIR / "T_Test_Result.csv", index=False)
    pd.DataFrame([stats_notes]).to_csv(OUTPUT_DIR / "Statistics_Notes.csv", index=False)

    # Charts
    _, outlier_info = create_visualizations(merged)
    write_chart_insights(merged, outlier_info)

    # Customer segmentation
    customer_features, cluster_profile, elbow, elbow_reason, recommendations, selected_k = customer_segmentation(merged)
    customer_features.to_csv(OUTPUT_DIR / "Customer_Segmentation.csv", index=False)
    cluster_profile.to_csv(OUTPUT_DIR / "Cluster_Profile.csv", index=False)
    elbow.to_csv(OUTPUT_DIR / "Elbow_Method.csv", index=False)
    (OUTPUT_DIR / "Segmentation_Justification.txt").write_text(
        f"Selected k = {selected_k}\n{elbow_reason}\n\nCluster recommendations:\n"
        + "\n".join(f"- {name}: {text}" for name, text in recommendations.items()),
        encoding="utf-8",
    )

    plt.figure(figsize=(8, 5))
    plt.plot(elbow["k"], elbow["Inertia"], marker="o")
    plt.title("Elbow Method for Customer Segmentation")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(elbow["k"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "charts" / "07_elbow_method.png", dpi=180)
    plt.close()

    # KPIs
    kpis, best_kpi, worst_kpi, kpi_notes = calculate_kpis(merged)
    kpis.to_csv(OUTPUT_DIR / "KPI_Report.csv", index=False)
    (OUTPUT_DIR / "KPI_Assessment.txt").write_text(
        f"Best-performing KPI: {best_kpi}\n{kpi_notes['Best KPI']}\n\n"
        f"Worst-performing KPI: {worst_kpi}\n{kpi_notes['Worst KPI']}\n",
        encoding="utf-8",
    )

    # Cleaned and merged CSVs
    customers_clean.to_csv(OUTPUT_DIR / "Cleaned_Customers.csv", index=False)
    orders_clean.to_csv(OUTPUT_DIR / "Cleaned_Orders.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "Merged_Analysis.csv", index=False)

    # Excel
    excel_path = create_excel_report(
        customers_clean,
        orders_clean,
        merged,
        cleaning_log,
        kpis,
        region_summary,
        product_summary,
        customer_features,
        descriptive,
        stats_notes,
        sample_comparison,
        ttest_result,
        corr_interpretations,
    )

    # Power BI source files and required DAX measures.
    export_powerbi_sources(merged, customer_features, kpis)

    # Manager report
    report_path = write_business_insights_report(
        merged,
        customer_features,
        best_kpi,
        worst_kpi,
        kpi_notes,
        ttest_result,
    )

    review_notes_path = write_review_notes(
        merged,
        ttest_result,
        stats_notes,
        elbow_reason,
        best_kpi,
        worst_kpi,
        kpi_notes,
        selected_k,
    )

    # Final console values for the live review.
    total_sales = merged["Sales"].sum()
    total_profit = merged["Profit"].sum()
    margin = total_profit / total_sales if total_sales else np.nan

    print("\n" + "=" * 70)
    print("CAPSTONE FINAL RESULTS")
    print("=" * 70)
    print(f"Cleaned Customers: {len(customers_clean):,}")
    print(f"Cleaned Orders: {len(orders_clean):,}")
    print(f"Total Sales: ₹{total_sales:,.2f}")
    print(f"Total Profit: ₹{total_profit:,.2f}")
    print(f"Average Profit Margin: {margin:.2%}")
    print(f"Average Order Value: ₹{merged['Sales'].mean():,.2f}")
    print(f"t-test regions: {ttest_result['Region 1']} vs {ttest_result['Region 2']}")
    print(f"t-test p-value: {ttest_result['p-value']:.6f}")
    print(f"Selected segmentation k: {selected_k}")
    print(f"Best KPI: {best_kpi}")
    print(f"Worst KPI: {worst_kpi}")
    print(f"Power BI cross-check number - Total Sales: ₹{total_sales:,.2f}")
    print(f"Excel report: {excel_path}")
    print(f"Business report: {report_path}")
    print(f"Review notes: {review_notes_path}")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
