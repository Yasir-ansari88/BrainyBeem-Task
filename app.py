import streamlit as st
import io
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="Heart Disease Data Explorer", layout="wide")

# task-1 

# Load the dataset in a Streamlit app without using built-in Pandas functions,
# manually parse the file format and delimiter,
# and display the dataset with pagination and sorting options.


def detect_delimiter(header_line):
    candidates = [",",";","\t","|"]
    counts = {d: header_line.count(d) for d in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","

def manual_parse_csv(raw_text):
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in normalized.split("\n") if ln.strip() != ""]

    if not lines:
        return [], [], ","

    delimiter = detect_delimiter(lines[0])

    def split_line(line):
        cells = line.split(delimiter)
        cleaned = []
        for c in cells:
            c = c.strip()
            if len(c) >= 2 and c[0] == '"' and c[-1] == '"':
                c = c[1:-1]
            cleaned.append(c)
        return cleaned

    headers = split_line(lines[0])
    rows = [split_line(ln) for ln in lines[1:]]

    fixed_rows = []
    for r in rows:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        elif len(r) > len(headers):
            r = r[: len(headers)]
        fixed_rows.append(r)

    return headers, fixed_rows, delimiter

def try_to_number(value):
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def manual_sort(rows, col_index, ascending=True):

    def sort_key(row):
        val = row[col_index]
        num = try_to_number(val)
        if num is not None:
            return (0, num)
        return (1, val.lower() if isinstance(val, str) else str(val))

    return sorted(rows, key=sort_key, reverse=not ascending)

#Task - 2
#Show summary statistics (mean, median, mode, min, max, std) 
#by implementing manual calculations without using built-in statistical functions.

def column_values_numeric(rows, col_index):
    vals = []
    for r in rows:
        n = try_to_number(r[col_index])
        if n is not None:
            vals.append(n)
    return vals

def manual_mean(values):
    if not values:
        return None
    total = 0.0
    for v in values:
        total += v
    return total / len(values)

def manual_median(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n %2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

def manual_mode(values):
    if not values:
        return None
    counts = {}
    for v in values :
        counts[v] = counts.get(v, 0) + 1
    max_count = 0 
    mode_val = None
    for v,c in counts.items():
        if c > max_count:
            max_count = c
            mode_val = v
    if max_count <= 1:
        return None
    return mode_val

def manual_min(values):
    if not values:
        return None
    m = values[0]
    for v in values[1:]:
        if v < m:
            m = v
    return m

def manual_max(values):
    if not values:
        return None
    m = values[0]
    for v in values[1:]:
        if v > m :
            m = v
    return m

def manual_std(values, mean_val):
    if not values or len(values) < 2:
        return None
    total_sq_diff = 0.0
    for v in values :
        diff = v - mean_val
        total_sq_diff += diff * diff
    variance = total_sq_diff / (len(values) - 1)
    return math.sqrt(variance)

def summarize_column(row, col_index):
    values = column_values_numeric(row, col_index)
    if not values:
        return None
    mean_v = manual_mean(values)
    return {
        "count" : len(values),
        "mean" : mean_v,
        "median" : manual_median(values),
        "mode" : manual_mode(values),
        "min" : manual_min(values),
        "max" : manual_max(values),
        "std" : manual_std(values, mean_v)
    }
    
#Task - 3
#Identify missing values without using .isna() or .dropna(), 
# apply multiple imputation techniques, and justify the method chosen in a report.

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?", "-", "missing"}

def is_missing(value):
    if value is None:
        return True
    return value.strip().lower() in MISSING_TOKENS

def find_missing_map(headers, rows):
    missing_map = {h: [] for h in headers}
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            if is_missing(val):
                missing_map[headers[ci]].append(ri)
    return missing_map

def apply_fill(rows, col_index, fill_value, as_string=False):
    new_rows = [list(r) for r in rows]
    for r in new_rows:
        if is_missing(r[col_index]):
            r[col_index] = str(fill_value) if not as_string else fill_value
    return new_rows

def impute_mean(rows, col_index):
    values = column_values_numeric(rows, col_index)
    fill = manual_mean(values)
    return apply_fill(rows, col_index, fill)


def impute_median(rows, col_index):
    values = column_values_numeric(rows, col_index)
    fill = manual_median(values)
    return apply_fill(rows, col_index, fill)

def impute_mode(rows, col_index):
    counts = {}
    for r in rows:
        v = r[col_index]
        if not is_missing(v):
            counts[v] = counts.get(v, 0) + 1
    fill = max(counts, key=counts.get) if counts else None
    return apply_fill(rows, col_index, fill, as_string=True)

def impute_forward_fill(rows, col_index):
    new_rows = [list(r) for r in rows]
    last_valid = None
    for r in new_rows:
        if is_missing(r[col_index]):
            if last_valid is not None:
                r[col_index] = last_valid
        else:
            last_valid = r[col_index]
    return new_rows

def impute_backward_fill(rows, col_index):
    new_rows = [list(r) for r in rows]
    next_valid = None
    for r in reversed(new_rows):
        if is_missing(r[col_index]):
            if next_valid is not None:
                r[col_index] = next_valid
        else:
            next_valid = r[col_index]
    return new_rows


def apply_fill(rows, col_index, fill_value, as_string=False):
    new_rows = [list(r) for r in rows]
    for r in new_rows:
        if is_missing(r[col_index]):
            r[col_index] = str(fill_value) if not as_string else fill_value
    return new_rows

IMPUTATION_METHODS = {
    "Mean imputation": impute_mean,
    "Median imputation": impute_median,
    "Mode imputation": impute_mode,
    "Forward fill": impute_forward_fill,
    "Backward fill": impute_backward_fill,
}
IMPUTATION_JUSTIFICATIONS = {
    "Mean imputation": (
        "Replaces missing values with the column mean. Best for numeric, "
        "roughly symmetric (non-skewed) columns with few outliers, since "
        "extreme values would otherwise pull the mean away from the "
        "'typical' value."
    ),
    "Median imputation": (
        "Replaces missing values with the column median. Preferred over the "
        "mean for skewed numeric columns (e.g. cholesterol, resting BP) or "
        "when outliers are present, because the median is robust to extreme "
        "values."
    ),
    "Mode imputation": (
        "Replaces missing values with the most frequent value. Appropriate "
        "for categorical columns (e.g. chest pain type, sex) where mean/"
        "median are meaningless."
    ),
    "Forward fill": (
        "Carries the last valid observation forward. Useful when row order "
        "reflects a meaningful sequence (e.g. repeated measurements over "
        "time for the same patient); less appropriate for i.i.d. tabular "
        "records like this dataset, included here for completeness/"
        "comparison."
    ),
    "Backward fill": (
        "Carries the next valid observation backward. Same caveat as "
        "forward fill — most defensible for ordered/time-series data."
    ),
}

#Task-4
#Check for duplicate rows using a custom comparison algorithm, remove them manually, 
#and analyze how duplicates impact data integrity and machine learning models.

def row_signature(row):
    sig = []
    for cell in row:
        n = try_to_number(cell)
        sig.append(n if n is not None else cell.strip().lower())
    return tuple(sig)


def find_duplicates(rows):
    seen = {}
    duplicate_indices = []
    for i, r in enumerate(rows):
        sig = row_signature(r)
        if sig in seen:
            duplicate_indices.append(i)
        else:
            seen[sig] = i
    return duplicate_indices


def remove_duplicates(rows):
    dup_idx = set(find_duplicates(rows))
    return [r for i, r in enumerate(rows) if i not in dup_idx]

#Task - 5
#Convert categorical columns (sex, cp, thal, etc.) into numerical 
# form without using built-in encoding functions, 
#manually mapping categories and handling unknown values dynamically.

def label_encode_fit(rows, col_index):
    mapping = {}
    next_code = 0
    for r in rows:
        v = r[col_index].strip()
        if v not in mapping and not is_missing(v):
            mapping[v] = next_code
            next_code += 1
    return mapping


def label_encode_transform(rows, col_index, mapping):
    new_rows = [list(r) for r in rows]
    next_code = (max(mapping.values()) + 1) if mapping else 0
    for r in new_rows:
        v = r[col_index].strip()
        if is_missing(v):
            continue
        if v not in mapping:
            mapping[v] = next_code
            next_code += 1
        r[col_index] = str(mapping[v])
    return new_rows, mapping

#Task-6 
#Normalize or standardize numerical features using manual calculations instead of built-in functions, 
#ensuring robust handling of outliers and skewed distributions.

def manual_quantile(sorted_values, q):
    if not sorted_values:
        return None
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def normalize_minmax(values):
    lo, hi = manual_min(values), manual_max(values)
    if lo is None or hi == lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def standardize_zscore(values):
    mean_v = manual_mean(values)
    std_v = manual_std(values, mean_v)
    if not std_v:
        return [0.0 for _ in values]
    return [(v - mean_v) / std_v for v in values]


def scale_robust(values):
    s = sorted(values)
    q1 = manual_quantile(s, 0.25)
    med = manual_quantile(s, 0.5)
    q3 = manual_quantile(s, 0.75)
    iqr = q3 - q1
    if not iqr:
        return [0.0 for _ in values]
    return [(v - med) / iqr for v in values]

#Task-7
#Perform one-hot encoding manually by implementing a custom encoding function 
# that dynamically adapts to new unseen categories in future data.

def one_hot_fit(rows, col_index):
    categories = []
    seen = set()
    for r in rows:
        v = r[col_index].strip()
        if not is_missing(v) and v not in seen:
            seen.add(v)
            categories.append(v)
    return sorted(categories)


def one_hot_transform(rows, col_index, categories):
    """
    Returns (new_headers_suffix, matrix) where matrix[i] is a list of 0/1
    per category. If a row's value isn't in `categories`, the category list
    is extended in place (dynamic adaptation to unseen categories) and every
    previously-built row gets a 0 backfilled for that new column.
    """
    categories = list(categories)  # local copy we can grow
    matrix = []
    for r in rows:
        v = r[col_index].strip()
        if not is_missing(v) and v not in categories:
            categories.append(v)
            for prev in matrix:
                prev.append(0)
        row_vec = [0] * len(categories)
        if not is_missing(v):
            row_vec[categories.index(v)] = 1
        matrix.append(row_vec)
    return categories, matrix

#Task - 8
#Create a new feature based on chol and thalach (e.g., "risk_factor")
# using a complex formulaincorporating domain knowledge,
# interaction terms, and percentile-based ranking.

def percentile_rank(values):
    n = len(values)
    if n <= 1:
        return [50.0 for _ in values]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for pos, idx in enumerate(order):
        ranks[idx] = 100.0 * pos / (n - 1)
    return ranks


def compute_risk_factor(chol_values, thalach_values):
    max_thalach = manual_max(thalach_values) or 1.0
    raw_scores = []
    for chol, thalach in zip(chol_values, thalach_values):
        chol_ratio = chol / 200.0
        thalach_deficit = 1 - (thalach / max_thalach)
        interaction = chol_ratio * thalach_deficit
        raw = 0.5 * chol_ratio + 0.3 * thalach_deficit + 0.2 * interaction
        raw_scores.append(raw)
    return percentile_rank(raw_scores), raw_scores


SCALING_METHODS = {
    "Min-Max normalization (0-1)": normalize_minmax,
    "Z-score standardization": standardize_zscore,
    "Robust scaling (median/IQR)": scale_robust,
}

#Task-9 
#Create histograms for all numerical features without using built-in plotting functions,
# implementing binning and frequency calculations manually.

def manual_histogram_bins(values, num_bins=10):
    lo, hi = manual_min(values), manual_max(values)
    if lo is None or hi == lo:
        return [lo], [len(values)]
    width = (hi - lo) / num_bins
    edges = [lo + i * width for i in range(num_bins + 1)]
    counts = [0] * num_bins
    for v in values:
        idx = int((v - lo) / width) if width else 0
        if idx >= num_bins:
            idx = num_bins - 1
        counts[idx] += 1
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(num_bins)]
    return centers, counts, edges, width

#Task-10
#Generate box plots to identify outliers by computing quartiles, IQR, and fences manually,
# and visualize them using raw Matplotlib drawing functions.

def manual_boxplot_stats(values):
    s = sorted(values)
    q1 = manual_quantile(s, 0.25)
    med = manual_quantile(s, 0.5)
    q3 = manual_quantile(s, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inliers = [v for v in s if lower_fence <= v <= upper_fence]
    whisker_lo = manual_min(inliers) if inliers else q1
    whisker_hi = manual_max(inliers) if inliers else q3
    outliers = [v for v in s if v < lower_fence or v > upper_fence]
    return {
        "q1": q1, "median": med, "q3": q3, "iqr": iqr,
        "lower_fence": lower_fence, "upper_fence": upper_fence,
        "whisker_lo": whisker_lo, "whisker_hi": whisker_hi,
        "outliers": outliers,
    }
def draw_manual_boxplot(ax, values, x_pos=1, width=0.5, label=""):
    stats = manual_boxplot_stats(values)
    box = patches.Rectangle((x_pos - width / 2, stats["q1"]), width,
                             stats["q3"] - stats["q1"], fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(box)
    ax.hlines(stats["median"], x_pos - width / 2, x_pos + width / 2, color="red", linewidth=1.5)
    ax.vlines(x_pos, stats["q1"], stats["whisker_lo"], color="black", linestyle="--", linewidth=1)
    ax.vlines(x_pos, stats["q3"], stats["whisker_hi"], color="black", linestyle="--", linewidth=1)
    ax.hlines(stats["whisker_lo"], x_pos - width / 4, x_pos + width / 4, color="black")
    ax.hlines(stats["whisker_hi"], x_pos - width / 4, x_pos + width / 4, color="black")
    if stats["outliers"]:
        ax.scatter([x_pos] * len(stats["outliers"]), stats["outliers"],
                    marker="o", facecolors="none", edgecolors="darkred", s=25)
    return stats

with st.sidebar:
    st.markdown("# Navigation")
    section = st.radio(
        "Daily Task",   
        [
            "Task-1 Load & Browse",
            "Task-2 Summary Statistics",
            "Task-3 Identify missing values",
            "Task-4 Check for duplicate rows",
            "Task-5 Convert categorical columns",
            "Task-6 Normalize or standardize",
            "Task-7 one-hot encoding",
            "Task-8 Feature Engineering",
            "Task-9 Histogram",
            "Task-10 Boxplot"

        ]

    )


st.title("Heart Disease Prediction — Data Explorer")
st.caption("All parsing, statistics, and missing-value handling below are implemented manually ")

uploaded = st.file_uploader("Upload the heart disease dataset (CSV/TSV/TXT)", type=["csv", "txt", "tsv"])

if uploaded is None:
    st.info("Upload a file to get started.")
    st.stop()

raw_bytes = uploaded.read()
try:
    raw_text = raw_bytes.decode("utf-8")
except UnicodeDecodeError:
    raw_text = raw_bytes.decode("latin-1")

headers, rows, delimiter = manual_parse_csv(raw_text)

if "rows" not in st.session_state or st.session_state.get("_source") != uploaded.name:
    st.session_state["rows"] = rows
    st.session_state["_source"] = uploaded.name

working_rows = st.session_state["rows"]

delim_display = {",": "comma ( , )", ";": "semicolon ( ; )", "\t": "tab", "|": "pipe ( | )"}
st.success(f"Parsed {len(working_rows)} rows × {len(headers)} columns. Detected delimiter: {delim_display.get(delimiter, delimiter)}")

if section == "Task-1 Load & Browse":
    st.subheader("Manual Dataset Viewer")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sort_col = st.selectbox("Sort by column", options=headers, key="sort_col")
    with col_b:
        sort_dir = st.radio("Direction", ["Ascending", "Descending"], horizontal=True, key="sort_dir")
    with col_c:
        page_size = st.number_input("Rows per page", min_value=5, max_value=200, value=10, step=5)

    sort_idx = headers.index(sort_col)
    sorted_rows = manual_sort(working_rows, sort_idx, ascending=(sort_dir == "Ascending"))

    total_rows = len(sorted_rows)
    total_pages = max(1, math.ceil(total_rows / page_size))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page - 1) * page_size
    end = start + page_size
    page_rows = sorted_rows[start:end]

    st.table([dict(zip(headers, r)) for r in page_rows])
    st.caption(f"Showing rows {start + 1}–{min(end, total_rows)} of {total_rows} (page {page} of {total_pages})")

elif section == "Task-2 Summary Statistics":
    st.subheader("Manually computed summary statistics")
    st.caption("mean / median / mode / min / max / std ")

    numeric_summary_rows = []
    for i, h in enumerate(headers):
        s = summarize_column(working_rows, i)
        if s is not None:
            numeric_summary_rows.append({
                "column": h,
                "count": s["count"],
                "mean": round(s["mean"], 3) if s["mean"] is not None else None,
                "median": round(s["median"], 3) if s["median"] is not None else None,
                "mode": s["mode"],
                "min": s["min"],
                "max": s["max"],
                "std": round(s["std"], 3) if s["std"] is not None else None,
            })

    if numeric_summary_rows:
        st.table(numeric_summary_rows)
    else:
        st.warning("No numeric columns detected.")

elif section == "Task-3 Identify missing values":
    st.subheader("Missing value detection (manual, no .isna()/.dropna())")

    missing_map = find_missing_map(headers, working_rows)
    missing_summary = [
        {"column": h, "missing_count": len(idxs), "missing_pct": round(100 * len(idxs) / len(working_rows), 2) if working_rows else 0}
        for h, idxs in missing_map.items()
    ]
    st.table(missing_summary)

    any_missing = any(len(idxs) > 0 for idxs in missing_map.values())

    if not any_missing:
        st.success("No missing values detected using the token set: " + ", ".join(sorted(MISSING_TOKENS)))
    else:
        st.markdown("---")
        st.subheader("Apply an imputation technique")

        cols_with_missing = [h for h, idxs in missing_map.items() if idxs]
        target_col = st.selectbox("Column to impute", options=cols_with_missing)
        method_name = st.selectbox("Imputation method", options=list(IMPUTATION_METHODS.keys()))

        st.info(IMPUTATION_JUSTIFICATIONS[method_name])

        if st.button("Apply imputation to this column"):
            col_idx = headers.index(target_col)
            method_fn = IMPUTATION_METHODS[method_name]
            new_rows = method_fn(working_rows, col_idx)
            st.session_state["rows"] = new_rows
            st.success(f"Applied '{method_name}' to column '{target_col}'. Missing count now: "
                       f"{sum(1 for r in new_rows if is_missing(r[col_idx]))}")
            st.rerun()

        st.markdown("---")
        st.subheader("📝 Method-choice report")
        st.markdown(
            """
**How to decide which imputation method to use, column by column:**

1. **Check the column type.** Numeric (age, cholesterol, resting BP) → mean/median.
   Categorical (chest pain type, thal, sex) → mode.
2. **Check for skew/outliers in numeric columns.** Plot or compare mean vs. median —
   if they diverge a lot, the column is skewed and median is the safer, outlier-robust choice.
   If roughly symmetric, mean is fine and slightly more statistically efficient.
3. **Consider missingness pattern.** If missingness is random (MCAR) and small in
   volume (<5%), simple mean/median/mode imputation is usually adequate.
   If missingness correlates with the outcome (heart disease diagnosis) or another
   variable, simple imputation can bias the model — consider model-based imputation
   (e.g. regression or KNN imputation) instead, or add a
   'was_missing' indicator flag as an extra feature.
4. **Forward/backward fill** are generally *not* appropriate for this dataset because
   rows represent independent patients, not a time-ordered sequence — they're included
   in this app only for completeness/comparison.

**Bottom line for this dataset:** use median for skewed numeric clinical measurements,
mean for roughly-normal numeric columns, and mode for categorical columns; avoid
forward/backward fill unless row order is genuinely meaningful.
            """
        )

elif section == "Task-4 Check for duplicate rows":
    st.subheader("Duplicate detection & removal")
    dup_indices = find_duplicates(working_rows)
    st.metric("Duplicate rows found", len(dup_indices))

    if dup_indices:
        preview = [{"row_index": i, **dict(zip(headers, working_rows[i]))} for i in dup_indices[:20]]
        st.write("Preview of duplicate rows (first 20):")
        st.table(preview)

        st.markdown("**Impact on data integrity / ML models if left in:**")
        st.markdown(
            f"""
- Inflates the effective sample size ({len(dup_indices)} of {len(working_rows)} rows are repeats),
  which biases summary statistics and any class-balance assumptions.
- If duplicates land across train/test splits, the same patient record can appear in both,
  causing **data leakage** and artificially inflated validation accuracy.
- Distance-based models (KNN) and tree splits can overweight duplicated patterns,
  skewing feature importance.
            """
        )

        if st.button("Remove duplicate rows"):
            st.session_state["rows"] = remove_duplicates(working_rows)
            st.success(f"Removed {len(dup_indices)} duplicate rows. New row count: {len(st.session_state['rows'])}")
            st.rerun()
    else:
        st.success("No duplicate rows detected.")

elif section == "Task-5 Convert categorical columns":
    st.subheader("Manual categorical → numeric encoding")
    cat_col = st.selectbox("Column to encode", options=headers, key="label_enc_col")
    col_idx = headers.index(cat_col)

    if st.button("Fit & apply label encoding"):
        mapping = label_encode_fit(working_rows, col_idx)
        new_rows, mapping = label_encode_transform(working_rows, col_idx, mapping)
        st.session_state["rows"] = new_rows
        st.session_state["label_map_" + cat_col] = mapping
        st.success(f"Encoded '{cat_col}' with {len(mapping)} categories.")
        st.rerun()

    map_key = "label_map_" + cat_col
    if map_key in st.session_state:
        st.write("Category → code mapping:")
        st.table([{"category": k, "code": v} for k, v in sorted(st.session_state[map_key].items(), key=lambda x: x[1])])

elif section == "Task-6 Normalize or standardize":
    st.subheader("Manual normalization / standardization")
    numeric_cols = [h for i, h in enumerate(headers) if column_values_numeric(working_rows, i)]
    scale_col = st.selectbox("Column to scale", options=numeric_cols, key="scale_col")
    method_name = st.selectbox("Method", options=list(SCALING_METHODS.keys()), key="scale_method")

    col_idx = headers.index(scale_col)
    values = column_values_numeric(working_rows, col_idx)
    scaled = SCALING_METHODS[method_name](values)

    st.write("Preview (first 10 values):")
    st.table([{"original": round(o, 3), "scaled": round(s, 4)} for o, s in list(zip(values, scaled))[:10]])

    if "Robust" in method_name:
        st.info("Robust scaling centers on the median and divides by IQR instead of mean/std, so extreme outliers don't dominate the scale — the right default for skewed clinical measurements like cholesterol.")

    if st.button("Apply scaling to this column"):
        new_rows = [list(r) for r in working_rows]
        vi = 0
        for r in new_rows:
            n = try_to_number(r[col_idx])
            if n is not None:
                r[col_idx] = str(round(scaled[vi], 6))
                vi += 1
        st.session_state["rows"] = new_rows
        st.success(f"Applied {method_name} to '{scale_col}'.")
        st.rerun()

elif section == "Task-7 one-hot encoding":
    st.subheader("Manual one-hot encoding (dynamically adapts to new categories)")

    oh_col = st.selectbox("Column to one-hot encode", options=headers, key="oh_col")
    oh_idx = headers.index(oh_col)

    if st.button("Fit & apply one-hot encoding"):
        categories = one_hot_fit(working_rows, oh_idx)
        final_categories, matrix = one_hot_transform(working_rows, oh_idx, categories)
        new_headers = headers[:oh_idx] + headers[oh_idx + 1:] + [f"{oh_col}__{c}" for c in final_categories]
        new_rows = []
        for r, vec in zip(working_rows, matrix):
            base = r[:oh_idx] + r[oh_idx + 1:]
            new_rows.append(base + [str(x) for x in vec])
        st.session_state["rows"] = new_rows
        st.session_state["headers_override"] = new_headers
        st.success(f"One-hot encoded '{oh_col}' into {len(final_categories)} columns: {', '.join(final_categories)}")
        st.rerun()

    if "headers_override" in st.session_state:
        st.info("Headers changed by one-hot encoding — re-upload")
        headers = st.session_state["headers_override"]

    st.markdown("---")
    st.markdown("**Try an unseen category (demonstrates dynamic column growth):**")
    test_val = st.text_input("Type a category value not in the original data")
    if test_val:
        categories = one_hot_fit(working_rows, oh_idx)
        sim_row = list(working_rows[0])
        sim_row[oh_idx] = test_val
        grown_categories, matrix = one_hot_transform(working_rows + [sim_row], oh_idx, categories)
        st.write(f"Categories after seeing '{test_val}': {grown_categories}")
        st.caption("Notice a new column was appended and all earlier rows were backfilled with 0 for it — that's the dynamic adaptation.")

elif section == "Task-8 Feature Engineering":
    st.subheader("Engineered feature")

    lower_headers = [h.lower() for h in headers]
    default_chol = headers[lower_headers.index("chol")] if "chol" in lower_headers else headers[0]
    default_thalach = headers[lower_headers.index("thalach")] if "thalach" in lower_headers else headers[min(1, len(headers) - 1)]

    c1, c2 = st.columns(2)
    with c1:
        chol_col = st.selectbox("Cholesterol-like column", options=headers, index=headers.index(default_chol))
    with c2:
        thalach_col = st.selectbox("Max-heart-rate-like column (thalach)", options=headers, index=headers.index(default_thalach))

    chol_idx, thalach_idx = headers.index(chol_col), headers.index(thalach_col)
    chol_vals = column_values_numeric(working_rows, chol_idx)
    thalach_vals = column_values_numeric(working_rows, thalach_idx)

    if len(chol_vals) != len(thalach_vals) or not chol_vals:
        st.warning("Selected columns need to be numeric with no missing values for this preview — impute first if needed.")
    else:
        risk, raw = compute_risk_factor(chol_vals, thalach_vals)
        st.write("Preview (first 10 patients):")
        st.table([
            {"chol": c, "thalach": t, "risk_factor (percentile 0-100)": round(rf, 1)}
            for c, t, rf in list(zip(chol_vals, thalach_vals, risk))[:10]
        ])

        if st.button("Add risk_factor as a new column"):
            new_rows = [list(r) for r in working_rows]
            for r, rf in zip(new_rows, risk):
                r.append(str(round(rf, 3)))
            st.session_state["rows"] = new_rows
            st.session_state["headers_override"] = headers + ["risk_factor"]
            st.success("Added 'risk_factor' column.")
            st.rerun()

elif section == "Task-9 Histogram":
    st.subheader("Histograms")

    plot_numeric_cols = [h for i, h in enumerate(headers) if column_values_numeric(working_rows, i)]

    st.markdown("#### Single-column histogram (manual binning)")
    hist_col = st.selectbox("Column", options=plot_numeric_cols, key="hist_col")
    num_bins = st.slider("Number of bins", 5, 30, 10)
    hvals = column_values_numeric(working_rows, headers.index(hist_col))
    centers, counts, edges, width = manual_histogram_bins(hvals, num_bins)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(centers, counts, width=width * 0.9, color="#4C72B0", edgecolor="black")
    ax.set_title(f"Histogram of {hist_col} (manual binning)")
    ax.set_xlabel(hist_col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

elif section == "Task-10 Boxplot":
    st.subheader("Box plot with manually computed quartiles / IQR / fences")
    plot_numeric_cols = [h for i, h in enumerate(headers) if column_values_numeric(working_rows, i)]
    box_col = st.selectbox("Column", options=plot_numeric_cols, key="box_col")
    bvals = column_values_numeric(working_rows, headers.index(box_col))
    fig2, ax2 = plt.subplots(figsize=(3, 4))
    stats = draw_manual_boxplot(ax2, bvals, x_pos=1, label=box_col)
    ax2.set_xlim(0.5, 1.5)
    ax2.set_xticks([1])
    ax2.set_xticklabels([box_col])
    ax2.set_title(f"Box plot of {box_col}")
    st.pyplot(fig2)
    st.table([{
        "Q1": round(stats["q1"], 2), "median": round(stats["median"], 2), "Q3": round(stats["q3"], 2),
        "IQR": round(stats["iqr"], 2), "lower_fence": round(stats["lower_fence"], 2),
        "upper_fence": round(stats["upper_fence"], 2), "outlier_count": len(stats["outliers"]),
    }])