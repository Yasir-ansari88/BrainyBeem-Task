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

def manal_max(values):
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
        "max" : manal_max(values),
        "std" : manual_std(values, mean_v)
    }
    

    

with st.sidebar:
    st.markdown("# Navigation")
    section = st.radio(
        "Daily Task",   
        [
            "Task-1 Load & Browse",
            "Task-2 Summary Statistics",
        ]

    )

st.title("Heart Disease Prediction — Data Explorer")
st.caption(
    "All parsing, statistics, and missing-value handling below are implemented "
    "manually (no pandas.read_csv, no .describe(), no .isna()/.dropna())."
)

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