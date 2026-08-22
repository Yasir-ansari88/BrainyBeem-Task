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

with st.sidebar:
    st.markdown("# Navigation")
    st.markdown("### Daily Tasks")
    section = st.radio(
        "Task-1 Load & Browse",
        "Task-2 Summary State"
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

# if section == "task-1 Load & Browse":
