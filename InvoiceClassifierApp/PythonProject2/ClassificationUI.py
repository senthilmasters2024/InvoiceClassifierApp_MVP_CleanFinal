import streamlit as st
import os

# Define target folders (modify if needed)
# Dynamically locate the root of the .NET output directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # where Python script is
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

TRAIN_PATH = os.path.join(BASE_DIR, "TrainData")
INVOICE_PATH = os.path.join(BASE_DIR, "Invoices")

# Ensure directories exist
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(INVOICE_PATH, exist_ok=True)

# Predefined label options
predefined_labels = ["healthcare", "craftsman", "capitalincome"]

st.title("🧾 Invoice Classifier Uploader")

# Upload Training Data
st.header("📚 Upload Training Data")
label = st.selectbox("Select Label/Category", predefined_labels)

training_files = st.file_uploader("Upload .pdf or .txt files for this label", accept_multiple_files=True,
                                  type=["pdf", "txt"])

if label and training_files:
    label_dir = os.path.join(TRAIN_PATH, label.lower().replace(" ", "_"))
    os.makedirs(label_dir, exist_ok=True)

    for file in training_files:
        file_path = os.path.join(label_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success(f"✅ Uploaded {len(training_files)} files to category '{label}'")

# Upload Invoices to Classify
st.header("🧾 Upload Invoices to Classify")

invoice_files = st.file_uploader("Upload invoice documents (.pdf only)", accept_multiple_files=True, type=["pdf"])

if invoice_files:
    for file in invoice_files:
        file_path = os.path.join(INVOICE_PATH, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success(f"✅ Uploaded {len(invoice_files)} invoice(s) for classification")

# Option to show existing folders
st.header("📁 Folder Preview")
if st.button("Show Uploaded Data"):
    st.write("📂 Training Data Labels:")
    for label_folder in os.listdir(TRAIN_PATH):
        st.markdown(f"- **{label_folder}**: {len(os.listdir(os.path.join(TRAIN_PATH, label_folder)))} files")

    st.write("📂 Invoices to Classify:")
    st.markdown(f"- {len(os.listdir(INVOICE_PATH))} files")

# Final note
st.info("After uploading, you can now run your .NET classification app.")
