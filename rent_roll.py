import streamlit as st
import pandas as pd
import os
import json
import shutil
import glob
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


# For OpenAI API
import openai
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

def main():
    st.title("Rent Roll Processing App")

    # Sidebar for file metadata selection
    st.sidebar.header("File Metadata")
    origin = st.sidebar.selectbox("Origin", ["Successful RedIQ Processing", "Failed RedIQ Processing"])
    template_type = st.sidebar.selectbox("Template Type", ["OneSite", "Yardi", "Resman", "Entrada", "AMSI", "Other"])
    file_type = st.sidebar.selectbox("File Type", ["Single-line Data Rows", "Multi-line Data Rows"])

    # File upload
    uploaded_file = st.file_uploader("Upload Rent Roll Excel File", type=["xlsx", "xls"])

    if uploaded_file:
        # Process the file
        process_file(uploaded_file, origin, template_type, file_type)

def process_file(uploaded_file, origin, template_type, file_type):
    st.write("Processing file:", uploaded_file.name)
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        fp = tmp_file.name
        tmp_file.write(uploaded_file.getbuffer())

    # Read the Excel file
    sheet_data = pd.read_excel(fp, sheet_name=0, header=None)
    display_df_with_unique_cols(sheet_data.head(), "Original Data:")

    # **First Bot: Standardization Process**
    # ------------------------------
    st.subheader("Step 1: Standardization Bot")
    standardized_df = standardize_data(sheet_data)
    if standardized_df is None:
        return  # Stop processing if standardization failed

    # Save standardized data to CSV
    standardized_output_path = os.path.join('outputs', f'standardized_{uploaded_file.name}.csv')
    os.makedirs('outputs', exist_ok=True)
    standardized_df.to_csv(standardized_output_path, index=False)
    st.success(f"Standardized data saved to {standardized_output_path}")

    # Standardization Review
    st.subheader("Standardization Review")
    standardization_status = st.radio("Is the standardization correct?", ["Correct", "Incorrect"], key="std_status")
    standardization_comments = st.text_area("Comments on Standardization", "", key="std_comments")

    # Save feedback
    if st.button("Submit Standardization Feedback"):
        save_feedback(uploaded_file.name, origin, template_type, file_type,
                      standardization_status, standardization_comments, "Standardization")
        st.success("Standardization feedback submitted.")

        # Only proceed with LLM processing if standardization is marked as correct
        if standardization_status == "Correct":
            # **Second Bot: LLM Processing**
            # ------------------------------
            st.subheader("Step 2: LLM Processing Bot")
            llm_df = llm_processing(standardized_df)

            if llm_df is not None:
                # Save LLM output to CSV
                llm_output_path = os.path.join('outputs', f'llm_output_{uploaded_file.name}.csv')
                llm_df.to_csv(llm_output_path, index=False)
                st.success(f"LLM output data saved to {llm_output_path}")

                # LLM Output Review
                st.subheader("LLM Output Review")
                llm_status = st.radio("Is the LLM output correct?", ["Correct", "Incorrect"], key="llm_status")
                llm_comments = st.text_area("Comments on LLM Output", "", key="llm_comments")

                # Save feedback
                if st.button("Submit LLM Feedback"):
                    save_feedback(uploaded_file.name, origin, template_type, file_type,
                                llm_status, llm_comments, "LLM Processing")
                    st.success("LLM feedback submitted.")
            else:
                st.error("LLM processing failed.")
        else:
            st.warning("Please correct the standardization issues before proceeding to LLM processing.")

def standardize_data(sheet_data):
    # Continue with processing steps
    # The code from your Jupyter Notebook adapted here

    # Define the keywords for identifying header rows
    keywords = [
        # Unit-related
        'unit', 'unit id', 'unit number', 'unit no', 'unit designation',

        # Move-in/out dates
        'move-in', 'move in', 'movein', 'move-in date', 'move in date', 'moveindate',
        'move-out', 'move out', 'moveout', 'move-out date', 'move out date', 'moveoutdate',

        # Lease-related
        'lease', 'lease start', 'lease start date', 'lease begin', 'start of lease',
        'lease end', 'lease end date', 'lease expiration', 'end of lease',

        # Rent-related
        'rent', 'market rent', 'lease rent', 'market + addl.', 'market',

        # Occupancy status
        'unit status', 'lease status', 'occupancy', 'unit/lease status',

        # Floor plan
        'floorplan', 'floor plan',

        # Square footage
        'sqft', 'sq ft', 'square feet', 'square ft', 'square footage', 'sq. ft.', 'sq.ft',
        'unit sqft', 'unit size',

        # Codes and transactions
        'code', 'charge code', 'trans code', 'transaction code', 'description'
    ]

    # Convert the data to string and lowercase for searching
    normalized_data = sheet_data.applymap(lambda x: str(x).lower() if pd.notnull(x) else '')

    # Count keyword occurrences in each row
    normalized_data['keyword_count'] = normalized_data.apply(
        lambda row: sum(row.str.contains('|'.join(keywords), regex=True)),
        axis=1
    )

    # Display rows with significant keyword counts to identify headers
    header_candidates = normalized_data[normalized_data['keyword_count'] >= 3]

    # Adjusted function to merge headers to the bottom row instead of the top
    def merge_and_select_first_header_to_bottom(df, keyword_column):
        # Sort the dataframe by index to ensure order
        df = df.sort_index()
        merged_header = None
        final_header = None

        for idx, row in df.iterrows():
            # If no header has been merged yet, start with the first row
            if merged_header is None:
                merged_header = row
                final_header = row
                continue

            # Check if the current row is adjacent to the previous one
            if idx - merged_header.name == 1:
                # Merge rows and check for keyword count improvement
                combined_row = merged_header[:-1] + " " + row[:-1]
                combined_keyword_count = sum(
                    combined_row.str.contains('|'.join(keywords), regex=True)
                )

                # Merge if the keyword count improves
                if combined_keyword_count > merged_header[keyword_column]:
                    row[:-1] = combined_row
                    row[keyword_column] = combined_keyword_count
                    final_header = row
                continue

            # Break after processing the first valid header
            break

        # Return the selected header
        return pd.DataFrame([final_header])

    # Apply the adjusted logic to merge and select the first valid header
    selected_header_df = merge_and_select_first_header_to_bottom(header_candidates, 'keyword_count')

    if selected_header_df.empty:
        st.error("Could not find a suitable header row. Please check the input file.")
        return None

    # Set the selected header as the DataFrame columns
    sheet_data.columns = selected_header_df.iloc[0, :-1]
    # Remove rows up to and including the header
    data_start_idx = selected_header_df.index[0] + 1
    df = sheet_data[data_start_idx:].reset_index(drop=True)
    # Clean up the DataFrame
    df.columns = df.columns.str.strip()

    # Now standardize the headers using the GPT model
    # Instructions for the GPT model
    instructions_prompt = """
    We aim to standardize headers across multiple documents to ensure consistency and ease of processing. Below are examples of how various column names might appear in different documents and the standardized format we want to achieve:

    Standardized Column Headers:
    - Unit: Includes variations like "Unit", "Unit Id", "Unit Number", "Unit No.", "bldg-unit"
    - Floor Plan Code: Includes variations like "Floor Plan", "Plan Code", "Floorplan"
    - Sqft: Includes variations like "Sqft", "Unit Sqft", "Square Feet", "Sq. Ft."
    - Occupancy Status: Includes variations like "Unit Status", "Lease Status", "Occupancy", "Unit/Lease Status"
    - Market Rent: Includes variations like "Market Rent", "Market + Addl.", 'Gross Market Rent'
    - Lease Start Date: Includes variations like "Lease Start", "Lease Start Date", "Start of Lease"
    - Lease Expiration: Includes variations like "Lease End", "Lease End Date", "Lease Expiration Date"
    - Move In Date: Includes variations like "Move-In", "Move In Date", "Move In"
    - Move-Out Date: Includes variations like "Move-Out", "Move Out Date", "Move Out"
    - Charge Codes: Includes variations like "Trans Code", "Charge Codes", "Description"
    - Charges or credits: this is Charges as in dollar amount (which is differeent from charge code)

    Examples of Standardized Headers:
    Unit No., Floor Plan Code, Sqft, Occupancy Status, Market Rent, Lease Start Date, Lease Expiration, Move In Date, Move-Out Date, Charge Codes

    Task:
    Your task is to analyze the headers provided in a list and map each header to its corresponding standardized column name. If a header does not match any standardized category, retain it as-is.

    Key Details:
    1. The input is a list of column names.
    2. The output must be a list of the same size, with each header mapped to its standardized name or retained as-is if no match is found.
    3. Be mindful of slight differences in naming, abbreviations, or spacing in headers. Use the examples above as a reference for mapping.
    4. If a header is unclear or does not match a category, make an educated guess or retain the original formatting with corrections for consistency.
    5. If a specific rule or example is not provided, update the header format to follow Pascal Case and ensure clarity. Apply your best judgment to map headers to the standardized list or format them consistently while preserving their original intent.

    Task:
    1. Standardize the provided headers according to the categories above.
    2. Return the result as a JSON object with a key 'standardized_headers' containing the list of standardized headers.
    3. Preserve empty strings as they are.
    4. Apply consistent formatting (Pascal Case, clarity, etc.)
    5. If no clear standardization exists, keep the original header.

    Example Input:
    ['unit', 'floorplan', 'sqft', 'unit/lease status']

    Example Output:
    {"standardized_headers": ["Unit", "Floor Plan Code", "Sqft", "Occupancy Status"]}
    """

    # Get the list of headers to standardize
    headers_to_standardize = list(df.columns)


    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    def gpt_model(instructions_prompt, header):
        """
        Use GPT model to standardize headers

        Args:
            instructions_prompt (str): Detailed instructions for header standardization
            header (list): List of headers to be standardized

        Returns:
            list: Standardized headers
        """
        # Convert header list to a string for the prompt
        headers_str = ", ".join(repr(h) for h in header)

        # Append the input headers to the instructions
        full_prompt = f"{instructions_prompt}\n\nInput Headers:\n{headers_str}"

        messages = [
            {"role": "system", "content": instructions_prompt},
            {"role": "user", "content": f"Standardize these headers: {headers_str}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}
        )

        # Parse the response
        response_content = response.choices[0].message.content

        # Try to parse the response as JSON
        try:
            standardized_headers = json.loads(response_content)['standardized_headers']
        except (json.JSONDecodeError, KeyError):
            # Fallback to parsing the text response
            standardized_headers = eval(response_content)

        return standardized_headers

    # Function to make column names unique
    def make_column_names_unique(column_names):
        cols = pd.Series(column_names)
        cols = cols.fillna('Unnamed')  # Replace NaN with 'Unnamed'
        cols = cols.replace('', 'Unnamed')  # Replace empty strings with 'Unnamed'

        # Generate a mask of duplicated column names
        duplicates = cols.duplicated(keep=False)
        counts = {}
        for idx, col in enumerate(cols):
            if col in counts:
                counts[col] += 1
                cols[idx] = f"{col}_{counts[col]}"
            else:
                counts[col] = 0
                if duplicates[idx]:
                    cols[idx] = f"{col}_{counts[col]}"

        return cols.tolist()

    with st.spinner('Standardizing headers using GPT model...'):
        standardized_headers = gpt_model(instructions_prompt, headers_to_standardize)

    # Make column names unique
    standardized_headers = make_column_names_unique(standardized_headers)

    df.columns = standardized_headers
    display_df_with_unique_cols(df.head(), "Data with Identified Headers:")

    # Drop rows where all values are null
    df = df.dropna(how='all')
    df = df.replace({r'[\*,]': ''}, regex=True)
    # Drop rows containing only strings by checking if any non-string (numeric or date) values exist in a row
    df = df[df.apply(lambda row: any(pd.to_numeric(row, errors='coerce').notnull()), axis=1)]
    df.reset_index(drop=True, inplace=True)

    # Now proceed to find breaking point and extract unit_df
    def find_breaking_point(data):
        for index, row in data.iterrows():
            if pd.notnull(row.get('Unit')):  # When `Unit` is present
                # Check for required numeric values in `Sqft` and `Market Rent`
                lease_start_exists = 'Lease Start Date' in data.columns
                if not (
                    (pd.notnull(row.get('Sqft')) and float(row.get('Sqft', 0)) < 10000) and  # Sqft should be less than 10,000
                    (pd.notnull(row.get('Market Rent')) or
                    (lease_start_exists and pd.notnull(row.get('Lease Start Date'))))
                ):
                    return index

                # Ensure `Occupancy Status` is categorical (string) only if it exists
                if 'Occupancy Status' in data.columns:
                    if pd.notnull(row.get('Occupancy Status')) and not isinstance(row.get('Occupancy Status'), str):
                        return index

                # Ensure `Charge Codes` is categorical (string) only if it exists
                if 'Charge Codes' in data.columns:
                    if pd.notnull(row.get('Charge Codes')) and not isinstance(row.get('Charge Codes'), str):
                        return index
            else:  # When `Unit` is absent
                # Ensure `Sqft` and `Market Rent` are not present
                if pd.notnull(row.get('Sqft')) or pd.notnull(row.get('Market Rent')):
                    return index

                # Check `Charge Codes` has no associated values if it exists
                if 'Charge Codes' in data.columns:
                    if pd.notnull(row.get('Charge Codes')) and row.isnull().all():
                        return index

        return None  # Return None if no breaking point is found

    breaking_point = find_breaking_point(df)
    if breaking_point is not None:
        unit_df = df[:breaking_point]
    else:
        unit_df = df

    # Clean up the DataFrame by dropping rows and columns with all NaN values
    unit_df.dropna(axis=0, how='all', inplace=True)
    unit_df.dropna(axis=1, how='all', inplace=True)

    display_df_with_unique_cols(unit_df.head(), "Final Standardized Data:")
    st.write(f"Number of unique units: {unit_df['Unit'].nunique()}")

    return unit_df

def llm_processing(unit_df):
    # Next, we move onto chunking and LLM processing
    st.write("Processing LLM Output...")

    def create_unit_based_batches(data, unit_column, batch_units=1, overlap_units=0):
        """
        Split a DataFrame into overlapping batches based on units.

        Args:
            data (pd.DataFrame): Input DataFrame to split into batches.
            unit_column (str): Column name identifying units.
            batch_units (int): Maximum number of units in each batch. Default is 1.
            overlap_units (int): Number of overlapping units between batches. Default is 0.

        Returns:
            list of pd.DataFrame: List of overlapping DataFrame batches.
        """
        batches = []
        data['unit_group'] = data[unit_column].fillna(method='ffill')  # Forward-fill NaN rows to associate them with units
        unique_units = data['unit_group'].unique()  # Identify unique units

        start = 0
        while start < len(unique_units):
            # Determine the range of unique units for the current batch
            end = start + batch_units
            selected_units = unique_units[start:end]

            # Filter the DataFrame for rows corresponding to the selected units
            batch = data[data['unit_group'].isin(selected_units)]
            batches.append(batch.drop(columns=['unit_group']))  # Drop the helper column before returning

            # Move to the next batch with overlap
            start += (batch_units - overlap_units)

        return batches

    unit_batches = create_unit_based_batches(unit_df, unit_column='Unit')
    st.write(f'Number of unit batches: {len(unit_batches)}')

    instructions_prompt =  """
        You are an AI assistant that processes rent roll data
        Given a CSV input representing rental units, your task is to meticulously extract and map all information
        for every single unit into a structured JSON format. Ensure that all details, including unit information,
        lease dates, occupancy status, market rent and all associated charges or credits, are accurately captured and correctly mapped for each unit.
        
        The names of fields in the JSON should strictly adhere to the following:

        'Unit No.'
        'Floor Plan Code'
        'Net sf'
        'Occupancy Status'
        'Enter "F" for Future Lease'
        'Market Rent'
        'Lease Start Date'
        'Lease Expiration'
        
        and we will have few others as well
        """
    # Set your OpenAI API key securely (already set in standardize_data)
    # openai.api_key = st.secrets["OPENAI_API_KEY"]

    # Directory to save individual outputs
    output_dir = 'model_outputs_parallel'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Function to process a single batch
    def process_single_batch(idx_batch):
        idx, batch = idx_batch
        # Convert the input DataFrame to CSV format (string)
        user_prompt = batch.to_csv(index=False)
        # Get the model's output
        model_output = process_unit_batches(instructions_prompt, user_prompt)
        # Save the raw model output to a file
        output_file = os.path.join(output_dir, f'model_output_{idx}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(model_output)
        return idx  # Return idx to identify which batch was processed

    # Function to process unit batches and save outputs in parallel
    def process_and_save_outputs_parallel(unit_batches, instructions_prompt):
        total_batches = len(unit_batches)
        start_time = time.time()

        # Use ThreadPoolExecutor to process batches in parallel
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {executor.submit(process_single_batch, (idx, batch)): idx for idx, batch in enumerate(unit_batches)}

            # Display progress in Streamlit
            progress_bar = st.progress(0)
            for i, future in enumerate(as_completed(futures)):
                idx = futures[future]
                try:
                    result_idx = future.result()
                    # Update progress
                    progress = (i + 1) / total_batches
                    progress_bar.progress(progress)
                except Exception as e:
                    st.error(f'An error occurred while processing batch {idx}: {e}')

        elapsed_time = time.time() - start_time
        st.write(f'All batches processed in {elapsed_time:.2f} seconds.')

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    def process_unit_batches(instructions_prompt, prompt):

        messages = [
            {"role": "system", "content": instructions_prompt},
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:radix:rent-rolls-onesitev02:AXNvm6Wi",
            messages=messages,
        )

        return response.choices[0].message.content
    # Process unit batches in parallel
    with st.spinner('Processing unit batches in parallel...'):
        process_and_save_outputs_parallel(unit_batches, instructions_prompt)

    # Combine saved outputs
    def combine_saved_outputs(output_dir='model_outputs_parallel'):
        # Initialize a list to hold parsed outputs
        parsed_outputs = []

        # Get all output files
        output_files = sorted(glob.glob(os.path.join(output_dir, 'model_output_*.json')))

        for output_file in output_files:
            with open(output_file, 'r', encoding='utf-8') as f:
                model_output = f.read()
                # Parse the model's output as JSON
                try:
                    output_json = json.loads(model_output)
                    parsed_outputs.append(output_json)
                except json.JSONDecodeError as e:
                    st.error(f"Error decoding JSON from {output_file}: {e}")

        # Initialize an empty dictionary to hold the combined data
        combined_data = {}

        for output in parsed_outputs:
            for unit, records in output.items():
                if unit not in combined_data:
                    combined_data[unit] = records
                else:
                    existing_records = combined_data[unit]
                    if isinstance(records, list):
                        for record in records:
                            if record not in existing_records:
                                existing_records.append(record)
                    else:
                        if records not in existing_records:
                            existing_records.append(records)

        return combined_data

    # Combine outputs and display results
    combined_data = combine_saved_outputs()

    # Convert combined data to DataFrame
    rows = []
    def flatten_data(unit, details):
        if isinstance(details, list):
            for item in details:
                rows.append({'Unit': unit, **item})
        elif isinstance(details, dict):
            rows.append({'Unit': unit, **details})
        else:
            rows.append({'Unit': unit, 'Details': details})

    for unit, details in combined_data.items():
        flatten_data(unit, details)

    if rows:
        llm_df = pd.DataFrame(rows)
        display_df_with_unique_cols(llm_df.head(), "LLM Output Data:")
        return llm_df
    else:
        st.error("No data was extracted by the LLM.")
        return None

def save_feedback(file_name, origin, template_type, file_type, status, comments, stage):
    feedback = {
        "File Name": file_name,
        "Origin": origin,
        "Template Type": template_type,
        "File Type": file_type,
        "Stage": stage,
        "Status": status,
        "Comments": comments
    }

    # Append feedback to a CSV file
    feedback_file = "feedback_log.csv"
    feedback_df = pd.DataFrame([feedback])

    if os.path.exists(feedback_file):
        feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        feedback_df.to_csv(feedback_file, index=False)

def display_df_with_unique_cols(df, message=""):
    """Helper function to display DataFrame with unique column names"""
    if message:
        st.write(message)

    # Create a copy for display purposes
    display_df = df.copy()

    # Create unique column names by adding suffix to duplicates
    seen = {}
    new_cols = []
    for col in display_df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}" if col != '' else f"Unnamed_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col if col != '' else "Unnamed")

    display_df.columns = new_cols
    st.dataframe(display_df)

if __name__ == "__main__":
    main()