# The utilities needed by `AI_medical_scribe.ipynb` are reported here to keep the
# notebook slim and readable

import yaml
from IPython.display import HTML, Markdown, display


def load_input(path):
    # open document
    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)

    # generate data
    data = "\n".join(
        [
            data["speaker"] + ": " + data["sentence"]
            for data in yaml_data["conversation"]
        ]
    )
    return data


def display_metrics_table(headers, row_data):
    """
    Display a flexible table in a Jupyter Notebook with an arbitrary number of columns
    and rows, allowing for custom headers and row content. This version correctly
    handles numeric and string comparisons for pass status, formats numeric values, and
    includes an explanation column if present, ensuring the explanation text is
    displayed.

    Parameters:
    - headers (list): List of column headers.
    - row_data (list of lists): Each sub-list contains the values for one row, matching
      the headers in order, and includes an optional explanation as the last element if
      present.
    """

    # Helper function to determine pass status
    def pass_status(score, threshold):
        try:
            score_float, threshold_float = float(score), float(threshold)
            return "✅" if score_float > threshold_float else "❌"
        except ValueError:
            # Handle string comparison
            return "✅" if score == threshold else "❌"

    # Helper function to format numeric values
    def format_value(value):
        try:
            return "{:.2f}".format(float(value))
        except ValueError:
            return value  # Handle non-numeric values gracefully

    # Construct table headers
    header_html = "".join(f"<th>{header}</th>" for header in headers)

    # Construct table rows
    rows_html = ""
    for row in row_data:
        # Check if an explanation is provided based on the row's length
        has_explanation = True if headers[-1] == "Explanation" else False

        # Calculate and append the "Passed" status
        metric, score, threshold = row[0], row[1], row[2]
        formatted_row = [
            metric,
            format_value(score),
            format_value(threshold),
            pass_status(score, threshold),
        ]

        # Append explanation if present
        if has_explanation:
            formatted_row.append(row[-1])  # Ensure the explanation is included

        # Generate HTML for the current row
        row_html = "".join(f"<td>{cell}</td>" for cell in formatted_row)
        rows_html += f"<tr>{row_html}</tr>"

    # Assemble the complete HTML table
    html_table = f"""
    <table>
    <tr>{header_html}</tr>
    {rows_html}
    </table>
    """
    display(HTML(html_table))


def load_yaml_input(path):
    """
    Loads a YAML file and returns its stringified representation.

    Args:
    - path (str): The file path to the YAML file.

    Returns:
    - str: A string representation of the YAML file content.
    """
    try:
        with open(path, "r") as file:
            # Load the YAML content
            data = yaml.safe_load(file)
        # Convert the Python data structure back into a YAML-formatted string
        # safe_dump ensures it's safely converted back into a string that represents YAML format
        yaml_string = yaml.safe_dump(data, sort_keys=False)
        return yaml_string
    except Exception as e:
        # Handle potential errors (e.g., file not found, YAML parsing errors)
        return f"Error loading or processing YAML file: {e}"


def text_compare(reference, output, missing_tokens):
    ref_css = reference
    THEME = "dark"  # can be "dark" or "light"
    COLOR = "#fcbdbd" if THEME == "light" else "#fc4949"
    open_tag = f'<span style="background-color: {COLOR}">'
    close_tag = "</span>"

    for target in missing_tokens:
        index_open = ref_css.find(target)
        index_close = index_open + len(target)
        ref_css = (
            ref_css[:index_open]
            + open_tag
            + ref_css[index_open:index_close]
            + close_tag
            + ref_css[index_close:]
        )

    ref_css = ref_css.replace("\n", "<br>")
    summary_css = output.replace("\n", "<br>")

    display(Markdown("**Full conversation:**"))
    display(Markdown(ref_css))
    display(Markdown("**Summary:**"))
    display(Markdown(summary_css))
