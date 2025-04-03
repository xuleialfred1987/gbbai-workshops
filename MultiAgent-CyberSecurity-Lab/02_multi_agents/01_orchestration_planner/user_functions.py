import os
import json
import sqlite3
import datetime
import markdown
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from xhtml2pdf import pisa
from datetime import datetime
from nixtla import NixtlaClient
from typing import Any, Callable, Set, List, Optional, Literal

# These are the user-defined functions that can be called by the agent.


def fetch_current_datetime(format: Optional[str] = None) -> str:
    """
    Get the current time as a JSON string, optionally formatted.

    :param format (Optional[str]): The format in which to return the current time. Defaults to None, which uses a standard format.

    :return: The current time in JSON format.

    :rtype: str
    """
    current_time = datetime.now()

    # Use the provided format if available, else use a default format
    if format:
        time_format = format
    else:
        time_format = "%Y-%m-%d %H:%M:%S"

    time_json = json.dumps(
        {"current_time": current_time.strftime(time_format)})
    return time_json


def fetch_data(table_name: Literal['intrusion_attempts', 'incident_detection_rate']) -> str:
    """
    Fetch data from a specified SQLite database table, save it as a CSV file, and return the status and file path in JSON format.

    :param table_name: The name of the table to fetch data from. Must be one of: 'intrusion_attempts' or 'incident_detection_rate'.
    :rtype: Literal['intrusion_attempts', 'incident_detection_rate']

    :return: A JSON string containing the status of the operation and the CSV file path (on success) or an error message (on failure).
    :rtype: str
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('./database/cyber_data.db')

        # Read all data from the intrusion_attempts table into a DataFrame
        intrusion_attempts = pd.read_sql_query(
            f"SELECT * FROM {table_name}", conn)

        # Save the DataFrame to a CSV file
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f'./data/{table_name}_{current_timestamp}.csv'
        intrusion_attempts.to_csv(csv_path, index=False)

        # Close the connection
        conn.close()

        return json.dumps({"success": True,
                           "message": csv_path})
    except Exception as e:
        return json.dumps({"success": False,
                           "message": f"An error occurred: {e}"})


def analyze_data(file_path: str,
                 method: Literal['forecast', 'anomaly_detection'],
                 horizon: Optional[int] = 0,
                 ftSteps: Optional[int] = 0) -> str:
    """
    Analyze a dataset using the specified analysis method (forecasting or anomaly detection).

    This function reads a pre-defined dataset based on the provided `file_name` argument, 
    performs the selected analysis method (`forecast` or `anomaly_detection`), and 
    returns the results in JSON format. If forecasting is selected, it generates 
    predictions over a specified horizon and optionally uses fine-tuning steps. 
    If anomaly detection is selected, it identifies anomalies in the data.

    :param file_name: The name of the dataset to analyze. 

    :param method: The analysis method to use. Must be one of ['forecast', 'anomaly_detection']:
                   - 'forecast': Predicts future values based on historical data.
                   - 'anomaly_detection': Detects anomalies in the data.

    :param horizon: (Optional) The forecast horizon, i.e., the number of future steps to predict.
                    Only applicable when `method='forecast'`. Default is 0.

    :param ftSteps: (Optional) The number of fine-tuning steps to apply during forecasting.
                    Only applicable when `method='forecast'`. Default is 0.

    :return: A string containing the path of the figure generated in the analysis. 
             If an error occurs, the returned string contains an error message.
    :rtype: str
    """
    if os.getenv("TIME_GEN_ENDPOINT") is None or os.getenv("TIME_GEN_KEY") is None:
        if method == 'forecast':
            return "./figures/Intrusion_Attempts_forecast_plot.png"
        else:
            return "./figures/Incident_Detection_Rate_anomalies_plot.png"
    
    nixtla_client = NixtlaClient(
        base_url=os.getenv("TIME_GEN_ENDPOINT"),
        api_key=os.getenv("TIME_GEN_KEY"),
    )

    if file_path is not None:
        file_path = file_path if file_path.endswith(
            'csv') else f'./data/{file_path}.csv'
        file_name = file_path.split('/')[-1].split('.')[0]
        df = pd.read_csv(file_path)

        # Initialize variables
        horizon_int = None
        ftSteps_int = None

        try:
            horizon_int = int(horizon)
        except (ValueError, TypeError) as e:
            return f"Error converting 'horizon': {e}"

        try:
            ftSteps_int = int(ftSteps)
        except (ValueError, TypeError) as e:
            return f"Error converting 'ftSteps': {e}"

        if ftSteps is None:
            ftSteps_int = 0

        try:
            if method == 'forecast':
                forecast_df = nixtla_client.forecast(
                    df=df,
                    h=horizon_int,
                    finetune_steps=ftSteps_int,
                    time_col="timestamp",
                    target_col="value",
                )

                fig = nixtla_client.plot(
                    df=df, forecasts_df=forecast_df, time_col="timestamp", target_col="value"
                )

                ax = fig.axes[0]
                ax.legend(["Actual Values", "Forecasted Values"],
                          loc="upper right", bbox_to_anchor=(1.14, 1), borderaxespad=0)
                ax.set_title(f"{file_name}: Forecasted vs Actual Values", fontsize=18,
                             fontweight="bold", color="teal", pad=15)

                fig_name = f"./figures/{file_name}_forecast_plot.png"
                fig.savefig(fig_name, dpi=300)

                return fig_name
            elif method == "anomaly_detection":
                anomalies_df = nixtla_client.detect_anomalies(
                    df,
                    time_col="ds",
                    target_col="y",
                    freq="D",
                )
                anomalies_df = anomalies_df.rename(columns={
                    "TimeGEN": "TimeGPT",
                    "TimeGEN-lo-99": "TimeGPT-lo-99",
                    "TimeGEN-hi-99": "TimeGPT-hi-99"
                })

                fig = nixtla_client.plot(
                    df, anomalies_df, time_col="ds", target_col="y")

                ax = fig.axes[0]
                ax.legend(["Actual Values", "TimeGPT", "TimeGPT_level_99", "TimeGPT_anomalies_level_99"],
                          loc="upper right", bbox_to_anchor=(1.21, 1), borderaxespad=0)
                ax.set_title(f"{file_name}: Anomalies on Actual Values", fontsize=18,
                             fontweight="bold", color="teal", pad=15)

                fig_name = f"./figures/{file_name}_anomalies_plot.png"
                fig.savefig(fig_name, dpi=300)

                return fig_name
            else:
                return "Invalid method specified."
        except Exception as e:
            return json.dumps({"error": str(e)})
    else:
        return "No data file provided."


def generate_radar_chart(categories: List[str], values: List[int]) -> str:
    """
    Generate a radar chart visualization for given categories and their corresponding values.

    This function creates a radar chart (also known as a spider chart) to visually represent the performance
    or metrics of different categories. It saves the generated chart as a PNG file in the local directory.

    :param categories: A list of category names to be displayed on the radar chart. Each category represents
                       a dimension on the radar chart.
                       Example: ['Vulnerability Score', 'Detection Rate', 'Response Time']

    :param values: A list of numerical values corresponding to the categories. These values should be on a
                   uniform scale (e.g., 0 to 10) for proper visualization.
                   Example: [8, 7, 6]

    :return: A tuple containing:
             - A status indicator (bool): True if the radar chart was successfully generated and saved,
               False if an error occurred.
             - A result message (str): If successful, the message contains the file path of the saved radar chart.
               If an error occurs, the message contains an error description.

             Example (Success):
             (True, './figures/cybersecurity_radar_chart.png')

             Example (Error):
             (False, 'An error occurred while generating the radar chart: FileNotFoundError')

    :rtype: str
    """
    try:
        # Number of categories
        num_vars = len(categories)

        # Compute the angle for each category
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]  # Close the radar chart loop

        # Close the loop for the data as well
        values += values[:1]

        # Create the figure
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, color='grey', size=10)

        # Draw y-labels (range of values)
        ax.set_rlabel_position(30)
        plt.yticks([2, 4, 6, 8], ["2", "4", "6", "8"], color="grey", size=7)
        plt.ylim(0, 10)

        # Plot data
        ax.plot(angles, values, linewidth=2,
                linestyle='solid', label="Current Metrics")

        # Fill area
        ax.fill(angles, values, color='blue', alpha=0.4)

        # Add a title
        plt.title('Cybersecurity Metrics Radar Chart',
                  size=15, color='black', y=1.1)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Define image file path
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f'./figures/cybersecurity_radar_chart_{current_timestamp}.png'

        # Save the figure locally
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Return success status and the image path
        return image_path

    except Exception as e:
        # Return failure status and an error message
        return f"An error occurred while generating the radar chart: {str(e)}"


def generate_report(markdown_content: str) -> str:
    """
    Generate a PDF report from Markdown content.

    This function takes a string containing Markdown-formatted content, converts it to HTML,
    and then generates a PDF report from the HTML. The generated PDF is saved in the './reports/' directory
    with a filename that includes a timestamp.

    :param markdown_content: A string containing the content in Markdown format.

    :return: A string indicating the path to the generated PDF file if successful, or an error message if
             the process fails.

    :rtype: str
    """
    try:
        html_content = markdown.markdown(markdown_content)
        with open("output.html", "w") as file:
            file.write(html_content)

        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"./reports/report_{current_timestamp}.pdf"
        with open(report_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)

        if pisa_status.err == 0:
            # Return success status and the image path
            return report_path
        else:
            # Return an error message
            return f"An error occurred while generating the radar chart: {pisa_status.err}"

    except Exception as e:
        # Return an error message
        return f"An error occurred while generating the radar chart: {str(e)}"


# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
    fetch_current_datetime,
    fetch_data,
    analyze_data,
    generate_radar_chart,
    generate_report
}
