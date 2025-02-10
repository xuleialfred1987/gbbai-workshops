import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def get_conversation_md(conversation):
    """
    Function to return a conversation in Markdown (MD) format as a string.
    """
    messages = conversation.get("data", [])
    if not messages:
        return "No messages found in the conversation."

    # Initialize a list to hold the Markdown lines
    md_lines = []
    md_lines.append("# Conversation")
    md_lines.append("___")  # Markdown horizontal line

    # Iterate through the messages
    # Reversing to maintain chronological order
    for message in reversed(messages):
        role = message.get("role", "unknown").capitalize()
        timestamp = message.get("created_at")
        content = message.get("content", [])

        # Convert timestamp to a readable format
        if timestamp:
            timestamp = datetime.fromtimestamp(
                timestamp).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        else:
            timestamp = "Unknown time"

        # Extract the text content
        message_text = ""
        for item in content:
            if item.get("type") == "text":
                message_text += item["text"].get("value", "")

        # Append the message in Markdown format
        md_lines.append(f"### **{role}** ({timestamp})")
        md_lines.append(f"{message_text}")
        md_lines.append("___")  # Markdown horizontal line

    # Join the lines with newlines to form the complete Markdown string
    return "\n".join(md_lines)


def import_csv_to_sqlite(csv_path, table_name, db_name='./database/cyber_data.db'):
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

    print(f"Data imported successfully to {db_name}")


def read_and_plot_csv(file_path: str):
    """
    Read data from a CSV file and plot it.

    :param file_path: The path to the CSV file.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamp'], df['value'], marker='o', linestyle='-')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Data from CSV File')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
