# MultiAgent CyberSecurity Lab

Welcome to the **Multi-Agent Workshop** repository! This workshop is designed to help participants explore and gain hands-on experience with Azure AI tools and agents. Whether you are a beginner or an expert in AI, this workshop will guide you through the exciting world of Azure AI, from foundational concepts to advanced multi-agent systems.

---

## Workshop Overview

The workshop is divided into two main parts:

### **Part 1: Introduce Azure AI Foundry**
In this section, participants will be introduced to the Azure AI Foundry. This will include an overview of the tools, capabilities, and potential use cases for building AI-powered applications using Azure.

### **Part 2: Hands-on Labs**
This is the practical part of the workshop, where participants will engage in hands-on activities to build and interact with AI agents. The labs are divided into two categories:

#### **1. Single Agent Labs**
- **Time keeper**: An AI agent specialized in providing accurate and real-time time information.
- **Cyber collector**: The cyber collector can collect information about cyber security from its knowledge base.
- **DB reader**: This agent fetches data from a specified SQLite database table, saves it as a CSV file, and returns the file path.
- **Data analyzer**: An agent that analyzes data using forecasting or anomaly detection.
- **Security evaluator**: An AI agent that assists users in generating radar charts for cybersecurity metrics.
- **Report writer**: An AI agent responsible for integrating information and analysis results from various agents to produce a professional and detailed report.

#### **2. Multi-Agent Labs**
- **Manual Planner**: Design a manual task planner to coordinate the single agents described above.
  - The AI Planner coordinates the report generation workflow by managing agent interactions and ensuring task completion.
  - The AI Planner will use the outputs of each agent to make decisions about how to proceed with the report generation process.
  - The AI Planner adheres to user guidelines while managing the timeline and coordination of the report generation process.
- **Auto Gen**: Explore the use of multiple agents working together automatically to solve complex tasks. These include:
  - **Models**: Understand how models power agents.
  - **Agents**: Learn how multiple agents interact.
  - **Teams**: Configure teams of agents for collaborative tasks.
  - **Selector Group Chat**: Enable group chats between agents for task selection and planning.
  - **Swarms**: Dive into the concept of agent "swarms" for distributed AI problem-solving.

---

## Prerequisites

Before starting the workshop, ensure you have the following:
- An Azure account with access to Azure AI Founrdry and Azure OpenAI services.
- Basic understanding of AI concepts and Python programming.
- Installed dependencies (details provided in the [Setup](#setup) section).

---

## Setup

1. Clone or download this repository:
    ```bash
    git clone https://github.com/xuleialfred1987/gbbai-workshops.git
    cd multi-agent

2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt

3. Configure your Azure credentials:
    - Set up the necessary environment variables or configuration files for Azure authentication.

4. Start exploring the labs

## Folder Structure

```plaintext
azure-ai-workshop/
│
├── single_agent/                    # Single Agent labs 
│   ├── 00_start.ipynb   
│   ├── 01_time_keeper.ipynb   
│   ├── 02_cyber_collector.ipynb   
│   ├── 03_db_reader.ipynb   
│   ├── 04_data_analyzer.ipynb   
│   ├── 05_security_evaluator.ipynb    
│   └── 06_report_writer.ipynb                              
├── multi_agents/                    # Multi-Agent labs
│   ├── manaul_planner
│   ├── auto_gen/                
│       ├── models/
│       ├── agents/
│       ├── teams/
│       ├── selector-group-chat/
│       └── swarms/
├── requirements.txt                # Python dependencies
└── README.md                       # Workshop documentation
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

We hope you enjoy this workshop and gain valuable insights into Azure AI. Happy learning!
