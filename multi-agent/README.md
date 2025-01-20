# Azure AI Workshop

Welcome to the **Azure AI Workshop** repository! This workshop is designed to help participants explore and gain hands-on experience with Azure AI tools and agents. Whether you are a beginner or an expert in AI, this workshop will guide you through the exciting world of Azure AI, from foundational concepts to advanced multi-agent systems.

---

## Workshop Overview

The workshop is divided into two main parts:

### **Part 1: Introduce Azure AI Foundry**
In this section, participants will be introduced to the Azure AI Foundry. This will include an overview of the tools, capabilities, and potential use cases for building AI-powered applications using Azure.

### **Part 2: Hands-on Labs**
This is the practical part of the workshop, where participants will engage in hands-on activities to build and interact with AI agents. The labs are divided into two categories:

#### **1. Single Agent Labs**
- **Code Agent**: Learn how to create and customize an AI agent that performs coding tasks.
- **Search Agent**: Build an AI agent that can perform intelligent search and retrieval.
- **Writer Agent**: Create an agent that generates content, such as articles or documentation.

#### **2. Multi-Agent Labs**
- **Manual Planner**: Design a manual task planner to coordinate the single agents described above.
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
    git clone https://github.com/xxx/azure-ai-workshop.git
    cd azure-ai-workshop

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
│   ├── code-agent/   
│   ├── search-agent/   
│   ├── writer-agent/    
│   └── manual-planner/                               
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