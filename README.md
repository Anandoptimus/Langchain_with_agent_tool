# LangChain Agent with Tools for Weather and Wikipedia Search

This project demonstrates the use of **LangChain** in creating an agent capable of interacting with external APIs to:

1. Fetch current weather data based on location coordinates (latitude and longitude).
2. Search Wikipedia for information based on a query.

The project leverages **LangChain**, **OpenAI**, and external APIs to build a versatile agent.

---

## 🧰 Features

- **Weather Fetching**: Retrieve current temperature for any location by providing latitude and longitude.
- **Wikipedia Search**: Perform a Wikipedia search and get summaries of relevant pages.
- **OpenAI Integration**: Utilizes OpenAI's GPT-3.5-turbo for natural language processing and function calling.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/langchain-weather-wikipedia-agent.git
cd langchain-weather-wikipedia-agent
```

### 2. Install Required Dependencies
- pip install -r requirements.txt

### 3. Set Up Environment Variables
- OPENAI_API_KEY="your-openai-api-key"

## 💡 Example Inputs
- You can test various inputs like:
- Weather: "What is the weather in vellore?"
- Wikipedia: "What is LangChain?"

📚 Notes
- No Assumptions: The agent fetches only the data that is explicitly available.
- Error Handling: If no results are found, the agent responds accordingly.

🚀 Contribution
- Feel free to open Issues or contribute via Pull Requests to improve the agent's capabilities, extend features, or optimize the code.
