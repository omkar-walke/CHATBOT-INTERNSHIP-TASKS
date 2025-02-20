import pandas as pd
import json
import datetime

class AnalyticsTracker:
    def __init__(self, log_file="chat_logs.json"):
        self.log_file = log_file
        self.df = self.load_logs()

    def load_logs(self):
        try:
            with open(self.log_file, "r") as file:
                data = [json.loads(line) for line in file]
            return pd.DataFrame(data)
        except FileNotFoundError:
            return pd.DataFrame(columns=["timestamp", "user_id", "query", "intent", "response"])

    def log_interaction(self, user_id, query, intent, response):
        new_entry = {
            "timestamp": str(datetime.datetime.now()),
            "user_id": user_id,
            "query": query,
            "intent": intent,
            "response": response
        }
        
        # ✅ FIX: Use pd.concat instead of append
        self.df = pd.concat([self.df, pd.DataFrame([new_entry])], ignore_index=True)

        # Append to JSON file
        with open(self.log_file, "a") as logfile:
            logfile.write(json.dumps(new_entry) + "\n")

        print(f"✅ Interaction Logged: {new_entry}")  # Debugging message
