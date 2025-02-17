from google import genai
from google.genai import types
import yfinance as yf
from dotenv import load_dotenv, dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from flask import Flask
from flask_socketio import SocketIO, emit
import os
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class Agent:
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        self.client = genai.Client(api_key=self.google_api_key)
        self.TTS_client = ElevenLabs(api_key=self.elevenlabs_api_key)
        self.model = "gemini-2.0-flash"


    ## function calling which extracts out Company name from question and tell whether question is for Stock Pice or not
    def get_company_name(self, company_name : str, type : str) -> (dict[str, str]):
        """Give a sentence input, you job is to first find out whether it asking for stock price or not.
        If not then set 'company_name' as 'NIL' and 'type' as 'NIL' and return else

        Here are some metches of compnay names and their symbols, just use symbols corresponding to a comoany name :
        1)Apple - 'AAPL'
        2) McDonald - 'MCD'
        3) Nike - 'NKE'
        4) Starbucks - 'SBUX'

        Args:
            company_name : symbol corresponding to name of the company if founded in the statement otherwise set it as 'NIL'.
            type : if input sentence asks for stock info then set it as 'get_stock_info' else set it as 'NIL'

        Returns:
            A dictionary containing the set company_name and type
        """
        return {
            "company_name": company_name,
            "type": type
        }
    
    
    def response_prompt_financeAPI(self, query, price):
        delimiter = "###"
    
        prompt = f"""{delimiter}
        You are a financial assistant. Your task is to generate a refined and informative response to the given user query.

        User Query: {query}

        The latest price of the relevant stock/asset is: {price}

        Instructions:
        - Ensure your response is short 1 liner , clear, concise, and professional.
        Provide the modified response below and on ending ask whether user have any further question or not?
        {delimiter}
        """

        return prompt
    

    def response_prompt_general(self, query):
        delimeter = "###"

        prompt = f"""{delimeter}
        You are an AI assistant who responds to the user queries in short 1 or 2 lines. And ask whether you can help user with any queries related to stock market.
        The query is {query}
        {delimeter}"""

        return prompt
    
    
    def get_stock(self, stock_symbol):
        stock_data = yf.Ticker(stock_symbol)
        return stock_data.info

    
    def get_stock_details(self, company_symbol):
        company_info = self.get_stock(company_symbol)
        stock_price = company_info['currentPrice']
        ## print(stock_price)
        return stock_price


    def generate_response(self, query):
        config = types.GenerateContentConfig(tools=[self.get_company_name],
                                     automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        ## response 
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            config=config,
            contents=query
        )

        response_part1 = response.candidates[0].content.parts[0].function_call
        response_part2 = response.candidates[0].content.parts[0]
        
        if hasattr(response_part1, "args"):  # Check if function call exists
            return response_part1.args
        else:
            return {
                "company_name": "NIL",
                "type": "NIL"
            }
            

    

    def general_LLM_response(self, prompt):
        response = self.client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return (response.text)
    

    def pipeline(self, query):
        arguements = self.generate_response(query) 
        ## deciding whether arguements favours API call
        if arguements['type'] == 'get_stock_info':
            price = self.get_stock_details(arguements['company_name'])
            prompt = self.response_prompt_financeAPI(query, price)
            response = self.general_LLM_response(prompt)
            return response
        else:
            ## if Question asked not favors API then use general LLM
            prompt = self.response_prompt_general(query)
            response = self.general_LLM_response(prompt)
            return response
        
    def TTS(self, text):
        audio = self.TTS_client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        play(audio)
        
'''

### if __name__ == "__main__":
    agent = Agent()
    ## ask_question = input("Do you want to ask question?  --> ")
    agent.TTS("Hii Umesh. How Can I help you today?")
    while True:
        query = input("Query -> ")
        print(f"Query : {query}")
        if query.lower() == 'no':
            break
        res = agent.pipeline(query)
        print(f"Response : {res}")
        ## TTS agent 
        agent.TTS(res)
        print(f"--------------------------------------------------------------------------------------------")

    print("Thank You!!! :)") 

'''

agent = Agent()

@socketio.on("message")
def handle_message(query):
    print(f"Received query: {query}")
    response = agent.pipeline(query)
    emit("response", response)  # Send response back to client
    agent.TTS(response)  # Play response via ElevenLabs TTS

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=6000, debug=True, use_reloader=False)