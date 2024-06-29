import requests
from bs4 import BeautifulSoup
import openai
from lib.security.openai_key import OpenaiKey



class CheckTradeByNews:
    def __init__(self):
        self.err = "Failed to retrieve content"


    # Function to fetch content from a URL
    def fetch_content(self,url):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text()
        else:
            return self.err
        
    def check_currency_pair_is_bad(self,curr,dir):
        try:
            dirStr = 'buying' if dir == 1 else ('selling' if dir == 2 else '')
            if dirStr == '':
                return False
            

            # Example URL
            url = "https://www.dailyfx.com/market-news"
            content = self.fetch_content(url)
            if content == self.err:
                return True
            
            # Define the initial prompt with fetched content
            initial_prompt = f"Here is the news content from the website dailyfx.com : {content} "
            

            # OpenAI API key
            openai.api_key = OpenaiKey['key']

            # Example questions
            questions = [
            
                "answer only with yes or no ,  depending on dailyfx.com news content provided , Is "+ dirStr +" "+ curr +" is bad ?"
                
                

            ]


            # Ask each question and print the response
            for question in questions:
                answer = self.ask_model(question,initial_prompt)
                if 'YES' in answer.upper():
                    return True
            

            return False
        
        except :
            return True


        

    # Function to ask questions to the model
    def ask_model(self,question,initial_prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                #{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": initial_prompt},
                {"role": "user", "content": question},
            ]
        )
        return response['choices'][0]['message']['content']



