import openai
import json

from legacy_fucntions import legacy_fucntions
import legacy

class LegacyEngine:
    def __init__(self, system_name) -> None:
        self.functions = legacy_fucntions[system_name]
        self.legacy = self.__get_legacy(system_name) 

    def command(self, command):
        return self.__rorry_bot(command)

    # 실제로는 Legacy시스템에 원격 접속
    def __get_legacy(self, system_name) -> legacy.Legacy:
        return legacy.legacy_map[system_name]

    def __rorry_bot(self, command, temperature=0, max_tokens=1024):

        client = openai.OpenAI()

        messages = [
            {
                "role": "user",
                "content": command
            }
        ]

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=self.functions,
            function_call="auto",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        result = completion.choices[0].message
        #print(result)
        # function_call 존재하는 경우 function_call.name과 동일한 이름의 Legacy method 실행
        if result.function_call:
            # 가끔 function 이름 앞에 'functions/' 문자열이 붙는 경우 remove
            function_name = result.function_call.name.replace('functions/','')
            arguments = json.loads(result.function_call.arguments)
            response = getattr(self.legacy, function_name)(**arguments)
            #response = f"Legacy 함수 호출 : {function_name}, {arguments}"
            print(f"Legacy 함수 호출 : {function_name}, {arguments}")
            rtn = 1
        else:
            response = result
            rtn = -1
            
        return rtn, str(response)