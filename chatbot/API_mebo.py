import requests
import json
import pandas as pd
import os
import random

def mebo(comment_user, user_id):
    url = "https://api-mebo.dev/api"
    headers = {"Content-Type":"application/json"}
    data ={"api_key":"d39a05a8-7075-4579-8ada-deac9a9eca7a184f55c64cb1d6", "agent_id":"646a0544-f966-49d7-a2a2-85aedb02ff0e184e1fa44561f", "utterance":comment_user, "uid":user_id}
    json_data = json.dumps(data)
    res = requests.post(url, data=json_data, headers=headers)

    #print(res) //Response[200]が返ると正常
    res_json = json.loads(res.text)
    #print(res_json)
    #print("mebo:{}" .format(res_json["bestResponse"]["utterance"]))
    return res_json

#mebo("ありがとう")

def textfile(path, text):
    file = open(path, 'a')
    file.write(text+'\n')
    file.close()

while(True):
    print("会話型ロボットです。こんにちは。")
    print("１：会話する、２：またにする")
    print("数字を選択してください：", end="")
    i = int(input())
    if(i == 1):
        id_num = random.randint(10000, 99999)
        user_id = "klab_botrobot_" + str(id_num)
        path = r'C:\Users\81703\OneDrive\ドキュメント\研究室\demo\mebo_' + user_id + '.txt'
        file = open(path, 'w')
        
        print("気軽に話しかけてください。")
        print("「さようなら」と入力すると会話が終了します。")
        print()
        j = 1
        while(True):
            #print(user_id)
            print("ユーザ：", end='')
            comment_user = input()

            if(j != 3):
                j = j + 1
                if(comment_user == "さようなら"):
                    print()
                    break
                else:
                    response = mebo(comment_user, user_id)
                    comment_bot = response["bestResponse"]["utterance"]
                    option = response["bestResponse"]["options"]
                    
                    textfile(path, comment_bot)
                    print("mebo:{}" .format(comment_bot))
                    print("返答の候補:", end="")
                    print(option)

            else:
                j = j + 1
                print("mebo:突然ですが、いくつかの質問に答えていただくとあなたに合った研究室を紹介できます。")
                print("mebo:「紹介してほしい」または「今はいい」で答えてください。")
                print("ユーザ：", end='')
                comment_user = input()
                response = mebo(comment_user, user_id)
                comment_bot = response["bestResponse"]["utterance"]
                option = response["bestResponse"]["options"]
                
                textfile(path, comment_bot)
                print("mebo:{}" .format(comment_bot))
                print("返答の候補:", end="")
                print(option)

            
    else:
        print("またね")
        break