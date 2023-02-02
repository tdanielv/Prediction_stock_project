import time

import requests
import json
import pprint


BOT_TOKEN  = '5321476850:AAGgKK501MtxMe9tUVLtEw2VxS0LKiZlZ2g'
METHOD_NAME = 'sendMessage'
CHAT_ID = '580379590'
TEXT = 'Hello'
API_URL = f'https://api.telegram.org/bot'
MAX_COUNTER = 100
OFFSET = -2
COUNTER = 0
response = requests.get(API_URL)

while COUNTER < MAX_COUNTER:
    print('attempt = ', COUNTER)

    updates = requests.get(f'{API_URL}{BOT_TOKEN}/getUpdates?offset={OFFSET+1}').json()
    if updates['result']:
        for result in updates['result']:
            OFFSET = result['update_id']
            chat_id = result['message']['from']['id']
            name =  result['message']['from']['first_name']
            text = f'Hello, {name}'
            requests.get(f'{API_URL}{BOT_TOKEN}/sendMessage?chat_id={chat_id}&text={text}')

    time.sleep(1)
    COUNTER += 1
