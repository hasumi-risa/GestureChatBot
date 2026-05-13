import os
import pprint
import json
import corenlp

# パーサの生成
base_dir = './src/NLP/'
corenlp_dir = base_dir + "stanford-corenlp-full-2013-06-20/"
properties_file = base_dir + "user.properties"
parser = corenlp.StanfordCoreNLP(
    corenlp_path=corenlp_dir,
    properties=properties_file) # propertiesを設定

# パースして結果をpretty print
result_json = json.loads(parser.parse("I am Alice."))
pprint.pprint(result_json)