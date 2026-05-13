from googletrans import Translator
 
text = "こんにちは。"

tr = Translator()
translated = tr.translate(text, dest="en").text
print(text)
print(translated)

print()