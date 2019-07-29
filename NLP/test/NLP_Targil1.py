import pandas as pd
import numpy as np


file_path = r"D:\DS\NLP\NLP_Targil1\targil1.csv"
text = pd.read_csv(file_path,header = None)

text = text.loc[0].values[0]



from translate import Translator
translator= Translator(from_lang="german",to_lang="spanish")
translation = translator.translate("Guten Morgen")
print(translation)













