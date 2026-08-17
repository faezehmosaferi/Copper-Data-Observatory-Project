import pandas as pd 

def build_tuples():
     df = pd.read_excel(
    "copper-lme-stock.xlsx",
    sheet_name="Sheet1"
)

     df.columns = ['date', 'lme-copper-stock']
     result = tuple(
              zip(df["date"], ['LME-COPPER-STOCK']*len(df), df['lme-copper-stock'], ['Ton']*len(df),
                  ['westmetal.com']*len(df))
                 )
     return result

result = build_tuples()
print(type(result))

