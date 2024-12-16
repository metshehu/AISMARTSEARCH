import csv

import pandas as pd

csfile=pd.read_csv('static/uploads/01_BIA_Njoftimi me Lenden - Syllabusi.csv')
print(csfile)
print('-'*5)
print(csfile.vectors[0][:10])
