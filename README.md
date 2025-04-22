## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## Developed by :YUVAN SUNDAR S
## Reg No: 212223040250
# CODING AND OUTPUT:
 ```py
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/91212134-9972-4149-9062-b944eee764ee)
```py
 from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
 pm=['Hot','Warm','Cold']
 e1=OrdinalEncoder(categories=[pm])
 e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/60e33830-cde0-4416-ae6c-3a62618defbb)
```py
 df['bo2']=e1.fit_transform(df[["ord_2"]])
 df
```
![image](https://github.com/user-attachments/assets/a82844f2-5a0d-437b-8089-061ad70602b4)
```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/36bc5e52-2b97-423f-a615-7e9c6a5ef58c)
```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/cb539601-dbf0-47db-b1ad-e4efc1cd225e)
```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/a965e66e-56f9-4d92-af4c-767dae4d5399)
```py
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/0b462857-5a11-4132-9253-4f94fb333463)
```py
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/a3a30a7d-e80b-477b-af0f-4dd439ecbb4c)
```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/7a3b8b79-83c0-497a-8882-6700d6908f28)
```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/797b5ec0-ba30-4698-9826-e772eb179a16)
```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/fd37f995-0502-45d0-9377-bd6d3ae635bd)
```py
df.skew()
```
![image](https://github.com/user-attachments/assets/614b5b60-3554-4dd2-bd0d-6c0bca829eb8)
```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f7b8f8e4-d6a1-48c0-bd7d-85b3c8a5a868)
```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/427ba7aa-1ad6-400b-b08a-5c740376c1d0)
```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/50f14ba4-fabb-4a84-81ae-90126f8a587f)
```py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/c022482c-8439-49a8-b57e-4a971549ed88)
```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/569912d7-be90-4751-8f6f-38c31dad085a)
```py
 df.skew()
```
![image](https://github.com/user-attachments/assets/5757fffc-811a-4c0d-9293-e55262473793)
```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/874a80d9-f9dd-4ebc-b34c-b6ba91cd4a19)
```py
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```
![image](https://github.com/user-attachments/assets/a6c186b0-3757-4e29-bca8-ea78a55492b0)
```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ea7a9948-a074-49c3-971c-48a6f97ed0a3)
```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/301b915d-7c65-474e-ad78-851c3fea1414)
```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/29e641b9-04e5-4dda-a03b-7b2044a68048)
```py
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/38c6f188-04d8-4d0b-a1b6-215710a519ba)
```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```
![image](https://github.com/user-attachments/assets/47722f10-8fb7-4a9e-940e-f9573581abaf)
```py
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 dt["Age_1"]=qt.fit_transform(dt[["Age"]])
 sm.qqplot(dt['Age'],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/9dadaeb2-f9b4-4956-9b99-a5a020227878)
```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/70a49adf-99b0-4edc-ba19-2ed5fb58f26e)
# RESULT:
  Thus the given data, Feature Encoding, Transformation process and save the data to a file
  was performed successfully.

       
