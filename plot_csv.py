# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
df = pd.read_csv("loss_log.csv")
# Preview the first 5 lines of the loaded data 
print(df.head())

import matplotlib.pyplot as plt


plt.close()
# gca stands for 'get current axis'
ax = plt.gca()

df.plot(kind='line',x='epoch',y='training_loss',ax=ax)
df.plot(kind='line',x='epoch',y='val_loss', color='red', ax=ax)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training_loss', 'val_loss'], loc='upper right')
plt.savefig("losses.png", dpi=400)
plt.show()
plt.close()

