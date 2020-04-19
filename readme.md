###  Introduction
This code and the explenation behind can be found at the following youtube link.
https://www.youtube.com/watch?v=lUI6VMj43PE

All I've done is reupload it to github as the link in the video description is no longer available.

### Dataset
http://www.cis.fordham.edu/wisdm/dataset.php 

### Tree structure
```
/
├── data/
│   ├── readme.txt
│   ├── WISDM_ar_v1.1_raw.txt
│   ├── WISDM_ar_v1.1_raw_about.txt
│   ├── WISDM_ar_v1.1_trans_about.txt
│   └── WISDM_ar_v1.1_transformed.arff
└── HAR classifier with TensorFlow 2.0.ipynb
```


```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)
```

    2.1.0
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

```


```python
processed_list = []

with open('data/WISDM_ar_v1.1_raw.txt') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        try:
            line = line.split(",")
            last = line[5].split(";")[0]
            last = last.strip()
            if last == '':
                break
            temp = line[:5] + [last]
            processed_list.append(temp)
        except Exception as e:
            print(e)
            print("error at line number:", i, line)
```

    list index out of range
    error at line number: 281873 ['\n']
    list index out of range
    error at line number: 281874 ['\n']
    list index out of range
    error at line number: 281875 ['\n']
    


```python
columns = ['user', 'activity', 'time', 'x', 'y', 'z']
```


```python
data = pd.DataFrame(data=processed_list, columns=columns)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>activity</th>
      <th>time</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>Jogging</td>
      <td>49105962326000</td>
      <td>-0.6946377</td>
      <td>12.680544</td>
      <td>0.50395286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>Jogging</td>
      <td>49106062271000</td>
      <td>5.012288</td>
      <td>11.264028</td>
      <td>0.95342433</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>Jogging</td>
      <td>49106112167000</td>
      <td>4.903325</td>
      <td>10.882658</td>
      <td>-0.08172209</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>Jogging</td>
      <td>49106222305000</td>
      <td>-0.61291564</td>
      <td>18.496431</td>
      <td>3.0237172</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>Jogging</td>
      <td>49106332290000</td>
      <td>-1.1849703</td>
      <td>12.108489</td>
      <td>7.205164</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (343416, 6)




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 343416 entries, 0 to 343415
    Data columns (total 6 columns):
     #   Column    Non-Null Count   Dtype 
    ---  ------    --------------   ----- 
     0   user      343416 non-null  object
     1   activity  343416 non-null  object
     2   time      343416 non-null  object
     3   x         343416 non-null  object
     4   y         343416 non-null  object
     5   z         343416 non-null  object
    dtypes: object(6)
    memory usage: 15.7+ MB
    


```python
data.isnull().sum()
```




    user        0
    activity    0
    time        0
    x           0
    y           0
    z           0
    dtype: int64




```python
data['activity'].value_counts()
```




    Walking       137375
    Jogging       129392
    Upstairs       35137
    Downstairs     33358
    Sitting         4599
    Standing        3555
    Name: activity, dtype: int64




```python
data['x'] = data['x'].astype('float')
data['y'] = data['y'].astype('float')
data['z'] = data['z'].astype('float')
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 343416 entries, 0 to 343415
    Data columns (total 6 columns):
     #   Column    Non-Null Count   Dtype  
    ---  ------    --------------   -----  
     0   user      343416 non-null  object 
     1   activity  343416 non-null  object 
     2   time      343416 non-null  object 
     3   x         343416 non-null  float64
     4   y         343416 non-null  float64
     5   z         343416 non-null  float64
    dtypes: float64(3), object(3)
    memory usage: 15.7+ MB
    


```python
fs = 20
activities = data['activity'].value_counts().index
```


```python
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15,7), sharex=True)
    plot_axis(ax0, data['time'], data['x'], 'X-axis')
    plot_axis(ax1, data['time'], data['y'], 'Y-axis')
    plot_axis(ax2, data['time'], data['z'], 'Z-axis')
    plt.subplots_adjust(hspace=0.2)
    plt.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim(min(y) - np.std(y), max(y), + np.std(y))
    ax.set_xlim(min(x), max(x))
    ax.grid(True)

for activity in activities:
    data_for_plot = data[(data['activity'] == activity)][:fs * 10]
    plot_activity(activity, data_for_plot)
```


![png](HAR%20classifier%20with%20TensorFlow%202.0_files/HAR%20classifier%20with%20TensorFlow%202.0_13_0.png)



![png](HAR%20classifier%20with%20TensorFlow%202.0_files/HAR%20classifier%20with%20TensorFlow%202.0_13_1.png)



![png](HAR%20classifier%20with%20TensorFlow%202.0_files/HAR%20classifier%20with%20TensorFlow%202.0_13_2.png)



![png](HAR%20classifier%20with%20TensorFlow%202.0_files/HAR%20classifier%20with%20TensorFlow%202.0_13_3.png)



![png](HAR%20classifier%20with%20TensorFlow%202.0_files/HAR%20classifier%20with%20TensorFlow%202.0_13_4.png)



![png](HAR%20classifier%20with%20TensorFlow%202.0_files/HAR%20classifier%20with%20TensorFlow%202.0_13_5.png)



```python
df = data.drop(['user', 'time'], axis=1).copy()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jogging</td>
      <td>-0.694638</td>
      <td>12.680544</td>
      <td>0.503953</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jogging</td>
      <td>5.012288</td>
      <td>11.264028</td>
      <td>0.953424</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jogging</td>
      <td>4.903325</td>
      <td>10.882658</td>
      <td>-0.081722</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jogging</td>
      <td>-0.612916</td>
      <td>18.496431</td>
      <td>3.023717</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jogging</td>
      <td>-1.184970</td>
      <td>12.108489</td>
      <td>7.205164</td>
    </tr>
  </tbody>
</table>
</div>




```python
min_activity = df['activity'].value_counts().min()
df['activity'].value_counts()
```




    Walking       137375
    Jogging       129392
    Upstairs       35137
    Downstairs     33358
    Sitting         4599
    Standing        3555
    Name: activity, dtype: int64




```python
walking = df[df['activity'] == 'Walking'].head(min_activity).copy()
jogging = df[df['activity'] == 'Jogging'].head(min_activity).copy()
upstairs = df[df['activity'] == 'Upstairs'].head(min_activity).copy()
downstairs = df[df['activity'] == 'Downstairs'].head(min_activity).copy()
sitting = df[df['activity'] == 'Sitting'].head(min_activity).copy()
standing = df[df['activity'] == 'Standing'].head(min_activity).copy()
```


```python
balanced_df = pd.DataFrame()
balanced_df = balanced_df.append([walking, jogging, upstairs, downstairs, sitting, standing])
balanced_df.shape
```




    (21330, 4)




```python
balanced_df['activity'].value_counts()
```




    Standing      3555
    Jogging       3555
    Sitting       3555
    Upstairs      3555
    Downstairs    3555
    Walking       3555
    Name: activity, dtype: int64




```python
balanced_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>597</th>
      <td>Walking</td>
      <td>0.844462</td>
      <td>8.008764</td>
      <td>2.792171</td>
    </tr>
    <tr>
      <th>598</th>
      <td>Walking</td>
      <td>1.116869</td>
      <td>8.621680</td>
      <td>3.786457</td>
    </tr>
    <tr>
      <th>599</th>
      <td>Walking</td>
      <td>-0.503953</td>
      <td>16.657684</td>
      <td>1.307553</td>
    </tr>
    <tr>
      <th>600</th>
      <td>Walking</td>
      <td>4.794363</td>
      <td>10.760075</td>
      <td>-1.184970</td>
    </tr>
    <tr>
      <th>601</th>
      <td>Walking</td>
      <td>-0.040861</td>
      <td>9.234595</td>
      <td>-0.694638</td>
    </tr>
  </tbody>
</table>
</div>




```python
label = LabelEncoder()
balanced_df['label'] = label.fit_transform(balanced_df['activity'])

```


```python
print(label.classes_)
balanced_df.head()
```

    ['Downstairs' 'Jogging' 'Sitting' 'Standing' 'Upstairs' 'Walking']
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>597</th>
      <td>Walking</td>
      <td>0.844462</td>
      <td>8.008764</td>
      <td>2.792171</td>
      <td>5</td>
    </tr>
    <tr>
      <th>598</th>
      <td>Walking</td>
      <td>1.116869</td>
      <td>8.621680</td>
      <td>3.786457</td>
      <td>5</td>
    </tr>
    <tr>
      <th>599</th>
      <td>Walking</td>
      <td>-0.503953</td>
      <td>16.657684</td>
      <td>1.307553</td>
      <td>5</td>
    </tr>
    <tr>
      <th>600</th>
      <td>Walking</td>
      <td>4.794363</td>
      <td>10.760075</td>
      <td>-1.184970</td>
      <td>5</td>
    </tr>
    <tr>
      <th>601</th>
      <td>Walking</td>
      <td>-0.040861</td>
      <td>9.234595</td>
      <td>-0.694638</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



### Standardized data


```python
x = balanced_df[['x', 'y', 'z']]
y = balanced_df['label']
```


```python
scaler = StandardScaler()
x = scaler.fit_transform(x)

scaled_x = pd.DataFrame(data=x, columns=['x', 'y', 'z'])
scaled_x['label'] = y.values

scaled_x
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000503</td>
      <td>-0.099190</td>
      <td>0.337933</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.073590</td>
      <td>0.020386</td>
      <td>0.633446</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.361275</td>
      <td>1.588160</td>
      <td>-0.103312</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.060258</td>
      <td>0.437573</td>
      <td>-0.844119</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.237028</td>
      <td>0.139962</td>
      <td>-0.698386</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21325</th>
      <td>-0.470217</td>
      <td>0.178084</td>
      <td>0.261019</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21326</th>
      <td>-0.542658</td>
      <td>0.193692</td>
      <td>0.248875</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21327</th>
      <td>-0.628514</td>
      <td>0.197593</td>
      <td>0.261019</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21328</th>
      <td>-0.781444</td>
      <td>0.049322</td>
      <td>0.155768</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21329</th>
      <td>-0.800225</td>
      <td>0.267827</td>
      <td>0.475569</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>21330 rows × 4 columns</p>
</div>



### Frame preparation


```python
import scipy.stats as stats
```


```python
frame_size = fs * 4 # one frame is constituted of 4 seconds of datah
hop_size = fs * 2
```


```python
def get_frames(df, frame_size, hop_size):
    N_FEATURES = 3
    frames = []
    labels = []
    
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i:i+frame_size]
        y = df['y'].values[i:i+frame_size]
        z = df['z'].values[i:i+frame_size]
        
        # retrieve the most often used label in this segment
        label = stats.mode(df['label'][i:i+frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)
    
    # bring the segment into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)
    
    return frames, labels

```


```python
x, y = get_frames(scaled_x, frame_size, hop_size)
```


```python
x.shape, y.shape
```




    ((532, 80, 3), (532,))




```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
```


```python
x_train.shape, x_test.shape 
```




    ((425, 80, 3), (107, 80, 3))




```python
x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)

```


```python
x_train[0].shape, x_test[0].shape 
```




    ((80, 3, 1), (80, 3, 1))



### 2D CNN Model


```python
model = Sequential()
model.add(Conv2D(16, (2, 2), activation='relu', input_shape=x_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))
```


```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```


```python
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)
```

    Train on 425 samples, validate on 107 samples
    Epoch 1/10
    425/425 [==============================] - 0s 812us/sample - loss: 1.6733 - accuracy: 0.2000 - val_loss: 1.4258 - val_accuracy: 0.3832
    Epoch 2/10
    425/425 [==============================] - 0s 198us/sample - loss: 1.3383 - accuracy: 0.4612 - val_loss: 1.1003 - val_accuracy: 0.7664
    Epoch 3/10
    425/425 [==============================] - 0s 176us/sample - loss: 1.0803 - accuracy: 0.5953 - val_loss: 0.7904 - val_accuracy: 0.8598
    Epoch 4/10
    425/425 [==============================] - 0s 165us/sample - loss: 0.7919 - accuracy: 0.7318 - val_loss: 0.5458 - val_accuracy: 0.8692
    Epoch 5/10
    425/425 [==============================] - 0s 164us/sample - loss: 0.5684 - accuracy: 0.8024 - val_loss: 0.3716 - val_accuracy: 0.8879
    Epoch 6/10
    425/425 [==============================] - 0s 161us/sample - loss: 0.4384 - accuracy: 0.8588 - val_loss: 0.3508 - val_accuracy: 0.8598
    Epoch 7/10
    425/425 [==============================] - 0s 160us/sample - loss: 0.3753 - accuracy: 0.8612 - val_loss: 0.2985 - val_accuracy: 0.8692
    Epoch 8/10
    425/425 [==============================] - 0s 162us/sample - loss: 0.2819 - accuracy: 0.9106 - val_loss: 0.2684 - val_accuracy: 0.8692
    Epoch 9/10
    425/425 [==============================] - 0s 158us/sample - loss: 0.2309 - accuracy: 0.9388 - val_loss: 0.2782 - val_accuracy: 0.8785
    Epoch 10/10
    425/425 [==============================] - 0s 165us/sample - loss: 0.2326 - accuracy: 0.9271 - val_loss: 0.2474 - val_accuracy: 0.8785
    


```python
def plot_learning_curve(history, epochs):
    # Plot training and valiation accuracy values    
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title("Model accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
    # Plot training and validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title("Model loss")
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
```


```python
plot_learning_curve(history, 10)
```


![png](HAR%20classifier%20with%20TensorFlow%202.0_files/HAR%20classifier%20with%20TensorFlow%202.0_40_0.png)



![png](HAR%20classifier%20with%20TensorFlow%202.0_files/HAR%20classifier%20with%20TensorFlow%202.0_40_1.png)


### Confusion Matrix


```python
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
```


```python
y_pred = model.predict_classes(x_test)
```


```python
mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7));
```


![png](HAR%20classifier%20with%20TensorFlow%202.0_files/HAR%20classifier%20with%20TensorFlow%202.0_44_0.png)



```python
model.save_weights("model.h5")
```


```python

```
