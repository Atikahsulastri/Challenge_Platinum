{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From g:\\Pribadi\\Andi\\Challenge_Platinum\\challenge_platinum\\Lib\\site-packages\\keras\\src\\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import requests\n",
    "from io import StringIO\n",
    "import tensorflow\n",
    "import keras\n",
    "import sklearn\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Text</th>\n",
       "      <th>Kategori</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>warung ini dimiliki oleh pengusaha pabrik tahu...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>mohon ulama lurus dan k212 mmbri hujjah partai...</td>\n",
       "      <td>neutral</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>lokasi strategis di jalan sumatera bandung . t...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>betapa bahagia nya diri ini saat unboxing pake...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>duh . jadi mahasiswa jangan sombong dong . kas...</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                Text  Kategori\n",
       "0  warung ini dimiliki oleh pengusaha pabrik tahu...  positive\n",
       "1  mohon ulama lurus dan k212 mmbri hujjah partai...   neutral\n",
       "2  lokasi strategis di jalan sumatera bandung . t...  positive\n",
       "3  betapa bahagia nya diri ini saat unboxing pake...  positive\n",
       "4  duh . jadi mahasiswa jangan sombong dong . kas...  negative"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Prepare data set\n",
    "df = pd.read_csv('train_preprocess.tsv', sep='\\t', header=None, names = ['Text', 'Kategori']) \n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Kategori\n",
       "positive    6416\n",
       "negative    3436\n",
       "neutral     1148\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Kategori.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Text Cleansing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re \n",
    "\n",
    "def cleansing(sent):\n",
    "    # Mengubah kata menjadi huruf kecil semua dengan menggunakan fungsi lower()\n",
    "    string = sent.lower()\n",
    "    # Menghapus emoticon dan tanda baca menggunakan \"RegEx\" dengan script di bawah\n",
    "    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)\n",
    "    return string"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Text</th>\n",
       "      <th>Kategori</th>\n",
       "      <th>text_clean</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>warung ini dimiliki oleh pengusaha pabrik tahu...</td>\n",
       "      <td>positive</td>\n",
       "      <td>warung ini dimiliki oleh pengusaha pabrik tahu...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>mohon ulama lurus dan k212 mmbri hujjah partai...</td>\n",
       "      <td>neutral</td>\n",
       "      <td>mohon ulama lurus dan k212 mmbri hujjah partai...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>lokasi strategis di jalan sumatera bandung . t...</td>\n",
       "      <td>positive</td>\n",
       "      <td>lokasi strategis di jalan sumatera bandung   t...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>betapa bahagia nya diri ini saat unboxing pake...</td>\n",
       "      <td>positive</td>\n",
       "      <td>betapa bahagia nya diri ini saat unboxing pake...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>duh . jadi mahasiswa jangan sombong dong . kas...</td>\n",
       "      <td>negative</td>\n",
       "      <td>duh   jadi mahasiswa jangan sombong dong   kas...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                Text  Kategori  \\\n",
       "0  warung ini dimiliki oleh pengusaha pabrik tahu...  positive   \n",
       "1  mohon ulama lurus dan k212 mmbri hujjah partai...   neutral   \n",
       "2  lokasi strategis di jalan sumatera bandung . t...  positive   \n",
       "3  betapa bahagia nya diri ini saat unboxing pake...  positive   \n",
       "4  duh . jadi mahasiswa jangan sombong dong . kas...  negative   \n",
       "\n",
       "                                          text_clean  \n",
       "0  warung ini dimiliki oleh pengusaha pabrik tahu...  \n",
       "1  mohon ulama lurus dan k212 mmbri hujjah partai...  \n",
       "2  lokasi strategis di jalan sumatera bandung   t...  \n",
       "3  betapa bahagia nya diri ini saat unboxing pake...  \n",
       "4  duh   jadi mahasiswa jangan sombong dong   kas...  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Menambahkan kolom text clean\n",
    "df['text_clean'] = df.Text.apply(cleansing)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "neg = df.loc[df['Kategori'] == 'negative'].text_clean.tolist()\n",
    "neu = df.loc[df['Kategori'] == 'neutral'].text_clean.tolist()\n",
    "pos = df.loc[df['Kategori'] == 'positive'].text_clean.tolist()\n",
    "\n",
    "neg_sentiment = df.loc[df['Kategori'] == 'negative'].Kategori.tolist()\n",
    "neu_sentiment = df.loc[df['Kategori'] == 'neutral'].Kategori.tolist()\n",
    "pos_sentiment = df.loc[df['Kategori'] == 'positive'].Kategori.tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "total_data = pos + neu + neg\n",
    "labels = pos_sentiment + neu_sentiment + neg_sentiment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'warung ini dimiliki oleh pengusaha pabrik tahu yang sudah puluhan tahun terkenal membuat tahu putih di bandung   tahu berkualitas   dipadu keahlian memasak   dipadu kretivitas   jadilah warung yang menyajikan menu utama berbahan tahu   ditambah menu umum lain seperti ayam   semuanya selera indonesia   harga cukup terjangkau   jangan lewatkan tahu bletoka nya   tidak kalah dengan yang asli dari tegal  '"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "total_data[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tokenizer.pickle has been created.\n",
      "x_pad_sequences.pickle has been created.\n"
     ]
    }
   ],
   "source": [
    "#Tokenizing and Applying pad_sequences\n",
    "\n",
    "import pickle\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "\n",
    "max_features = 100000\n",
    "tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)\n",
    "tokenizer.fit_on_texts(total_data)\n",
    "\n",
    "with open('tokenizer.pickle', 'wb') as handle:\n",
    "  pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "  print('tokenizer.pickle has been created.')\n",
    "\n",
    "X = tokenizer.texts_to_sequences(total_data)\n",
    "X = pad_sequences(X)\n",
    "\n",
    "with open('x_pad_sequences.pickle','wb') as handle:\n",
    "  pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "  print('x_pad_sequences.pickle has been created.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "11000"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[False False  True]\n",
      " [False False  True]\n",
      " [False False  True]\n",
      " ...\n",
      " [ True False False]\n",
      " [ True False False]\n",
      " [ True False False]]\n"
     ]
    }
   ],
   "source": [
    "Y = pd.get_dummies(labels)\n",
    "Y = Y.values\n",
    "\n",
    "print(Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "y_labels.pickle has been created\n"
     ]
    }
   ],
   "source": [
    "with open('y_labels.pickle','wb') as handle:\n",
    "  pickle.dump(Y,handle, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "  print(\"y_labels.pickle has been created\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(11000, 3)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[False, False,  True],\n",
       "       [False, False,  True],\n",
       "       [False, False,  True],\n",
       "       [False, False,  True],\n",
       "       [False, False,  True],\n",
       "       [False, False,  True],\n",
       "       [False, False,  True],\n",
       "       [False, False,  True],\n",
       "       [False, False,  True],\n",
       "       [False, False,  True]])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y[0:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       a      b     c\n",
       "0  False  False  True\n",
       "1  False  False  True\n",
       "2  False  False  True\n",
       "3  False  False  True\n",
       "4  False  False  True"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "some_df = pd.DataFrame(data=Y, columns=['a','b','c'])\n",
    "some_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>labels</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       a      b     c    labels\n",
       "0  False  False  True  positive\n",
       "1  False  False  True  positive\n",
       "2  False  False  True  positive\n",
       "3  False  False  True  positive\n",
       "4  False  False  True  positive"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "some_df['labels'] = labels\n",
    "some_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>labels</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>7564</th>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         a      b      c    labels\n",
       "7564  True  False  False  negative"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "some_df[some_df['labels']=='negative'].iloc[0:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>labels</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>6416</th>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>neutral</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          a     b      c   labels\n",
       "6416  False  True  False  neutral"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "some_df[some_df['labels']=='neutral'].iloc[0:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>labels</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       a      b     c    labels\n",
       "0  False  False  True  positive"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "some_df[some_df['labels']=='positive'].iloc[0:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "file = open(\"x_pad_sequences.pickle\",'rb')\n",
    "x = pickle.load(file)\n",
    "file.close()\n",
    "\n",
    "file = open(\"y_labels.pickle\",'rb')\n",
    "y = pickle.load(file)\n",
    "file.close()\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Embedding, LSTM\n",
    "from tensorflow.keras import optimizers\n",
    "from tensorflow.keras.callbacks import EarlyStopping"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From g:\\Pribadi\\Andi\\Challenge_Platinum\\challenge_platinum\\Lib\\site-packages\\keras\\src\\backend.py:873: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.\n",
      "\n",
      "WARNING:tensorflow:From g:\\Pribadi\\Andi\\Challenge_Platinum\\challenge_platinum\\Lib\\site-packages\\keras\\src\\optimizers\\__init__.py:309: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.\n",
      "\n",
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " embedding (Embedding)       (None, 96, 100)           10000000  \n",
      "                                                                 \n",
      " lstm (LSTM)                 (None, 64)                42240     \n",
      "                                                                 \n",
      " dense (Dense)               (None, 3)                 195       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 10042435 (38.31 MB)\n",
      "Trainable params: 10042435 (38.31 MB)\n",
      "Non-trainable params: 0 (0.00 Byte)\n",
      "_________________________________________________________________\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:`lr` is deprecated in Keras optimizer, please use `learning_rate` or use the legacy optimizer, e.g.,tf.keras.optimizers.legacy.Adam.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "None\n",
      "Epoch 1/2\n",
      "WARNING:tensorflow:From g:\\Pribadi\\Andi\\Challenge_Platinum\\challenge_platinum\\Lib\\site-packages\\keras\\src\\utils\\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From g:\\Pribadi\\Andi\\Challenge_Platinum\\challenge_platinum\\Lib\\site-packages\\keras\\src\\utils\\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From g:\\Pribadi\\Andi\\Challenge_Platinum\\challenge_platinum\\Lib\\site-packages\\keras\\src\\engine\\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From g:\\Pribadi\\Andi\\Challenge_Platinum\\challenge_platinum\\Lib\\site-packages\\keras\\src\\engine\\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "734/734 [==============================] - 112s 138ms/step - loss: 0.4554 - accuracy: 0.8207 - val_loss: 0.3517 - val_accuracy: 0.8650\n",
      "Epoch 2/2\n",
      "734/734 [==============================] - 100s 136ms/step - loss: 0.2019 - accuracy: 0.9268 - val_loss: 0.3301 - val_accuracy: 0.8827\n"
     ]
    }
   ],
   "source": [
    "embed_dim = 100\n",
    "units = 64\n",
    "\n",
    "model=Sequential()\n",
    "model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))\n",
    "model.add(LSTM(units, dropout=0.2))\n",
    "model.add(Dense(3,activation='softmax'))\n",
    "model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "print(model.summary())\n",
    "\n",
    "adam=optimizers.Adam(lr=0.001)\n",
    "model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])\n",
    "\n",
    "es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)\n",
    "history = model.fit(x_train, y_train, epochs=2, batch_size=12, validation_data=(x_test, y_test), verbose=1, callbacks=[es])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "69/69 [==============================] - 3s 22ms/step\n",
      "Testing selesai\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.84      0.83      0.84       685\n",
      "           1       0.86      0.80      0.83       233\n",
      "           2       0.91      0.93      0.92      1282\n",
      "\n",
      "    accuracy                           0.88      2200\n",
      "   macro avg       0.87      0.85      0.86      2200\n",
      "weighted avg       0.88      0.88      0.88      2200\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "\n",
    "predictions = model.predict(x_test)\n",
    "y_pred = predictions\n",
    "matrix_test = metrics.classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))\n",
    "print(\"Testing selesai\")\n",
    "print(matrix_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "g:\\Pribadi\\Andi\\Challenge_Platinum\\challenge_platinum\\Lib\\site-packages\\keras\\src\\engine\\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n",
      "  saving_api.save_model(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model has been created.\n"
     ]
    }
   ],
   "source": [
    "model.save('model.h5')\n",
    "print('Model has been created.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 2s 2s/step\n",
      "Text:   rasa syukur  cukup  \n",
      "Sentiment:  positive\n"
     ]
    }
   ],
   "source": [
    "import re\n",
    "from keras.models import load_model\n",
    "\n",
    "input_text = \"\"\"\n",
    "Rasa syukur, cukup.\n",
    "\"\"\"\n",
    "\n",
    "def cleansing(sent):\n",
    "    # Mengubah kata menjadi huruf kecil semua dengan menggunakan fungsi lower()\n",
    "    string = sent.lower()\n",
    "    # Menghapus emoticon dan tanda baca menggunakan \"RegEx\" dengan script di bawah\n",
    "    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)\n",
    "    return string\n",
    "\n",
    "sentiment = ['negative', 'neutral', 'positive']\n",
    "\n",
    "text = [cleansing(input_text)]\n",
    "predicted = tokenizer.texts_to_sequences(text)\n",
    "guess = pad_sequences(predicted, maxlen=X.shape[1])\n",
    "\n",
    "model = load_model('model.h5')\n",
    "prediction = model.predict(guess)\n",
    "polarity = np.argmax(prediction[0])\n",
    "\n",
    "print(\"Text: \",text[0])\n",
    "print(\"Sentiment: \",sentiment[polarity])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "challenge_platinum",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
