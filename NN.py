{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "70c7cf9f-4e43-4f17-b262-715577666ba5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to /Users/ondom/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package punkt to /Users/ondom/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package wordnet to /Users/ondom/nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "# import library\n",
    "import pandas as pd\n",
    "import re\n",
    "import nltk\n",
    "nltk.download('stopwords')\n",
    "nltk.download('punkt')\n",
    "nltk.download('wordnet')\n",
    "import string\n",
    "from io import StringIO\n",
    "import Sastrawi\n",
    "import joblib\n",
    "import pickle\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns \n",
    "from wordcloud import WordCloud\n",
    "\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.tokenize import sent_tokenize\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "from Sastrawi.Stemmer.StemmerFactory import StemmerFactory\n",
    "from nltk import pos_tag\n",
    "\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.model_selection import KFold\n",
    "\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import cross_val_score, GridSearchCV\n",
    "from sklearn.metrics import classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c4824363-15ef-41e4-91ba-da4333ed020c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import raw dataset for preprocessing\n",
    "df = pd.read_csv('train_preprocess.tsv',sep ='\\t', names = [\"text\", \"sentiment\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "40b5202d-8c2e-414a-9c18-22efa26f5256",
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
       "      <th>text</th>\n",
       "      <th>sentiment</th>\n",
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
       "                                                text sentiment\n",
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
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3a2b5c1b-203c-42ba-a998-0c7a305bac0a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "positive    6416\n",
       "negative    3436\n",
       "neutral     1148\n",
       "Name: sentiment, dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['sentiment'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9800daa9-8da1-48ea-97ce-a6213ab9cec6",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "               text sentiment\n",
      "count         11000     11000\n",
      "unique        10933         3\n",
      "top     tidak kesal  positive\n",
      "freq              4      6416\n"
     ]
    }
   ],
   "source": [
    "# inspect the Dataframe\n",
    "print(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8de4c085-e9c3-4b4d-b1fe-55ab8eb11736",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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uP3FxsZisVhYsmRJgut6eXklaXtJ5ePjQ758+VJ0Rfj777/P22+/zfPPP897771H9uzZcXBwYPDgwXG+50qUKMH+/fv57bff+P333/npp5/44osveOedd3j33XcB6/dv7dq1WbBgAcuWLeO///0v48aNY/78+TRt2jTZGZPi5ZdfZvr06QwePJgaNWrg6+uLxWKhY8eOqT4108Ok1fepiGRuKnwmKFy4MCtWrKBWrVpJLk0eHh7Uq1ePVatWcfz48XijQMl16dIlVq5cybvvvhvnooDw8PAkbefBEaRHKViwIAD79++PMwoTHR3NkSNHaNCgQZL2/Tj3Svnp06eBlP05PExS3n9KFS5cGMMwCAoKomjRoqmyzaTmb9GiBdOmTWP9+vXUqFEjyfubN28eTz31FF9//XWc5ZcvXyZnzpxxlnl6etKhQwc6dOhAdHQ0zzzzDGPGjOGNN96wTaGSN29eXnrpJV566SUiIyOpWLEiY8aMSXHhu/f9+bhyO2/ePLp3785HH31kW3br1i0uX74cZ72k/pzs2LGD2NjYOKN8+/btsz0vIvI4OofPBO3btycmJob33nsv3nN3796N94/Dg0aMGIFhGDz33HNERUXFe37r1q18++23Scp0b1TgwVGACRMmJGk79+bie9x7AOu5jy4uLnz66adx9vv1119z5coVmjdvnqR937Ny5coEly9evBj49xBySv8cEpKU959SzzzzDI6Ojrz77rvx/twMw+DChQtJ3mZS8w8bNgxPT0969eqV4N1lDh06xMSJEx/6ekdHx3jZf/zxR06ePBln2YPvxcXFhZIlS2IYBnfu3CEmJibeIdNcuXKRL18+bt++naj38ij+/v7UqVOHb775hmPHjsV57v78Cb2fSZMmxbvzTVI+52bNmnHmzBnmzp1rW3b37l0mTZqEl5cXdevWTerbEZEsSCN8Jqhbty59+/Zl7NixhIWF0ahRI5ydnQkPD+fHH39k4sSJtG3b9qGvr1mzJp9//jkvvfQSxYsXj3OnjdWrV/PLL78wevToJGXy8fGhTp06jB8/njt37pA/f36WLVvGkSNHkrSdSpUqATBw4EAaN26Mo6MjHTt2THBdf39/3njjDd59912aNGlCq1at2L9/P1988QVVqlSJc4FBUrRu3ZqgoCBatmxJ4cKFuX79OitWrODXX3+lSpUqtGzZEkj5n0NCypcvj6OjI+PGjePKlSu4urpSr169h54nmBKFCxdm9OjRvPHGG0RERPD000/j7e3NkSNHWLBgAX369GHo0KFJ3qafnx9TpkzB29sbT09PqlWr9tDzNQsXLsx3331Hhw4dKFGiRJw7baxbt842fcjDtGjRglGjRtGzZ09q1qzJzp07mT17drzz7ho1akSePHmoVasWuXPnZu/evXz22Wc0b94cb29vLl++TIECBWjbti3lypXDy8uLFStWsHnz5jijbSnx6aef8uSTT1KxYkX69OlDUFAQERERLFq0yHYruhYtWjBr1ix8fX0pWbIk69evZ8WKFfGmaErK90mfPn2YOnUqPXr0YOvWrQQGBjJv3jzWrl3LhAkTUnTRjIhkIel9WbA9edjEyw9KaCoIwzCMadOmGZUqVTLc3d0Nb29vo0yZMsawYcOMU6dOJWr/W7duNTp37mzky5fPcHZ2NrJly2bUr1/f+Pbbb21TXdw/GeuDeGAC4xMnThht2rQx/Pz8DF9fX6Ndu3bGqVOn4q33qClS7t69a7z88suGv7+/YbFYEjVFy2effWYUL17ccHZ2NnLnzm28+OKLcSZeftw+H/T9998bHTt2NAoXLmy4u7sbbm5uRsmSJY0333zTNoXF/RLz55DQ1B2GkfAkuV9++aVRqFAhw9HRMVETL//4449xXv+w76uHfQY//fST8eSTTxqenp6Gp6enUbx4caN///7G/v374+RMaFqThL43f/75Z6NkyZKGk5NToqdoOXDggNG7d28jMDDQcHFxMby9vY1atWoZkyZNijNZdELTsrz66qtG3rx5DXd3d6NWrVrG+vXr431WU6dONerUqWPkyJHDcHV1NQoXLmy89tprxpUrVwzDsE4x89prrxnlypUzvL29DU9PT6NcuXLGF1988dj3m9hpWQzDMHbt2mX7GXFzczOKFStmvP3227bnL126ZPTs2dPImTOn4eXlZTRu3NjYt29fvPdtGIn/PjEM68TL97br4uJilClTJl62pPysi0jWYzEMnckrIiIiYs90Dp+IiIiInVPhExEREbFzKnwiIiIidk6FT0RERMTOqfCJiIiI2DkVPhERERE7p8InIiIiYudU+ERERETsnAqfiIiIiJ1T4RMRERGxcyp8IiIiInZOhU9ERETEzqnwiYiIiNg5FT4RERERO6fCJyIiImLnVPhERERE7JwKn4iIiIidU+ETERERsXMqfCIiIiJ2ToVPRERExM6p8ImIiIjYORU+ERERETunwiciIiJi51T4REREROycCp+IiIiInVPhExEREbFzKnwiIiIidk6FT0RERMTOqfCJiIiI2DkVPhERERE7p8InIiIiYudU+ERERETsnAqfiIiIiJ1T4RMReYzAwEAmTJhgdgwRkWRT4RMRuxMSEsLgwYPNjiEikmGo8IlIlmQYBnfv3jU7hohIulDhE5F0FRISwsCBAxk2bBjZs2cnT548jBw50vb85cuX6dWrF/7+/vj4+FCvXj22b99ue75Hjx48/fTTcbY5ePBgQkJCbM//+eefTJw4EYvFgsViISIigtWrV2OxWFiyZAmVKlXC1dWV0NBQDh06ROvWrcmdOzdeXl5UqVKFFStWpMMnISKSflT4RCTdffvtt3h6erJx40bGjx/PqFGjWL58OQDt2rUjMjKSJUuWsHXrVipWrEj9+vW5ePFiorY9ceJEatSoQe/evTl9+jSnT58mICDA9vzw4cP54IMP2Lt3L2XLliUqKopmzZqxcuVK/v77b5o0aULLli05duxYmrx3EREzOJkdQESynrJlyzJixAgAgoOD+eyzz1i5ciXu7u5s2rSJyMhIXF1dAfjwww9ZuHAh8+bNo0+fPo/dtq+vLy4uLnh4eJAnT554z48aNYqGDRvaHmfPnp1y5crZHr/33nssWLCAX375hQEDBqT0rYqIZAgqfCKS7sqWLRvncd68eYmMjGT79u1ERUWRI0eOOM/fvHmTQ4cOpcq+K1euHOdxVFQUI0eOZNGiRZw+fZq7d+9y8+ZNjfCJiF1R4RORdOfs7BznscViITY2lqioKPLmzcvq1avjvcbPzw8ABwcHDMOI89ydO3cSvW9PT884j4cOHcry5cv58MMPKVKkCO7u7rRt25bo6OhEb1NEJKNT4RORDKNixYqcOXMGJycnAgMDE1zH39+fXbt2xVkWFhYWp0S6uLgQExOTqH2uXbuWHj160KZNG8A64hcREZGs/CIiGZUu2hCRDKNBgwbUqFGDp59+mmXLlhEREcG6det488032bJlCwD16tVjy5YtzJw5k/DwcEaMGBGvAAYGBrJx40YiIiI4f/48sbGxD91ncHAw8+fPJywsjO3bt9O5c+dHri8ikhmp8IlIhmGxWFi8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e++999h8KaHClwWsOrKKp+c8zbXoa49fWcQOuboazHulB34uqTjx6t4PITYGKn2cetuUZMtod75ISLdu3bh58yZVq1bF0dGRQYMGxZkEefr06QwaNIgWLVoQHR1NnTp1WLx4sW3ELSYmhv79+3PixAl8fHxo0qQJn3zySYL7qlmzJv369aNDhw5cuHCBESNG2KZmeVCXLl0YM2YMBQsWpFatWnGee++99/D392fs2LEcPnwYPz8/KlasyH/+85+Hvs+mTZvi5OTEihUraNy4MQAODg78+uuvvPzyy9SpUwdPT0+aNm1qK7yJtWXLFp566inb41deeQWwHjqdMWMGAG3atGHKlCmMHTuWgQMHUqxYMX766SeefPIxV+X/I0+ePKxatYqQkBC6dOnCd999x5w5cxg4cCClS5emWLFifPrpp7apV8B6PuSnn37KqFGjeOedd6hduzarV6/m66+/pk+fPlSsWJGAgADef//9eFdMr1+/nitXrtC2bdskfRZJZTHunwRI7M6S8CW0mduG2zGJuBpRMoTiP0Wxb6en2THsyrThM+ldJuHRiBQrOhAqJ+OcQEmWW7duceTIEYKCgjLsRLoJCQkJoXz58rY58Ozd559/zi+//MLSpbo48HE6dOhAuXLlHlmiHyYpPw8a4bNjKw+v5JkfnlHZkyyt+zMH067sARz41HpuX+XP4DF3ORDJKvr27cvly5e5du1apriXrlmio6MpU6YMQ4YMSfN9qfDZqTVH19BqTitu3b1ldhQR05QtdZPJT8e/dVOqC/8CjBioMlmlTwTrIc4333zT7BgZnouLC2+99Va67EuFzw5tOLGB5t8158adtJ3TRyQj8/ExmPdyK9wdLz1+5dRwcKq19FWdptIn8STmggGRtKRpWezMttPbaDq7qS7QkCxvxmvjCfZekb47PfQVbH4xffcpIpIIKnx2ZOfZnTSa1YjLty6bHUXEVK/22EKbQsMfv2JaODgVdr5rzr5FRB5Chc9OHLhwgAazGnDh5gWzo4iYqnb1K3xQP3HTL6SZnSPhYMIT4oqImEGFzw6cu36OprObEnk90uwoIqbKnSuGub3q4uSQAa5M3/winPjV7BQiIoAKX6Z36+4tWs9pzeFLh82OImIqR0eDOcNeJa/7drOjWBkxsLYDnFtvdhIRERW+zMwwDJ5b8BzrT+gfFJHRLy0l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",
      "text/plain": [
       "<Figure size 500x700 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Assuming df is your DataFrame and \"label\" is the column containing sentiment labels\n",
    "\n",
    "# Count the occurrences of each sentiment label\n",
    "sentiment_counts = df[\"sentiment\"].value_counts()\n",
    "\n",
    "# Define colors for each sentiment\n",
    "colors = ['green', 'orange', 'blue']  # Add more colors as needed\n",
    "\n",
    "# Plotting a pie chart\n",
    "plt.figure(figsize=(5, 7))\n",
    "plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)\n",
    "\n",
    "# Adding a legend\n",
    "legend_labels = [f'{label} ({count} kata)' for label, count in zip(sentiment_counts.index, sentiment_counts)]\n",
    "plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 0.85))\n",
    "\n",
    "plt.title('Pie Chart of Sentiment Classification')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d721325b-9bc9-4b60-8395-73664fed4286",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Box plot\n",
    "df['ln_text'] = df['text'].str.len()\n",
    "sns.set_style('whitegrid')\n",
    "sns.boxplot(y = df['ln_text'] , x = df['sentiment']);\n",
    "plt.ylabel('Panjang Kata')\n",
    "plt.xlabel('Sentiment')\n",
    "plt.title('Persebaran Data Sentiment')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f338c410-566a-4305-ae13-609f0b00ce74",
   "metadata": {},
   "source": [
    "# __Text Cleansing__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d04d32ea-ca23-463b-9b44-9045b8412c11",
   "metadata": {},
   "outputs": [],
   "source": [
    "def cleansing(sent):\n",
    "    string = sent.lower()\n",
    "    string = re.sub(r'[^a-zA-Z0-9]',' ',string)\n",
    "    return string"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "41bbfc98-b7cd-43a6-8580-e9880875a218",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['text_clean'] = df['text'].apply(cleansing)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8048a0bf-9bf7-4d98-950e-24a1ea41ce9b",
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
       "      <th>text</th>\n",
       "      <th>sentiment</th>\n",
       "      <th>ln_text</th>\n",
       "      <th>text_clean</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>warung ini dimiliki oleh pengusaha pabrik tahu...</td>\n",
       "      <td>positive</td>\n",
       "      <td>404</td>\n",
       "      <td>warung ini dimiliki oleh pengusaha pabrik tahu...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>mohon ulama lurus dan k212 mmbri hujjah partai...</td>\n",
       "      <td>neutral</td>\n",
       "      <td>102</td>\n",
       "      <td>mohon ulama lurus dan k212 mmbri hujjah partai...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>lokasi strategis di jalan sumatera bandung . t...</td>\n",
       "      <td>positive</td>\n",
       "      <td>184</td>\n",
       "      <td>lokasi strategis di jalan sumatera bandung   t...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>betapa bahagia nya diri ini saat unboxing pake...</td>\n",
       "      <td>positive</td>\n",
       "      <td>93</td>\n",
       "      <td>betapa bahagia nya diri ini saat unboxing pake...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>duh . jadi mahasiswa jangan sombong dong . kas...</td>\n",
       "      <td>negative</td>\n",
       "      <td>214</td>\n",
       "      <td>duh   jadi mahasiswa jangan sombong dong   kas...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                text sentiment  ln_text  \\\n",
       "0  warung ini dimiliki oleh pengusaha pabrik tahu...  positive      404   \n",
       "1  mohon ulama lurus dan k212 mmbri hujjah partai...   neutral      102   \n",
       "2  lokasi strategis di jalan sumatera bandung . t...  positive      184   \n",
       "3  betapa bahagia nya diri ini saat unboxing pake...  positive       93   \n",
       "4  duh . jadi mahasiswa jangan sombong dong . kas...  negative      214   \n",
       "\n",
       "                                          text_clean  \n",
       "0  warung ini dimiliki oleh pengusaha pabrik tahu...  \n",
       "1  mohon ulama lurus dan k212 mmbri hujjah partai...  \n",
       "2  lokasi strategis di jalan sumatera bandung   t...  \n",
       "3  betapa bahagia nya diri ini saat unboxing pake...  \n",
       "4  duh   jadi mahasiswa jangan sombong dong   kas...  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88da340c-10b1-4ae6-a170-cd5fb141076e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55ff97ed-bca3-4dfd-8cf6-0ed4b72361c8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "03162dd6-da1c-430c-bece-dd8ceeff6330",
   "metadata": {},
   "source": [
    "# __Feature Extraction Bag of Word__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "196b1dce-6276-46cd-88ec-76e4e4768766",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_preprocessed = df['text_clean'].tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "47f8423d-4026-41aa-8228-f8acc953829b",
   "metadata": {},
   "outputs": [],
   "source": [
    "count_vect = CountVectorizer()\n",
    "count_vect.fit(data_preprocessed)\n",
    "\n",
    "x = count_vect.transform(data_preprocessed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "4b36d5e4-e71d-4afe-9d28-982b33afb2fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(count_vect, open(\"feature.p\", \"wb\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b376214b-84b0-4999-be2c-ea902ad32900",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "3a2bf900-f3f8-4434-9ebd-5384980c1e8e",
   "metadata": {},
   "source": [
    "# __Splitting Dataset__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "db662abc-549e-4bc0-a644-b258f76d1c30",
   "metadata": {},
   "outputs": [],
   "source": [
    "#berikut ini adalah split existing data menjadi 80% data trainning dan 20% data test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "e78bb44b-97c2-4350-a58c-20cdc5c475cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "classes = df['sentiment']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "95b596fe-b54d-48eb-99be-1b4cd7c8e354",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0        positive\n",
       "1         neutral\n",
       "2        positive\n",
       "3        positive\n",
       "4        negative\n",
       "           ...   \n",
       "10995    positive\n",
       "10996    positive\n",
       "10997     neutral\n",
       "10998    negative\n",
       "10999    positive\n",
       "Name: sentiment, Length: 11000, dtype: object"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "classes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "1db664e6-2871-449e-8ed4-b810169d4929",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x, classes, test_size = 0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "bb7699ad-f507-40d9-834b-f1f33f16c479",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x_train size: 8800\n",
      "y_train size: 8800\n",
      "x_test size: 2200\n",
      "y_test size: 2200\n"
     ]
    }
   ],
   "source": [
    "print(f\"x_train size: {x_train.shape[0]}\")\n",
    "print(f\"y_train size: {y_train.shape[0]}\")\n",
    "print(f\"x_test size: {x_test.shape[0]}\")\n",
    "print(f\"y_test size: {y_test.shape[0]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "17c9de49-4e35-4e9f-9c4c-c6c3ac58bd65",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MLPClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MLPClassifier</label><div class=\"sk-toggleable__content\"><pre>MLPClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "MLPClassifier()"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = MLPClassifier()\n",
    "model.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "ab8e5037-9648-4031-9fc8-2dcbd24e0932",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Dump model ke dalam pickle\n",
    "pickle.dump(model, open(\"model.p\", \"wb\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "88e3adf8-78f4-41b0-bab2-88c26e50eeab",
   "metadata": {},
   "source": [
    "# __Evaluasi Model__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "d2cbc838-1a53-494e-a408-74f879e12660",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "    negative       0.80      0.80      0.80       698\n",
      "     neutral       0.75      0.66      0.70       234\n",
      "    positive       0.88      0.91      0.90      1268\n",
      "\n",
      "    accuracy                           0.85      2200\n",
      "   macro avg       0.81      0.79      0.80      2200\n",
      "weighted avg       0.84      0.85      0.84      2200\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = model.predict(x_test)\n",
    "\n",
    "print(classification_report(y_test, y_pred)) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37dfe349-dba3-4129-af0c-ae6a2f048703",
   "metadata": {},
   "source": [
    "# __Cross Validation__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "38908daf-6328-4f1f-bdbb-98c556d93b15",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training ke- 1\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "    negative       0.78      0.79      0.78       680\n",
      "     neutral       0.77      0.65      0.70       239\n",
      "    positive       0.88      0.89      0.89      1281\n",
      "\n",
      "    accuracy                           0.83      2200\n",
      "   macro avg       0.81      0.78      0.79      2200\n",
      "weighted avg       0.83      0.83      0.83      2200\n",
      "\n",
      "======================================================\n",
      "Training ke- 2\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "    negative       0.80      0.77      0.79       706\n",
      "     neutral       0.75      0.71      0.73       220\n",
      "    positive       0.88      0.91      0.90      1274\n",
      "\n",
      "    accuracy                           0.85      2200\n",
      "   macro avg       0.81      0.80      0.81      2200\n",
      "weighted avg       0.84      0.85      0.84      2200\n",
      "\n",
      "======================================================\n",
      "Training ke- 3\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "    negative       0.79      0.81      0.80       682\n",
      "     neutral       0.86      0.71      0.78       215\n",
      "    positive       0.89      0.91      0.90      1303\n",
      "\n",
      "    accuracy                           0.86      2200\n",
      "   macro avg       0.85      0.81      0.83      2200\n",
      "weighted avg       0.86      0.86      0.86      2200\n",
      "\n",
      "======================================================\n",
      "Training ke- 4\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "    negative       0.78      0.80      0.79       698\n",
      "     neutral       0.78      0.66      0.71       229\n",
      "    positive       0.88      0.90      0.89      1273\n",
      "\n",
      "    accuracy                           0.84      2200\n",
      "   macro avg       0.81      0.79      0.80      2200\n",
      "weighted avg       0.84      0.84      0.84      2200\n",
      "\n",
      "======================================================\n",
      "Training ke- 5\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "    negative       0.77      0.81      0.79       670\n",
      "     neutral       0.79      0.65      0.71       245\n",
      "    positive       0.89      0.89      0.89      1285\n",
      "\n",
      "    accuracy                           0.84      2200\n",
      "   macro avg       0.82      0.79      0.80      2200\n",
      "weighted avg       0.84      0.84      0.84      2200\n",
      "\n",
      "======================================================\n",
      "\n",
      "\n",
      "\n",
      "Rata-rata Accuracy:  0.8442727272727272\n"
     ]
    }
   ],
   "source": [
    "#KF ini adalah unutk melihat konsistensi nilai presisi, recall, dan f1\n",
    "\n",
    "kf = KFold(n_splits=5,random_state=42,shuffle=True)\n",
    "\n",
    "accuracies = []\n",
    "\n",
    "y = classes\n",
    "\n",
    "for iteration, data in enumerate(kf.split(x), start=1):\n",
    "\n",
    "    data_train   = x[data[0]]\n",
    "    target_train = y[data[0]]\n",
    "\n",
    "    data_test    = x[data[1]]\n",
    "    target_test  = y[data[1]]\n",
    "\n",
    "    clf = MLPClassifier()\n",
    "    clf.fit(data_train,target_train)\n",
    "\n",
    "    preds = clf.predict(data_test)\n",
    "\n",
    "    # for the current fold only    \n",
    "    accuracy = accuracy_score(target_test,preds)\n",
    "\n",
    "    print(\"Training ke-\", iteration)\n",
    "    print(classification_report(target_test,preds))\n",
    "    print(\"======================================================\")\n",
    "\n",
    "    accuracies.append(accuracy)\n",
    "\n",
    "# this is the average accuracy over all folds\n",
    "average_accuracy = np.mean(accuracies)\n",
    "\n",
    "print()\n",
    "print()\n",
    "print()\n",
    "print(\"Rata-rata Accuracy: \", average_accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20192ff1-10d1-43b4-a561-a4c295da5876",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "4324b8a6-9356-4830-9728-73a716d5ae9f",
   "metadata": {},
   "source": [
    "# __Predict__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "d84a5e7c-9165-48e8-9d07-81750b8d1079",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sentiment:\n",
      "\n",
      "neutral\n"
     ]
    }
   ],
   "source": [
    "# Melakukan prediksi dengan model yang telah dibuat\n",
    "original_text = '''\n",
    "saya mau makan\n",
    "'''\n",
    "\n",
    "text = count_vect.transform([cleansing(original_text)])\n",
    "\n",
    "result = model.predict(text)[0]\n",
    "print(\"Sentiment:\")\n",
    "print()\n",
    "print(result)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bdbc9217-75b4-4a0a-ba8e-7a91e7777a98",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "b4efc27d-97b2-4feb-be3f-5641963e8a47",
   "metadata": {
    "scrolled": true
   },
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
       "      <th>True Data</th>\n",
       "      <th>Prediction</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1021</th>\n",
       "      <td>negative</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>477</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>479</th>\n",
       "      <td>negative</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>773</th>\n",
       "      <td>negative</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>746</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>77</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2024</th>\n",
       "      <td>neutral</td>\n",
       "      <td>neutral</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2129</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>positive</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>670</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>386</th>\n",
       "      <td>neutral</td>\n",
       "      <td>neutral</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>340</th>\n",
       "      <td>negative</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>919</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>613</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>54</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>929</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2135</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>187</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>211</th>\n",
       "      <td>positive</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2038</th>\n",
       "      <td>positive</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     True Data Prediction\n",
       "1021  negative   negative\n",
       "477   positive   positive\n",
       "479   negative   negative\n",
       "773   negative   negative\n",
       "746   positive   positive\n",
       "77    positive   positive\n",
       "2024   neutral    neutral\n",
       "2129  positive   positive\n",
       "20    positive   negative\n",
       "670   positive   positive\n",
       "386    neutral    neutral\n",
       "340   negative   negative\n",
       "919   positive   positive\n",
       "613   positive   positive\n",
       "54    positive   positive\n",
       "929   positive   positive\n",
       "2135  positive   positive\n",
       "187   positive   positive\n",
       "211   positive   negative\n",
       "2038  positive   positive"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prediction_data = pd.DataFrame({'True Data':target_test.values.ravel(), 'Prediction':preds})\n",
    "prediction_data.sample(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "e64840b9-887f-48c5-8d8a-4a908fd8e620",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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d1S9enUFBASwaxSeC87OzsqbN6+OHj1qOvbkyRP9/fffKlGihC5evKiZM2fK3d1dw4YN06ZNm1SwYEH98ccftisaeM7RVCDZcufOrREjRigwMNB0rF69eipcuLA++OADnT17VocOHdL48ePl6OiozJkzq3Pnzjp69Kj8/f11+fJlLVq0SIcOHTL9QpQrVy79+uuvCggI0KFDhzRy5EhJMm3D6ejoqMDAQN26dSvt3zCQSlLyvVO5cmXdv39fy5cvV0BAgKZPn26aNijFfW+EhobqypUriSYYuXLlUlBQkPbu3auAgAD5+/tr27ZtNtseGkhtPXr00Lx58/TLL7/o4sWLGj9+vB4/fqyWLVsqR44c+vbbb7VgwQIFBATot99+U2BgoCpWrJjgOtmyZdOVK1cUEhJig3cBPD9oKmCR119/3fTwOSluzvjChQsVGxurzp07a/DgwXrllVf04YcfSoqb4z1v3jytXbtWr732mv766y81btxYdnZ2kqRp06bp9OnTatWqlcaMGaPmzZvL3d1dp0+fliS1bdtWly9fVps2bfgLKzI0S793ihcvrlGjRmnhwoVq166djEajmjVrZnr9Sy+9JDc3N7322mum7xdzLVq0UJs2bTRkyBB17NhR+/fv16hRo3Tx4kUaCzwXevXqpU6dOmn8+PHq0KGDbt68qeXLlytPnjxydXWVr6+vfv75Z7Vq1Uo+Pj56//33Va9evQTX6dSpk/7444940xMBWM5g5Dc1PEPnzp1TdHR0vL8O9e3bV1WqVNHgwYNtWBkAAABSC0kFnqlr166pZ8+e2rNnjwIDA7V69Wrt3btXr776qq1LAwAAQCohqcAzt3DhQq1atUohISEqUaKEhgwZoiZNmti6LAAAAKQSmgoAAAAAVmH6EwAAAACr0FQAAAAAsApNBQAAAACr0FQAAAAAsApNBQAAAACr0FQAeGE0atRI5cqVM31UqlRJzZs317Jly1L1Pt26dZOvr68kafTo0Ro9evR/viYqKkrff/99iu+5bt06NWrUKNFz+/fvV7ly5VJ87XLlymn//v0peq2vr6+6deuW4nsDADKGLLYuAADS0tixY9WyZUtJUnR0tPbt26dx48YpV65cateuXarfb9y4cckat2nTJi1atEidO3dO9RoAAHjWSCoAvFCcnZ3l6uoqV1dXFSxYUO3bt1ft2rW1bdu2Z3Y/Z2fn/xzHI4MAABkZTQWAF16WLFlkZ2cnKW7q0uTJk9W4cWM1aNBAYWFhunHjhvr37y8PDw81atRIfn5+iomJMb1++/btatasmTw9PeXj4xPv3L+nP/34449q3ry5PDw81KVLF506dUr79+/XmDFjFBgYqHLlyun69esyGo2aP3++6tWrp+rVq6t///4KCgoyXefWrVt699135enpqfbt2+vatWspfv9hYWEaM2aMateurcqVK6t58+basWNHvDEHDx5U06ZN5eHhoffee0+hoaGmc+fOnVO3bt3k7u6uZs2aaeXKlSmuBQCQMdFUAHhhPXnyRNu2bdOePXvUuHFj0/F169Zp9uzZ8vPzU/bs2eXt7S0XFxetX79e06dP108//aRFixZJki5cuKChQ4eqa9euWrt2raKjo3X48OFE7/fHH39o3Lhxeuedd7RhwwZVrlxZ/fr1k5eXl8aOHasCBQpo9+7dKliwoFasWKGffvpJn3zyiVatWiUXFxf16tVLT548kSS99957io2N1erVq9WnTx999dVXKf46TJ06VZcvX9bSpUu1ceNGVa9eXePGjVNUVJRpzMqVKzVu3DitXLlSly9f1vTp0yVJkZGR6tOnj6pVq6YNGzZo1KhRWrBggX744YcU1wMAyHhYUwHghTJx4kRNnjxZUtwvxFmzZtU777yjNm3amMY0aNBAVatWlSTt3btXQUFBWr16tTJlyqSSJUtq1KhRGjNmjAYNGqS1a9eqevXq6tGjhyRp/Pjx+vXXXxO996pVq9S6dWt17dpVkjRy5EjZ2dkpNDRUzs7Oypw5s1xdXSVJX3zxhSZOnKhatWpJknx8fFSvXj398ccfKlq0qP766y/9+uuvKlSokMqUKaOTJ09q69atKfqa1KhRQz179lTZsmUlSb169dLq1asVEhKiggULSpK8vb31yiuvSJI+/PBD9ezZUx9++KG2bNkiFxcXDR06VJJUvHhxBQYG6uuvv34ma1QAAOkTTQWAF8qQIUPUtGlTSZKDg4NcXV2VOXPmeGMKFy5s+vfFixd1//59VatWzXQsNjZWkZGRunfvni5evKgKFSqYztnZ2cX73Nzly5fVpUsX0+f29vYaNWpUgnGPHj3SzZs3NWzYMGXK9P+BcmRkpK5cuaLHjx8rV65cKlSokOlclSpVUtxUtGvXTjt27ND333+vS5cu6e+//5akeNO4qlSpYvp3xYoVFR0drWvXrunSpUs6c+aMvLy8TOdjYmISfE0BAM83mgoALxQXFxe5ubk9dYyDg4Pp39HR0SpZsqQWLFiQYNw/C7D/vcj6n/UZ/5YlS/L+J/efX+Y/++wzlShRIt65nDlzau/evcm+Z3KMHDlSf/31l9q2bauuXbvK1dVVb7zxRrwx5k3CP/e2s7NTdHS0ateurQkTJqT4/gCAjI81FQDwFCVKlFBQUJDy5MkjNzc3ubm56fr165o3b54MBoPKlCmjEydOmMbHxsbqzJkziV7Lzc0t3rmYmBg1atRIhw8flsFgMB3PkSOHXFxcdOfOHdM9CxYsqNmzZ+vy5csqW7asQkNDdfXqVdNrTp8+naL3FxYWpo0bN+rTTz/VkCFD9Oqrr5oWYZs3LufOnTP9+/jx47Kzs1ORIkVUokQJXb58WUWKFDHVevToUS1fvjxF9QAAMiaaCgB4inr16qlw4cL64IMPdPbsWR06dEjjx4+Xo6OjMmfOrM6dO+vkyZNauHChLl26pJkzZ8bbpclct27dtGHDBq1fv15Xr17V9OnTZTQaValSJTk6Oio0NFRXrlxRdHS0evTooblz5+qXX37RlStX9OGHH+rIkSMqWbKkSpUqpdq1a2vs2LE6c+aMduzYoRUrVvzne/n999/jfezfv1/29vZydHTUtm3bdP36df3xxx/y8fGRpHgLtT/99FPt3btXR48e1ZQpU9SlSxc5OjqqTZs2ioyM1IQJE3Tx4kXt2rVLU6dOlYuLS+r8BwAAyBCY/gQAT5E5c2YtXLhQkydPVufOnZUtWzY1b97ctBbCzc1NCxcu1PTp07Vw4UI1adLEtKD532rUqKGJEydq/vz5unPnjipXrqxFixYpa9aseumll+Tm5qbXXntN33zzjXr37q1Hjx5pwoQJCgsLU+XKlbVkyRLlzJlTUtwv+ePHj1eXLl1UqFAhdevWTevWrXvqe+nTp0+8z/Pnz6/ff/9ds2fP1syZM7V8+XIVKVJEAwYM0Ny5c3X69GmVKlVKktSzZ0+NGzdO9+7dU4sWLTRixAhJkpOTkz7//HNNmzZN7dq1U65cufTWW2+pX79+Vn3dAQAZi8HIE5cAAAAAWIHpTwAAAACsQlMBAAAAwCo0FQAAAACsQlMBAAAAwCo0FQAAAACsQlMBAAAAwCo0FQAAAACsQlMBAAAAwCo0FQAAAACsQlMBAAAAwCo0FQAAAACs8n/OhM9yw1bxBQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1000x700 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def plot_confusion_matrix(y_test, y_pred):  \n",
    "     conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred))  \n",
    "     fig = plt.figure(figsize=(10, 7))  \n",
    "     sns.heatmap(conf_mat, annot=True, annot_kws={\"size\": 16}, fmt=\"g\", \n",
    "                xticklabels=[\"Negatif\", \"Neutral\", \"Positif\"],\n",
    "                yticklabels=[\"Negatif\", \"Neutral\", \"Positif\"])  \n",
    "     plt.title(\"Confusion Matrix\")  \n",
    "     plt.xlabel(\"Predicted Label\")  \n",
    "     plt.ylabel(\"True Label\")  \n",
    "     plt.show()\n",
    "   \n",
    "plot_confusion_matrix(y_test,y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ebfee77c-14e9-4d74-be41-2aef4953a73b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Platinum",
   "language": "python",
   "name": "platinum"
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
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
