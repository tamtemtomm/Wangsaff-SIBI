import re, json
from collections import Counter

class Correction():
  def __init__(self, WORDS:dict):
    self.WORDS = WORDS
  
  def __call__(self, word:str):
    candidates = (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])
    return max(candidates, key=self.P)
  
  def P(self, word): 
    return - self.WORDS.get(word, 0)

  def known(self, words):
    return set(w for w in words if w in self.WORDS)
  
  def edits1(self, word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes +  replaces + inserts)
  
  def edits2(self, word):
    return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

def load_bow(BOW_PATH):
    with open(BOW_PATH) as f:
        data = f.read()
    WORDS = json.loads(data)
    return WORDS