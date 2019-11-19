import pandas as pd
import sys
sys.modules[__name__].__dict__.clear()
vowels='aeiou'
word='Anurag Sethi'


#:wqword=word.lower
first=word[:0]
print(first)
if word[:0] in vowels:
    print('1')

