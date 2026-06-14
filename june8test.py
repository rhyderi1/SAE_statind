from datetime import datetime
import matplotlib as plt

now = datetime.now()

print(f'The datetime is {now:%Y%m%d%H%M%S}')