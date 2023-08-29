
import matplotlib.pyplot as plt   
import pandas as pd
from src.emotion.main import get_va_values
DATA_DIR = "data/interim"

(bd_v, bd_a) = get_va_values(f"{DATA_DIR}/test_wavs/blue_danube.wav")
(ms_v, ms_a) = get_va_values(f"{DATA_DIR}/test_wavs/moonlight_sonata.wav")
(vw_v, vw_a) = get_va_values(f"{DATA_DIR}/test_wavs/vivaldi-winter.wav")
print(bd_v, bd_a, ms_v, ms_a, vw_v, vw_a)

VAs = pd.read_csv(f"{DATA_DIR}/VAdata.csv")
emotions = VAs["Emotion"]
valences = VAs["Valence"]
arousals = VAs["      Arousal"]
annotate = VAs["Annotate"]

fig, ax = plt.subplots()
plt.xlim([1,9])
plt.ylim([1,9])
plt.xlabel("Valence")
plt.ylabel("Arousal")
ax.scatter(valences, arousals)

for i, txt in enumerate(emotions):
    if(annotate[i]):
        ax.annotate(txt, (valences[i], arousals[i]))

ax.scatter(bd_v, bd_a, color = "m")
ax.scatter(ms_v, ms_a, color = "r")
ax.scatter(vw_v, vw_a, color = "k")

plt.show()