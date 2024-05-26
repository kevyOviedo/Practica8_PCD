#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from PIL import Image
img = Image.open('chart.jpeg')
img_array = np.array(img)

years = range(2000, 2012)
apples = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931, 0.934, 0.936, 0.937, 0.9375, 0.9372, 0.939]
oranges = [0.962, 0.941, 0.930, 0.923, 0.918, 0.908, 0.907, 0.904, 0.901, 0.898, 0.9, 0.896]

flowers_df = sns.load_dataset("iris")
setosa_df = flowers_df[flowers_df.species == 'setosa']
versicolor_df = flowers_df[flowers_df.species == 'versicolor']
virginica_df = flowers_df[flowers_df.species == 'virginica']

tips_df = sns.load_dataset("tips")

flights_df = sns.load_dataset("flights").pivot(index="month", columns="year", values="passengers")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
# Use the axes for plotting
axes[0,0].plot(years, apples, 's-b')
axes[0,0].plot(years, oranges, 'o--r')
axes[0,0].set_xlabel('Anio')
axes[0,0].set_ylabel('Cosecha (Toneladas por hectarea)')
axes[0,0].legend(['Manzanas', 'Naranjas']);
axes[0,0].set_title('Cosechas en Kanto')
# Pass the axes into seaborn
axes[0,1].set_title('Largo de Sepal vs Ancho de Sepal')
sns.scatterplot(x=flowers_df.sepal_length,
y=flowers_df.sepal_width,
hue=flowers_df.species,
s=100,
ax=axes[0,1]);

# Use the axes for plotting
axes[0,2].set_title('Distribucion de anchura del Sepal')
axes[0,2].hist([setosa_df.sepal_width,
versicolor_df.sepal_width, virginica_df.sepal_width],
bins=np.arange(2, 5, 0.25),
stacked=True);
axes[0,2].legend(['Setosa', 'Versicolor',
'Virginica']);
axes[0,2].set_xlabel('Anchura Sepal')
axes[0,2].set_ylabel('Cantidad')
# Pass the axes into seaborn
axes[1,0].set_title('Facturas de restaurante')
sns.barplot(x='day', y='total_bill', hue='sex',
data=tips_df, ax=axes[1,0]);

# Pass the axes into seaborn
axes[1,1].set_title('Trafico aereo')
sns.heatmap(flights_df, cmap='Blues', ax=axes[1,1]);

# Plot an image using the axes
axes[1,2].set_title('Foto HOLA')
axes[1,2].imshow(img)
axes[1,2].grid(False)
axes[1,2].set_xticks([])
axes[1,2].set_yticks([])
axes[1,2].set_xlabel('Eje x de foto')
axes[1,2].set_ylabel('Eje y de foto')
plt.tight_layout(pad=2);

plt.show()