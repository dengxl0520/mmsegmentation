import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt
X = np.load('output.npy')

labels = np.load('labels.npy')
X = X[:2000]
labels = labels[:2000]

perplexity = np.arange(100, 200, 5)
divergence = []

# for i in perplexity:
#     model = TSNE(n_components=2, init="pca", perplexity=i)
#     reduced = model.fit_transform(X)
#     divergence.append(model.kl_divergence_)
# fig = px.line(x=perplexity, y=divergence, markers=True)
# fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
# fig.update_traces(line_color="red", line_width=1)
# fig.write_image('ttest.png')
# fig.show()
tsne = TSNE(n_components=2,perplexity=40, random_state=42)
X_train_tsne = tsne.fit_transform(X)
plt.scatter(X_train_tsne [:, 0], X_train_tsne [:, 1], 20, labels)
plt.show()