import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


from torch_geometric.datasets import KarateClub

# Import dataset from PyTorch Geometric
dataset = KarateClub()

# Print information
print(dataset)
print("------------")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

# Print first element
print(f"Graph: {dataset[0]}")

data = dataset[0]

print(f"x = {data.x.shape}")
print(data.x)

print(f"edge_index = {data.edge_index.shape}")
print(data.edge_index)

from torch_geometric.utils import to_dense_adj

A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
print(f"A = {A.shape}")
print(A)

print(f"y = {data.y.shape}")
print(data.y)

print(f"train_mask = {data.train_mask.shape}")
print(data.train_mask)

print(f"Edges are directed: {data.is_directed()}")
print(f"Graph has isolated nodes: {data.has_isolated_nodes()}")
print(f"Graph has loops: {data.has_self_loops()}")

from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(12, 12))
plt.axis("off")
nx.draw_networkx(
    G,
    pos=nx.spring_layout(G, seed=0),
    with_labels=True,
    node_size=800,
    node_color=data.y,
    cmap="hsv",
    vmin=-2,
    vmax=3,
    width=0.8,
    edge_color="grey",
    font_size=14,
)
# plt.show()


from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(dataset.num_features, 3)
        self.out = Linear(3, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        return h, z


model = GCN()
print(model)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


# Calculate accuracy
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)


# Data for animations
embeddings = []
losses = []
accuracies = []
outputs = []

# Training loop
for epoch in range(201):
    # Clear gradients
    optimizer.zero_grad()

    # Forward pass
    h, z = model(data.x, data.edge_index)

    # Calculate loss function
    loss = criterion(z, data.y)

    # Calculate accuracy
    acc = accuracy(z.argmax(dim=1), data.y)

    # Compute gradients
    loss.backward()

    # Tune parameters
    optimizer.step()

    # Store data for animations
    embeddings.append(h)
    losses.append(loss)
    accuracies.append(acc)
    outputs.append(z.argmax(dim=1))

    # Print metrics every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%")


""" from matplotlib import animation

plt.rcParams["animation.bitrate"] = 3000


def animate(i):
    G = to_networkx(data, to_undirected=True)
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=0),
        with_labels=True,
        node_size=800,
        node_color=outputs[i],
        cmap="hsv",
        vmin=-2,
        vmax=3,
        width=0.8,
        edge_color="grey",
        font_size=14,
    )
    plt.title(
        f"Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%",
        fontsize=18,
        pad=20,
    )


fig = plt.figure(figsize=(12, 12))
plt.axis("off")

anim = animation.FuncAnimation(
    fig, animate, np.arange(0, 200, 10), interval=500, repeat=True
)
# html = HTML(anim.to_html5_video())
# 将动画保存为 MP4 文件
try:
    print("正在保存动画，这可能需要一些时间...")
    anim.save("my_animation.mp4", writer="ffmpeg", fps=(1000 / 500))  # fps = 2
    print("动画已保存为 my_animation.mp4")
except Exception as e:
    print(f"保存动画时出错: {e}")
    print("请确保你已经安装了 ffmpeg 并且它在你的系统 PATH 中。")
    print(
        "或者，通过 plt.rcParams['animation.ffmpeg_path'] = '路径/到/ffmpeg' 来指定路径。"
    ) """

# Print embeddings
print(f"Final embeddings = {h.shape}")
print(h)


# Get first embedding at epoch = 0
embed = h.detach().cpu().numpy()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection="3d")
ax.patch.set_alpha(0)
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.scatter(
    embed[:, 0], embed[:, 1], embed[:, 2], s=200, c=data.y, cmap="hsv", vmin=-2, vmax=3
)

# plt.show()

from matplotlib import animation


def animate(i):
    embed = embeddings[i].detach().cpu().numpy()
    ax.clear()
    ax.scatter(
        embed[:, 0],
        embed[:, 1],
        embed[:, 2],
        s=200,
        c=data.y,
        cmap="hsv",
        vmin=-2,
        vmax=3,
    )
    plt.title(
        f"Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%",
        fontsize=18,
        pad=40,
    )


fig = plt.figure(figsize=(12, 12))
plt.axis("off")
ax = fig.add_subplot(projection="3d")
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

anim = animation.FuncAnimation(
    fig, animate, np.arange(0, 200, 10), interval=800, repeat=True
)

# 保存动画
try:
    print("正在保存动画，这可能需要一些时间...")
    # fps = 1000 / interval
    anim.save("my_3d_animation.mp4", writer="ffmpeg", fps=(1000 / 800))
    print("动画已保存为 my_3d_animation.mp4")
except Exception as e:
    print(f"保存动画时出错: {e}")
    print("请确保你已经安装了 ffmpeg 并且它在你的系统 PATH 中。")
    print(
        "或者，通过 plt.rcParams['animation.ffmpeg_path'] = '路径/到/ffmpeg' 来指定路径。"
    )
