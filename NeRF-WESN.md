# NeRF-WESN：输入、空间序列、储备池与路径损耗输出的完整公式

本文以当前 workspace 的实际实现为依据，用于理解代码和撰写论文。中文部分解释每一步的构造，后文提供可以改写进英文论文的方法描述。本文解释当前模型，不修改训练程序，也不预设模型一定优于其他方法。

## 1. 先回答最关键的问题：sequence 究竟是什么？

**一条 sequence，是从一个待预测接收机 Rx 出发，沿某一条射线从近到远生成的 16 个虚拟空间采样点，经位置编码和频率编码后得到的有序向量序列。**

当前默认设置下：

- 一个 Rx 生成 8 条射线。
- 每条射线生成 16 个空间采样点，因此产生一条长度为 16 的序列。
- 每个采样点对应一个 28 维输入向量。
- 一个 Rx 对应 8 条独立序列，而不是一条长度为 128 的序列。
- WESN 对序列中的每个点都输出特征，随后由渲染模块汇聚成 Rx 处的一个 PL 预测值。

可以先记住这条计算链：

```text
一个查询 Rx + Tx + 频率
          │
          ├── 从 Rx 出发构造 8 条水平射线
          │       │
          │       └── 每条射线按近到远采样 16 个点
          │                    │
          │                    └── 每点：27 维位置编码 + 1 维频率 = 28 维
          │                                      │
          │                         WESN：固定储备池 + 窗口 + 可训练读出
          │                                      │
          │                         每点：1 个 density + 32 维特征
          │                                      │
          └── Tx、方向和频率 ───────→ radiance MLP
                                                 │
                                    每点：log-amplitude + phase
                                                 │
                                    射线内复数加权求和
                                                 │
                                    射线之间平均功率
                                                 │
                                       -10 log10(·)
                                                 │
                             direct PL，或 FSPL + residual PL
```

以下三种理解都不符合当前代码：

1. 将训练集里不同 Rx 的 PL 测量值按某种顺序串起来，作为 WESN 的输入。
2. 将 f2、f3 或不同频率依次排列，作为 WESN 的 sequence。
3. 将不同 training steps 或不同 mini-batches 的 hidden state 连续传下去。

当前模型使用的是**射线上的有序空间特征序列**。训练 PL 只作为最终输出的监督标签，不作为这条序列的输入。

## 2. 代码入口与本文使用的符号

### 2.1 对应的源码

| 内容 | 文件与函数 |
| --- | --- |
| 接收网格、Tx、PL 标签读取 | [datasets/data_loader.py](../datasets/data_loader.py)，`load_scene`、`build_local_receiver_grid` |
| 射线、位置编码、radiance、渲染、两种 PL 模式 | [Models/model.py](../Models/model.py)，`RayField`、`RadioMapField` |
| 固定储备池、窗口和读出层 | [Models/wesn.py](../Models/wesn.py)，`ESNCell`、`WESN` |
| 频率特征、FSPL | [propagation.py](../propagation.py) |
| PL 损失、优化、预训练与微调 | [main.py](../main.py)，`fit_field`、`run_scene` |
| 当前配置 | [configs/icassp.yml](../configs/icassp.yml) |

`Models/model.py` 中还包含供 MLP attenuation 消融使用的 `SkipMLP`。默认 `attenuation: wesn` 时不经过这个 MLP 分支。

### 2.2 符号约定

| 符号 | 含义 | 当前默认值或维度 |
| --- | --- | --- |
| $i$ | 查询 Rx 的编号 | 每次 forward 中的 batch 索引 |
| $q$ | 射线编号 | $q=0,\ldots,R-1$ |
| $j$ | 射线上的采样点编号 | 本文使用 $j=1,\ldots,S$ |
| $B$ | 一次 forward 的 Rx 数量 | 默认训练 batch size 为 128 |
| $R$ | 每个 Rx 的射线数 | 8 |
| $S$ | 每条射线的采样点数，即 sequence length | 16 |
| $K_{\mathrm{PE}}$ | 坐标 Fourier 编码尺度数 | 4 |
| $C$ | 每个 sequence step 的输入维度 | 28 |
| $H_r$ | 储备池 hidden dimension | 32 |
| $L_w$ | 显式窗口长度 | 4，即当前点和前 3 个点 |
| $\mathbf r_i$ | Rx 坐标，单位 m | 3 维 |
| $\mathbf t$ | Tx 坐标，单位 m | 3 维 |
| $f$ | 载波频率 | f2 为 1.8 GHz，f3 为 3.5 GHz |
| $y_i(f)$ | Rx 处真实 PL 标签 | 一个 dB 标量 |
| $\widehat y_i(f)$ | Rx 处预测 PL | 一个 dB 标量 |

源码的采样点索引从 0 开始，本文公式从 1 开始；因此源码的 `(arange(S) + 0.5) / S` 对应本文的 $(j-1/2)/S$。

**`num_freq=4` 指坐标位置编码使用 4 个 Fourier 尺度，与数据集有几个载波频段无关。**

## 3. 外部输入和原始监督标签

### 3.1 一个预测任务的外部输入

对第 $i$ 个查询位置，模型接口为：

```python
prediction_db = model(rx_m, tx_m, frequency_ghz)
```

对应：

$$
\mathbf r_i=(x_i,y_i,z_i),\qquad
\mathbf t=(x_T,y_T,z_T),\qquad f.
$$

因此外部具有 7 个物理标量：Rx xyz、Tx xyz 和频率。但是内部**并非直接将这个 7 维向量送入 WESN**。Rx 用于生成射线采样点，Tx 在后续 radiance 分支中加入，频率进入两个分支。

场景尺寸也是模型已知的上下文，用于确定射线边界和坐标尺度。它由图像的宽、高推导，不是 material feature。

### 3.2 接收坐标如何从 PL 图构造

设 PL 图宽、高为 $W_{\mathrm{px}},H_{\mathrm{px}}$，网格间隔为：

$$
\Delta=0.25\ \mathrm m.
$$

列索引为 $c$、行索引为 $a$ 的位置对应：

$$
\mathbf r_{a,c}=(c\Delta,\ a\Delta,\ z_T).
$$

当前数据中 Rx 和 Tx 的高度相同。负的原始建筑 `X_min`、`Y_min` 不会再被加到这些局部坐标上。

模型使用的两个水平跨度为：

$$
L_x=(W_{\mathrm{px}}-1)\Delta,
\qquad
L_y=(H_{\mathrm{px}}-1)\Delta.
$$

这是首末像素中心的坐标跨度。图像覆盖的像素 footprint 为 $W_{\mathrm{px}}\Delta$、$H_{\mathrm{px}}\Delta$，两者差一个网格间隔，不能混写。

### 3.3 模型不接收哪些信息

当前 NeRF-WESN 不接收：

- material / building input 图像；
- 某个查询点的真实 PL；
- 查询点周围实测 PL 的 KNN 序列；
- 源频段测试位置的 PL；
- 其他 Rx 的 reservoir hidden state。

`building_id`、场景名称等用于管理实验，不作为神经网络特征。

### 3.4 标签的含义

读取的 PNG 像素保留原始量化 PL 的 dB 数值，转换为浮点数后使用，不除以 255。

当前 NeRF-WESN 的 direct 和 residual 模式都使用原始 PL dB 标签，**不进行 Min-Max 标签归一化**。后文的坐标归一化是另一件事。

## 4. 从一个 Rx 构造 8 条射线

### 4.1 射线方向

对 $q=0,\ldots,R-1$：

$$
\theta_q=\frac{2\pi q}{R},
\qquad
\mathbf d_q=(\cos\theta_q,\ \sin\theta_q,\ 0).
$$

当 $R=8$ 时，对应水平方向角：

$$
0^\circ,45^\circ,90^\circ,135^\circ,
180^\circ,225^\circ,270^\circ,315^\circ.
$$

这些方向固定且均匀分布在 x-y 平面，没有随机方向扰动。全部射线起点都是 Rx，且保持相同高度。

Tx 坐标不参与这一步。当前射线不要求指向 Tx，也没有用建筑几何计算反射或折射路径。

### 4.2 射线长度

射线沿 $\mathbf d_q$ 向外，理想情况下在首次碰到场景矩形边界时结束。

对水平坐标轴 $k\in\{x,y\}$，令对应跨度为 $L_k$。定义：

$$
b_{iqk}=\begin{cases}
(L_k-r_{ik})/d_{qk}, & d_{qk}>\epsilon_d,\\
-r_{ik}/d_{qk}, & d_{qk}<-\epsilon_d,\\
+\infty, & |d_{qk}|\le\epsilon_d,
\end{cases}
\qquad \epsilon_d=10^{-6}.
$$

代码使用的长度为：

$$
\ell_{iq}=\max\!\left(\min(b_{iqx},b_{iqy}),\ 0.01\ \mathrm m\right).
$$

这里的 0.01 m 是当前 `_rays` 实现对**射线积分段长度**的数值处理，避免边界外向射线长度为零。它与 Tx-Rx 的 FSPL 距离无关。

需要准确交代：对于恰好位于矩形边界、朝外的射线，这个处理会让部分内部采样点略微落在矩形外。因此不能将当前实现描述为“所有采样点都严格裁剪在建筑或地图内部”。本文只是记录已有实现，不改变这项设置。

### 4.3 中点采样

将每条射线分成 $S$ 段，在每段中点取样：

$$
\tau_j=\frac{j-\tfrac12}{S},
\qquad j=1,\ldots,S.
$$

$$
\boxed{
\mathbf p_{iqj}=\mathbf r_i+\tau_j\ell_{iq}\mathbf d_q.
}
$$

默认 $S=16$，取样比例为：

$$
\frac1{32},\frac3{32},\frac5{32},\ldots,\frac{31}{32}.
$$

第一个采样点在 Rx 附近，但不直接取 Rx；最后一个点接近射线终点，但不直接取边界端点。

**序列顺序就是 $\mathbf p_{iq1}\rightarrow\cdots\rightarrow\mathbf p_{iq16}$：从 Rx 附近走向场景边界。**

射线内部的物理采样间隔为：

$$
\Delta s_{iq}^{\mathrm{physical}}=\frac{\ell_{iq}}S.
$$

它取决于 Rx 和射线方向，通常不等于数据集的 0.25 m 网格间隔。0.25 m 控制候选 Rx 网格；$\ell_{iq}/S$ 控制内部虚拟射线采样。

## 5. 每个 sequence step 的 28 维输入如何形成

### 5.1 三个坐标分量都要归一化

定义：

$$
D=\max(L_x,L_y),\qquad \mathbf e=(L_x,L_y,D).
$$

$$
\overline{\mathbf p}_{iqj}
=\mathbf p_{iqj}\oslash\mathbf e
=\left(\frac{p_x}{L_x},\frac{p_y}{L_y},\frac{p_z}{D}\right).
$$

$\oslash$ 表示逐元素相除。后面 Tx 的归一化也使用同一个 $\mathbf e$。

**在 NeRF-WESN 中，高度 z 也除以 $D$。** 此处与普通 MLP baseline 的“x/y 缩放、z 保留米单位”不同，不能把 MLP 的公式直接搬到 NeRF-WESN。

### 5.2 Fourier 位置编码

对于归一化坐标 $\mathbf v=(v_x,v_y,v_z)$，定义：

$$
\gamma(\mathbf v)=\left[
\mathbf v,
\left\{\sin(\pi 2^k v_a)\right\}_{a\in\{x,y,z\},\,k=0}^{K_{\mathrm{PE}}-1},
\left\{\cos(\pi 2^k v_a)\right\}_{a\in\{x,y,z\},\,k=0}^{K_{\mathrm{PE}}-1}
\right].
$$

默认 $K_{\mathrm{PE}}=4$，每个坐标的四个尺度为：

$$
\pi,\ 2\pi,\ 4\pi,\ 8\pi.
$$

输出维度：

$$
D_{\mathrm{PE}}=3+3K_{\mathrm{PE}}+3K_{\mathrm{PE}}
=3(1+2K_{\mathrm{PE}})=27.
$$

与代码 `flatten(-2)` 一致，具体排列是：

```text
[x, y, z,
 sin(πx), sin(2πx), sin(4πx), sin(8πx),
 sin(πy), sin(2πy), sin(4πy), sin(8πy),
 sin(πz), sin(2πz), sin(4πz), sin(8πz),
 cos(πx), cos(2πx), cos(4πx), cos(8πx),
 cos(πy), cos(2πy), cos(4πy), cos(8πy),
 cos(πz), cos(2πz), cos(4πz), cos(8πz)]
```

上面 x、y、z 指归一化后的坐标。这里的 Fourier 编码由固定函数构造，本身没有可训练参数。

### 5.3 载波频率特征

令 $f_0=1\ \mathrm{GHz}$。频率编码是：

$$
\eta(f)=\ln\left(\frac f{f_0}\right).
$$

实际实现接受 GHz 数值，因此：

$$
\eta(f_2)=\ln1.8\approx0.5877867,
\qquad
\eta(f_3)=\ln3.5\approx1.2527630.
$$

它是自然对数，不是 $\log_{10}$，也不是 Min-Max 归一化或可训练 embedding。

### 5.4 单点输入与完整序列

第 $j$ 个点的输入为：

$$
\boxed{
\mathbf u_{iqj}=
\left[\gamma(\overline{\mathbf p}_{iqj}),\eta(f)\right]
\in\mathbb R^{28}.
}
$$

同一条射线的 sequence 为：

$$
\mathbf U_{iq}=
\begin{bmatrix}
\mathbf u_{iq1}\\
\mathbf u_{iq2}\\
\vdots\\
\mathbf u_{iqS}
\end{bmatrix}
\in\mathbb R^{16\times28}.
$$

同一条 sequence 的频率保持不变，随 j 变化的是空间位置编码。

Tx 和方向没有在这里显式拼接。Rx、方向通过生成的点序列间接影响 WESN；Tx 和方向会在 radiance MLP 中显式加入。

### 5.5 多个 Rx 和多条射线如何组织

位置和频率拼接后：

$$
\mathbf U\in\mathbb R^{B\times R\times S\times C}.
$$

传入 WESN 前：

```python
field_input.reshape(batch * num_rays, num_samples, -1)
```

也就是：

$$
[B,R,S,C]\longrightarrow[BR,S,C].
$$

第一维把“Rx 编号和射线编号”合并为独立 batch 项；**真正的 sequence 轴始终是长度 S 的第二维**。

当 $B=128,R=8$ 时，WESN 一次处理 1024 条长度为 16 的序列，不是一条包含 1024 个接收机的序列。

## 6. 固定随机储备池的状态更新

本节为与 PyTorch 中的 `inputs @ W_in` 保持一致，状态与输入采用行向量约定。

### 6.1 初始状态和更新方程

每一条 Rx—射线序列均从零状态开始：

$$
\mathbf h_{iq0}=\mathbf0\in\mathbb R^{H_r}.
$$

对 $j=1,\ldots,S$：

$$
\widetilde{\mathbf h}_{iqj}
=\tanh\left(
\mathbf u_{iqj}\mathbf W_{\mathrm{in}}
+\mathbf h_{iq,j-1}\mathbf W_{\mathrm{res}}
+\mathbf b_{\mathrm{res}}
\right),
$$

$$
\boxed{
\mathbf h_{iqj}=(1-\lambda)\mathbf h_{iq,j-1}
+\lambda\widetilde{\mathbf h}_{iqj}.
}
$$

其中：

$$
\mathbf W_{\mathrm{in}}\in\mathbb R^{C\times H_r},\quad
\mathbf W_{\mathrm{res}}\in\mathbb R^{H_r\times H_r},\quad
\mathbf b_{\mathrm{res}}\in\mathbb R^{H_r}.
$$

当前实际配置为 $H_r=32,\lambda=1$，所以状态更新简化为：

$$
\mathbf h_{iqj}=\tanh\left(
\mathbf u_{iqj}\mathbf W_{\mathrm{in}}
+\mathbf h_{iq,j-1}\mathbf W_{\mathrm{res}}
+\mathbf b_{\mathrm{res}}
\right).
$$

$\mathbf h_{iqj}$ 已经包含当前输入 $\mathbf u_{iqj}$，并通过递归依赖前面的空间输入。

### 6.2 随机权重如何构造

输入矩阵和初始循环矩阵使用 Xavier uniform 初始化。

对循环矩阵增加稀疏掩码：

$$
M_{ab}\sim\operatorname{Bernoulli}(c),
\qquad
\mathbf A=\widetilde{\mathbf W}_{\mathrm{res}}\odot\mathbf M.
$$

当前 RayField 的连接概率 $c=0.5$。它表示期望保留约一半的循环连接，不表示矩阵最终一定恰好一半元素非零。

使用最大特征值模定义谱半径：

$$
\rho(\mathbf A)=\max_k|\lambda_k(\mathbf A)|.
$$

若 $\rho(\mathbf A)>0$，缩放为：

$$
\mathbf W_{\mathrm{res}}
=\frac{\rho_0}{\rho(\mathbf A)}\mathbf A,
\qquad \rho_0=0.9.
$$

偏置初始化为全零。输入矩阵、循环矩阵和储备池偏置均以 `requires_grad=False` 固定。

谱半径设置是控制储备池动力学与记忆的设计选择；仅凭 0.9 这一设置，不能声称已经对任意输入和任意非线性配置严格证明 echo-state property。

### 6.3 状态在哪些边界重置

`WESN.forward()` 每次都会执行：

```python
h_t = torch.zeros(B_wesn, hidden_dim, ...)
```

因此：

- 不同射线之间没有 hidden state 传递。
- 不同 Rx 之间没有 hidden state 传递。
- 不同 mini-batches 之间没有 hidden state 传递。
- f2 和 f3 之间没有 hidden state 传递。

共享的是储备池矩阵和网络参数。预训练传到微调的是参数，而非上一频段处理完之后的 hidden state。

构造模型时使用一次零输入的 dummy forward，是为了提前创建 lazy 参数，让优化器和 checkpoint 能看见它们；那次 dummy forward 的状态不会保留给真实样本。

## 7. Window 如何构造，以及它为什么能保留局部空间信息

### 7.1 窗口作用于完整输入特征

默认 `win_len=4` 表示：

$$
[\mathbf u_{iqj},\mathbf u_{iq,j-1},\mathbf u_{iq,j-2},\mathbf u_{iq,j-3}].
$$

即**当前点加前面 3 个点，总共 4 个点**。

窗口中的每一项都是完整的 28 维向量，包括 27 维位置编码和 1 维频率。它不是对四个 PL 测量值做滑动窗口。

### 7.2 序列开头的零填充

对于 $j\le0$，令输入历史为完整零向量：

$$
\mathbf u_{iqj}=\mathbf0\in\mathbb R^{28}.
$$

零填充是把编码后的全部 28 维置零，不能理解为先虚构一个原点坐标再做编码：原点的 cosine 编码等于 1，且频率特征通常也不为零，两者并不相同。

拼接特征为：

$$
\boxed{
\mathbf v_{iqj}=
[\mathbf h_{iqj},\mathbf u_{iqj},\ldots,\mathbf u_{iq,j-L_w+1}]
\in\mathbb R^{H_r+L_wC}.
}
$$

前四个位置具体为：

$$
\begin{aligned}
\mathbf v_{iq1}&=[\mathbf h_{iq1},\mathbf u_{iq1},\mathbf0,\mathbf0,\mathbf0],\\
\mathbf v_{iq2}&=[\mathbf h_{iq2},\mathbf u_{iq2},\mathbf u_{iq1},\mathbf0,\mathbf0],\\
\mathbf v_{iq3}&=[\mathbf h_{iq3},\mathbf u_{iq3},\mathbf u_{iq2},\mathbf u_{iq1},\mathbf0],\\
\mathbf v_{iq4}&=[\mathbf h_{iq4},\mathbf u_{iq4},\mathbf u_{iq3},\mathbf u_{iq2},\mathbf u_{iq1}].
\end{aligned}
$$

后续位置以相同方式向前滑动，窗口不跨越不同射线。

### 7.3 两种互补的空间记忆

1. **储备池递归状态**：$\mathbf h_{iqj}$ 依赖当前点及其前缀，具有递归空间上下文。
2. **显式输入窗口**：最近 4 个输入直接拼接到读出层，为近期输入提供直接通道。

因此，不能把 `win_len=4` 解释为“整个模型只能看到 4 个点”。窗口之外的更早输入仍可能通过储备池状态影响当前输出。

这里的窗口连接可以描述为输入到读出层的直接连接或 time-delay embedding。它与 `SkipMLP` 中的隐藏层拼接不是同一个结构。

## 8. WESN 读出层输出什么

### 8.1 线性读出

WESN 对每个采样点输出：

$$
\boxed{
\mathbf o_{iqj}=
\mathbf v_{iqj}\mathbf W_{\mathrm{out}}^{\mathsf T}
+\mathbf b_{\mathrm{out}}
\in\mathbb R^{H_r+1}.
}
$$

PyTorch 中：

$$
\mathbf W_{\mathrm{out}}\in
\mathbb R^{(H_r+1)\times(H_r+L_wC)}.
$$

默认：

$$
H_r+L_wC=32+4\times28=144,
\qquad H_r+1=33.
$$

对应 `Linear(144, 33)`。

### 8.2 33 维输出的拆分

$$
\mathbf o_{iqj}=[\xi_{iqj},\mathbf z_{iqj}],
\quad \xi_{iqj}\in\mathbb R,
\quad \mathbf z_{iqj}\in\mathbb R^{32}.
$$

第一维经 softplus 变成非负 density：

$$
\sigma_{iqj}=\operatorname{softplus}(\xi_{iqj})
=\ln(1+e^{\xi_{iqj}}).
$$

剩余 32 维 $\mathbf z_{iqj}$ 作为 latent feature，送入后面的 radiance MLP。

模型使用所有 16 个位置的输出，不只使用最后一个 hidden state，也不先把 WESN 输出平均成一个向量。

这些内部 density 和 latent features 没有逐点真实标签，其参数通过最终 Rx 处 PL 的监督间接学习。

### 8.3 上下文依赖必须体现在论文符号中

严格来说：

$$
(\xi_{iqj},\mathbf z_{iqj})
=\mathcal F_{\Theta_{\mathrm{out}},\Omega}
(\mathbf u_{iq1},\ldots,\mathbf u_{iqj}),
$$

其中 $\Omega$ 表示固定储备池参数。

同一个空间点若由不同 Rx、不同射线前缀到达，可能具有不同 hidden state 和输出。因此，更准确的称呼是 **sequence-conditioned latent attenuation and features**。只写成完全不依赖历史的 $\sigma(\mathbf p,f)$，会遗漏当前 WESN 实现的核心性质。

## 9. Radiance MLP 的输入与输出

### 9.1 输入维度为 39

归一化 Tx：

$$
\overline{\mathbf t}=\mathbf t\oslash\mathbf e.
$$

对每个采样点拼接：

$$
\boxed{
\mathbf c_{iqj}=
[\mathbf z_{iqj},\overline{\mathbf t},\mathbf d_q,\eta(f)]
\in\mathbb R^{39}.
}
$$

| 部分 | 维度 |
| --- | ---: |
| WESN latent feature | 32 |
| 归一化 Tx xyz | 3 |
| 射线方向 xyz | 3 |
| 对数频率 | 1 |
| 总计 | 39 |

因此，Tx 的影响在此处显式进入神经网络；在 residual 模式中，Tx 还参与计算 FSPL。

### 9.2 MLP 方程

线性层宽度为：

$$
39\rightarrow32\rightarrow32\rightarrow2.
$$

前两层使用 SiLU：

$$
\operatorname{SiLU}(x)=x\operatorname{sigmoid}(x).
$$

采用列向量写出该 MLP：

$$
\mathbf a^{(1)}=\operatorname{SiLU}(\mathbf A_1\mathbf c+\mathbf b_1),
$$

$$
\mathbf a^{(2)}=\operatorname{SiLU}(\mathbf A_2\mathbf a^{(1)}+\mathbf b_2),
$$

$$
\begin{bmatrix}g_{iqj}\\v_{iqj}\end{bmatrix}
=\mathbf A_3\mathbf a^{(2)}+\mathbf b_3.
$$

这里仅为常见 MLP 公式改用列向量；实现仍遵循 PyTorch 的 `Linear` 约定。

### 9.3 log-amplitude 和 phase

两个输出解释为：

$$
A_{iqj}=\exp(g_{iqj}),
\qquad
\phi_{iqj}=\pi\tanh(v_{iqj}).
$$

因此：

- $g$ 是自然对数幅度；$A$ 才是正的幅度。
- $\phi\in(-\pi,\pi)$ 是可学习相位。
- 代码没有显式加入 $2\pi f d/c$ 的传播相位。
- 代码没有根据墙体材料计算反射系数，也没有输入天线方向图。

这些量用于构造可微的内部功率表示，不应直接称为已经标定的真实复信道或实测 multipath component。

## 10. 如何从逐点输出得到一条射线的复数贡献

### 10.1 积分步长使用归一化长度

代码使用：

$$
\delta_{iq}=\frac{\ell_{iq}}{DS}.
$$

注意，它不是以米表示的 $\ell_{iq}/S$，而是再除以场景尺度 $D$。因此 density $\sigma$ 作用于**归一化射线长度**，不能直接称为单位 dB/m 的实测衰减系数。

定义逐段 optical depth：

$$
\omega_{iqj}=\sigma_{iqj}\delta_{iq}.
$$

### 10.2 累积 transmittance

第 j 点之前的累积权重为：

$$
\boxed{
T_{iqj}=\exp\left(-\sum_{m=1}^{j-1}\omega_{iqm}\right).
}
$$

因此 $T_{iq1}=1$。求和只覆盖 j 之前的点，与源码中 `cumsum(...)[..., :-1]` 前面补零一致。

逐点 compositing weight 为：

$$
\boxed{
\alpha_{iqj}=1-\exp(-\omega_{iqj}).
}
$$

它们是受 NeRF 启发的体积合成量。当前计算顺序是沿 Rx 向外的采样顺序，不应将该累积过程直接解释成已经求解了从 Tx 到 Rx 的真实传播路径。

### 10.3 射线内的相干求和

一条射线的 latent complex response 为：

$$
\boxed{
E_{iq}=\sum_{j=1}^{S}
T_{iqj}\alpha_{iqj}A_{iqj}\exp(\mathrm i\phi_{iqj}).
}
$$

使用欧拉公式：

$$
\exp(\mathrm i\phi)=\cos\phi+\mathrm i\sin\phi,
$$

得到：

$$
\operatorname{Re}(E_{iq})
=\sum_j T_{iqj}\alpha_{iqj}A_{iqj}\cos\phi_{iqj},
$$

$$
\operatorname{Im}(E_{iq})
=\sum_j T_{iqj}\alpha_{iqj}A_{iqj}\sin\phi_{iqj}.
$$

这就是代码中需要计算 `real`、`imag` 的原因：不同采样点先按照其 learned phase 进行相干叠加，随后才求功率。

### 10.4 为什么与直接求幅度或功率之和不同

例如两个加权幅度都是 a 的项：

- 相位相同：复数和的幅度为 $2a$，功率为 $4a^2$。
- 相位相差 $\pi$：复数和可抵消，功率接近 0。

如果直接计算 $\sum_j A_j^2$，两种情况都会得到 $2a^2$，无法表达相消或相长。

这里说明的是**当前潜在表示的数学作用**。仅有 PL 标签，并不能保证网络学习到的相位就等于真实无线信道相位。

## 11. 如何将 8 条射线合成一个 Rx 预测

### 11.1 射线之间平均功率

先计算每条射线的功率：

$$
P_{iq}=|E_{iq}|^2
=\operatorname{Re}(E_{iq})^2+\operatorname{Im}(E_{iq})^2.
$$

再对方向平均：

$$
\boxed{
P_i^{\mathrm{latent}}=\frac1R\sum_{q=0}^{R-1}P_{iq}.
}
$$

所以当前实现是：

**射线内对 16 个采样点相干求和，射线之间对 8 个功率做非相干平均。**

不能把它改写成：

$$
\left|\sum_q E_{iq}\right|^2,
$$

因为那会引入不同方向之间的相位干涉项，改变当前模型。

也没有额外对 16 个采样点做一次 $1/S$ 平均；积分长度已进入 $\alpha$ 中。

### 11.2 负对数输出

定义网络的 dB score：

$$
\boxed{
s_\Theta(\mathbf r_i,\mathbf t,f)
=-10\log_{10}P_i^{\mathrm{latent}}
=-\frac{10}{\ln10}\ln P_i^{\mathrm{latent}}.
}
$$

其中 $\Theta$ 包括可训练的 WESN readout 和 radiance MLP 参数。

乘数 $10/\ln10$ 是自然对数到功率 dB 表示的数学换算，不是额外设置的 `output_scale_db`。

这是先平均线性功率，再取对数。通常不等于先求每条射线的 dB score 再做算术平均。

### 11.3 对 latent power 的解释范围

$P_i^{\mathrm{latent}}$ 是为拟合标量 PL 构造的潜在正量，当前没有单独把它标定成接收功率 W 或物理信道增益。

例如，把某条射线的全部复数贡献统一旋转一个相位，$|E_{iq}|^2$ 不变。因此仅凭最终 PL，无法唯一识别内部 phase。类似地，内部 density、amplitude 和 feature 也不能被视为已由 PL 唯一确定的物理传播参数。

论文中可以描述 differentiable power-based rendering 或 latent complex compositing；若声称恢复了真实 CSI、绝对相位或真实反射路径，需要额外的测量监督和验证。

## 12. 实际代码如何稳定计算上述公式

前面的公式给出模型的数学结构。本节解释 `_log_alpha`、`_log_power` 和 max shifting，便于逐行对照当前实现。

### 12.1 不直接计算可能溢出的绝对幅度

定义：

$$
w_{iqj}=\log T_{iqj}+\log\alpha_{iqj}+g_{iqj},
\qquad
m_{iq}=\max_j w_{iqj}.
$$

计算相对幅度：

$$
\widetilde A_{iqj}=\exp(w_{iqj}-m_{iq}).
$$

因为 $w_{iqj}-m_{iq}\le0$，不会由于直接计算很大的 $e^{g}$ 而溢出。

令：

$$
C_{iq}=\sum_j\widetilde A_{iqj}\cos\phi_{iqj},
\qquad
Q_{iq}=\sum_j\widetilde A_{iqj}\sin\phi_{iqj}.
$$

理想数学形式下：

$$
\log P_{iq}=2m_{iq}+\log(C_{iq}^2+Q_{iq}^2).
$$

再使用 log-sum-exp：

$$
\log P_i^{\mathrm{latent}}
=\operatorname{LSE}_q(\log P_{iq})-\log R.
$$

Max shifting 在精确算术下不改变前面定义的渲染表达式。

### 12.2 机器精度保护

实际代码在平方和上使用：

$$
\log\left(\max(C_{iq}^2+Q_{iq}^2,\epsilon_{\mathrm{tiny}})\right).
$$

这里：

$$
\epsilon_{\mathrm{tiny}}
=\operatorname{finfo}(\text{field dtype}).\mathrm{tiny}.
$$

普通 float32 下约为 $1.17549\times10^{-38}$。它保护的是经过 max shifting 的相干余量，不是人为设定一个固定的绝对接收功率下限。

`_log_alpha` 还对很小的数使用渐近表达式：

$$
\log\operatorname{softplus}(\xi)\approx\xi
\quad\text{当 }\xi<-20,
$$

$$
\log(1-e^{-\omega})\approx\log\omega
\quad\text{当 }\log\omega<-20.
$$

输出的 log-alpha 至少为 $\log\epsilon_{\mathrm{tiny}}$。这些是有限精度实现中的保护措施，极端情况下会与无限精度理想公式存在差别。

### 12.3 哪些部分使用 float64

当前实现中：

- 网络、储备池、相对幅度及 `real`、`imag` 的求和通常仍是 float32。
- 求和后的 real、imag 转为 float64，计算平方、平方和和 log。
- $2m$、射线 log-power 和跨射线 log-sum-exp 使用 float64。
- 最后的 PL score 转回 field dtype，通常是 float32。

所以准确的说法是“关键 log-power 计算使用双精度”，而非“整个 NeRF-WESN 都采用双精度”。

## 13. Direct 和 residual 两种模式的完整输出公式

### 13.1 Direct mode：直接拟合 PL

```yaml
prediction_mode: direct_pl
```

$$
\boxed{
\widehat y_i(f)=s_\Theta(\mathbf r_i,\mathbf t,f).
}
$$

这时 network score 就代表完整 PL dB。forward 不调用 FSPL。

### 13.2 Residual mode：FSPL 先验加可学习修正

```yaml
prediction_mode: fspl_residual
```

设真实 Tx-Rx 距离：

$$
d_i=\|\mathbf r_i-\mathbf t\|_2.
$$

自由空间路径损耗为：

$$
L_{\mathrm{FSPL},i}(f)
=20\log_{10}\left(\frac{4\pi d_i f_{\mathrm{Hz}}}{c_0}\right),
\qquad c_0=299\,792\,458\ \mathrm{m/s}.
$$

当 d 用 m、f 用 GHz 时：

$$
L_{\mathrm{FSPL},i}(f)
=20\log_{10}d_i+20\log_{10}f_{\mathrm{GHz}}+32.447783\ldots
$$

最终预测为：

$$
\boxed{
\widehat y_i(f)=L_{\mathrm{FSPL},i}(f)
+s_\Theta(\mathbf r_i,\mathbf t,f).
}
$$

此时将 $s_\Theta$ 解释为带符号的 dB residual：

$$
R_\Theta=s_\Theta.
$$

它可以为正，也可以为负。虽然 density 非负、amplitude 为正，但 latent power 可大于 1，因此 $-10\log_{10}P$ 仍然可以为负。代码没有强制 residual 必须增加 FSPL。

### 13.3 两个容易混淆的“衰减”

| 量 | 含义 | 范围/单位 |
| --- | --- | --- |
| $\sigma_{iqj}$ | 射线合成中的逐点 latent density | 非负，作用于归一化长度 |
| $R_\Theta$ | 最终加到 FSPL 上的修正 | 带符号的 dB 标量 |

WESN 逐点输出的 density 不是最终 Rx 处的 PL residual。中间还经过 latent feature、radiance MLP 和功率渲染。

### 13.4 距离和标签处理

FSPL 使用真实的正 Euclidean distance，不做 0.25 m 或 1 m 的距离截断。正好与 Tx 重合的 Rx 由数据加载器统一排除。

这与第 4 节的 0.01 m 射线长度处理属于两个不同变量。

两个模式都不对 PL 标签做 Min-Max，也不在输出上施加 sigmoid、PL clipping 或人工校准偏置。

## 14. 损失函数和训练到底更新哪些参数

### 14.1 真实监督始终是 Rx 处的一个 PL

对于一个训练 mini-batch $\mathcal B$：

$$
\mathcal L(\Theta)
=\frac1{|\mathcal B|}\sum_{i\in\mathcal B}
\left(\widehat y_i(f)-y_i(f)\right)^2.
$$

单位为 $\mathrm{dB}^2$。代码对应：

```python
prediction = model(rx[index], tx.expand(len(index), -1), frequency_ghz)
loss = (prediction - labels[index]).square().mean()
```

每个 Rx 只有一个最终监督值；16 个虚拟点没有各自的 PL 标签。

### 14.2 Direct 的目标

$$
\mathcal L_{\mathrm{direct}}
=\frac1{|\mathcal B|}\sum_i
\left(s_\Theta(\mathbf r_i,\mathbf t,f)-y_i(f)\right)^2.
$$

### 14.3 Residual 的目标

$$
\mathcal L_{\mathrm{residual}}
=\frac1{|\mathcal B|}\sum_i
\left(L_{\mathrm{FSPL},i}+s_\Theta-y_i\right)^2.
$$

等价地：

$$
\mathcal L_{\mathrm{residual}}
=\frac1{|\mathcal B|}\sum_i
\left(s_\Theta-\left[y_i-L_{\mathrm{FSPL},i}\right]\right)^2.
$$

第二个公式说明为什么可以称为学习 residual。不过实现是在 forward 中加 FSPL 后与原始 PL 标签比较，并没有先建立一个独立的、归一化后的 residual 数据集。

### 14.4 可训练参数的实际范围

记：

$$
\Omega=\{\mathbf W_{\mathrm{in}},\mathbf W_{\mathrm{res}},\mathbf b_{\mathrm{res}}\},
$$

$$
\Theta_{\mathrm{out}}=\{\mathbf W_{\mathrm{out}},\mathbf b_{\mathrm{out}}\},
\qquad
\Theta_{\mathrm{rad}}=\{\mathbf A_1,\mathbf b_1,\mathbf A_2,\mathbf b_2,\mathbf A_3,\mathbf b_3\}.
$$

训练时：

$$
\boxed{\Omega\ \text{固定},\qquad
\Theta=\{\Theta_{\mathrm{out}},\Theta_{\mathrm{rad}}\}\ \text{更新}.}
$$

**“只训练 output layer”只适用于 WESN attenuation 子模块。整个 NeRF-WESN 还训练 radiance MLP 的全部线性层。**

### 14.5 优化算法

当前使用 Adam，对 `requires_grad=True` 的参数优化。默认学习率为 $10^{-3}$，weight decay 为 $10^{-5}$，gradient clipping norm 为 5。

日志中的 loss 是上述 MSE；weight decay 由 Adam 处理。WESN 读出层本身是线性的，但后面还有 softplus、radiance MLP、指数、相干求和和 log。因此最终 PL 损失关于 readout 参数并非线性最小二乘问题。

**当前实现没有用 ridge regression、闭式解或一次性拟合 readout，而是通过最终 PL 损失进行梯度训练。**

由于储备池权重固定，且输入是给定的坐标特征，当前训练不会对循环权重进行更新。固定储备池仍需要执行递归前向计算，不能据此声称其前向计算成本为零。

## 15. 默认参数量：为什么不能说全模型只训练最后一层

默认结构：$C=28,H_r=32,L_w=4$。

| 模块 | 参数形状或计算方式 | 参数元素数 | 是否训练 |
| --- | --- | ---: | --- |
| Reservoir input matrix | $28\times32$ | 896 | 否 |
| Reservoir recurrent matrix | $32\times32$ | 1024 | 否 |
| Reservoir bias | $32$ | 32 | 否 |
| WESN readout | $33\times144+33$ | 4785 | 是 |
| Radiance 第 1 层 | $39\times32+32$ | 1280 | 是 |
| Radiance 第 2 层 | $32\times32+32$ | 1056 | 是 |
| Radiance 输出层 | $32\times2+2$ | 66 | 是 |
| 固定参数总数 | — | **1952** | 否 |
| 可训练参数总数 | — | **7187** | 是 |
| 全部参数元素数 | — | **9139** | — |

这里统计的是参数张量元素数；循环矩阵以稠密张量存储，即使部分元素被置零，也按 $32\times32$ 计数。方向、坐标尺度和 Fourier 尺度等 buffer 不计入可训练参数。

读出层和 radiance 中的 `nn.Linear` 使用 PyTorch 默认初始化；前面所说的 Xavier uniform 指储备池的输入矩阵和循环矩阵。

写文章时可以说：

> The fixed random reservoir reduces the number of trainable parameters in the attenuation module, while a linear readout and a downstream radiance MLP are optimized using the path-loss prediction loss.

能否获得更低误差、更快训练或更稳定优化，需要由实验结果说明。

## 16. Source、Scratch、Transfer 和 Zero-shot 中 sequence 有什么变化

### 16.1 数据和划分

当前为 10 scenarios、5 buildings。每个场景独立训练和测试，默认每个场景用 10% 有效 Rx 位置训练，其余位置测试，重复 seeds 0–9。

对一个场景，令训练位置集合为 $\mathcal A$，测试集合为 $\mathcal T$：

$$
\mathcal A\cap\mathcal T=\varnothing.
$$

源频段和目标频段的训练数据为：

$$
\mathcal D_s=\{(\mathbf r_i,\mathbf t,f_s,y_i(f_s)):i\in\mathcal A\},
$$

$$
\mathcal D_t=\{(\mathbf r_i,\mathbf t,f_t,y_i(f_t)):i\in\mathcal A\},
$$

其中 $f_s=1.8\ \mathrm{GHz}$，$f_t=3.5\ \mathrm{GHz}$。

同一个 Rx、同一条方向在两个频段中生成的几何点 $\mathbf p_{iqj}$ 相同，但输入 $\eta(f)$ 和监督 PL 不同。

### 16.2 一条 sequence 中的频率不变化

源频段预训练的某条序列为：

$$
[\gamma(\overline{\mathbf p}_{iq1}),\eta(f_s)],\ldots,
[\gamma(\overline{\mathbf p}_{iqS}),\eta(f_s)].
$$

目标频段训练的对应序列为：

$$
[\gamma(\overline{\mathbf p}_{iq1}),\eta(f_t)],\ldots,
[\gamma(\overline{\mathbf p}_{iqS}),\eta(f_t)].
$$

f2 和 f3 不会交替填入同一条序列。

### 16.3 参数如何迁移

记 `Train` 表示指定步数的 Adam 训练过程。它不表示已经求到了全局最优解。

源预训练：

$$
\Theta_s=\operatorname{Train}(\Theta_0;\Omega,\mathcal D_s,K_s).
$$

Scratch：

$$
\Theta_{\mathrm{scratch}}
=\operatorname{Train}(\Theta_0;\Omega,\mathcal D_t,K_t).
$$

Transfer：

$$
\Theta_{\mathrm{transfer}}
=\operatorname{Train}(\Theta_s;\Omega,\mathcal D_t,K_t).
$$

Zero-shot：直接使用 $\Theta_s$，以 $f_t$ 为条件进行预测，不做目标频段优化。

实际 `state_dict` 会复制整个模型，包括固定储备池、已训练的 readout 和 radiance MLP。微调时 readout、radiance MLP 都继续训练，储备池继续固定。每个训练阶段新建 Adam optimizer，不继承源阶段 optimizer 的动量。

同一场景和 seed 的 source 模型、Scratch 模型使用匹配的随机初始化。不同场景分别训练，没有通过当前 runner 建立一个跨场景共享的预训练模型。

### 16.4 训练步数与 sequence length 是不同概念

如果运行命令指定：

```text
pretrain_steps = 30000
target_steps = 30000
```

那么 Scratch 进行 30000 个 target optimizer steps；Transfer 使用 30000 个 source steps 和 30000 个 target steps；Zero-shot 使用 source 阶段得到的模型。

无论训练多少 steps，每次 forward 中的一条 WESN sequence 默认仍只有 16 个空间点，窗口长度仍为 4。

### 16.5 最终评估

所有方案在同一批目标频段测试坐标 $\mathcal T$ 上预测 f3。源频段测试位置的 PL 不参与源预训练，当前程序也不单独计算源频段测试图的重建误差。

指标以真实 dB 计算：

$$
\operatorname{RMSE}
=\sqrt{\frac1{|\mathcal T|}\sum_{i\in\mathcal T}
(\widehat y_i-y_i)^2},
$$

$$
\operatorname{MAE}
=\frac1{|\mathcal T|}\sum_{i\in\mathcal T}|\widehat y_i-y_i|.
$$

Transfer 的报告训练时间包括 source pretraining 和 target fine-tuning；Zero-shot 的训练时间包括 source pretraining。

## 17. 完整张量维度表

以下使用默认 $B=128,R=8,S=16,C=28,H_r=32,L_w=4$。实际最后一个 batch 的 B 可以更小。

| 步骤 | 一般形状 | 默认示例 |
| --- | --- | --- |
| 查询 Rx | $[B,3]$ | `[128,3]` |
| 广播后的 Tx | $[B,3]$ | `[128,3]` |
| 射线方向 | $[B,R,3]$ | `[128,8,3]` |
| 射线长度 | $[B,R]$ | `[128,8]` |
| 射线采样位置 | $[B,R,S,3]$ | `[128,8,16,3]` |
| 位置编码 | $[B,R,S,27]$ | `[128,8,16,27]` |
| 广播后的频率特征 | $[B,R,S,1]$ | `[128,8,16,1]` |
| 拼接输入 | $[B,R,S,28]$ | `[128,8,16,28]` |
| WESN 输入 | $[BR,S,28]$ | `[1024,16,28]` |
| Reservoir states | $[BR,S,32]$ | `[1024,16,32]` |
| States 加窗口 | $[BR,S,144]$ | `[1024,16,144]` |
| Readout 输出 | $[BR,S,33]$ | `[1024,16,33]` |
| 恢复 Rx/ray 轴 | $[B,R,S,33]$ | `[128,8,16,33]` |
| Density | $[B,R,S]$ | `[128,8,16]` |
| Latent feature | $[B,R,S,32]$ | `[128,8,16,32]` |
| Radiance 输入 | $[B,R,S,39]$ | `[128,8,16,39]` |
| Radiance 输出 | $[B,R,S,2]$ | `[128,8,16,2]` |
| 每条射线的 real、imag | 各 $[B,R]$ | 各 `[128,8]` |
| 每条射线的 log-power | $[B,R]$ | `[128,8]` |
| 合并方向后的 log-power | $[B]$ | `[128]` |
| 完整 PL 预测 | $[B]$ | `[128]` |
| 训练标签 | $[B]$ | `[128]` |

注意：最终 PL 的形状是 `[B]`，不是 `[B,R,S]`，更不是一整张 PL 图。一张图由多个 Rx 查询的预测按原网格位置排列得到。

## 18. 一个真实尺寸的具体例子：Scene02 的一条 sequence

### 18.1 场景与查询

当前 Scene02 的图像宽、高为：

$$
W_{\mathrm{px}}=99,\qquad H_{\mathrm{px}}=87.
$$

因此：

$$
L_x=24.5\ \mathrm m,\quad
L_y=21.5\ \mathrm m,\quad D=24.5\ \mathrm m.
$$

Tx 为：

$$
\mathbf t=(22.0,1.75,1.5)\ \mathrm m.
$$

选一个合法网格查询点作为例子：

$$
\mathbf r=(10.0,10.0,1.5)\ \mathrm m,
\qquad f=3.5\ \mathrm{GHz}.
$$

考察向正 x 的射线 $\mathbf d_0=(1,0,0)$：

$$
\ell_0=24.5-10=14.5\ \mathrm m.
$$

物理采样间隔：

$$
\Delta s^{\mathrm{physical}}=14.5/16=0.90625\ \mathrm m.
$$

渲染使用的归一化步长：

$$
\delta_0=14.5/(24.5\times16)\approx0.036989796.
$$

### 18.2 16 个空间点

所有点的 y、z 都为 10.0、1.5。x 坐标依次为：

```text
j= 1: 10.453125
j= 2: 11.359375
j= 3: 12.265625
j= 4: 13.171875
j= 5: 14.078125
j= 6: 14.984375
j= 7: 15.890625
j= 8: 16.796875
j= 9: 17.703125
j=10: 18.609375
j=11: 19.515625
j=12: 20.421875
j=13: 21.328125
j=14: 22.234375
j=15: 23.140625
j=16: 24.046875
```

这些点无须恰好落在原始 0.25 m 像素中心，也不需要有实测 PL。

### 18.3 第一个点的输入

$$
\overline{\mathbf p}_1
=\left(\frac{10.453125}{24.5},\frac{10}{21.5},\frac{1.5}{24.5}\right)
\approx(0.42665816,0.46511628,0.06122449).
$$

$$
\mathbf u_1=
[\gamma(0.42665816,0.46511628,0.06122449),\ 1.25276297]
\in\mathbb R^{28}.
$$

第 2 到第 16 个向量以相同方式构造。此时 frequency feature 始终为 1.25276297。

切换为源频段 1.8 GHz 时，这 16 个点的位置保持不变，最后一个频率特征统一改成 0.58778666；随后储备池状态也会随输入变化。

### 18.4 第四个点如何使用前面的点

处理第 4 个点时：

$$
\mathbf h_4=\tanh(\mathbf u_4\mathbf W_{\mathrm{in}}+
\mathbf h_3\mathbf W_{\mathrm{res}}+\mathbf b_{\mathrm{res}}),
$$

$$
\mathbf v_4=[\mathbf h_4,\mathbf u_4,\mathbf u_3,\mathbf u_2,\mathbf u_1]
\in\mathbb R^{144}.
$$

这就是“点相关性”进入当前模型的位置：前面点通过递归状态和窗口共同影响当前点的 latent 输出。

这四个显式窗口点之间的首末距离为 $3\times0.90625=2.71875\ \mathrm m$，不是固定的 $3\times0.25\ \mathrm m$。不同射线的物理窗口跨度会变化。

## 19. 接近源码的序列构造片段

下面片段用于阅读当前内部接口，不包含训练循环，也不需要采样点的真实 PL：

```python
import torch
from propagation import torch_frequency_feature

# model 是已经构造好的 RadioMapField。
# rx: [B,3]；tx: [B,3]；frequency_ghz 是标量或每个 Rx 的频率。
B = len(rx)
R, S = model.num_rays, model.num_samples

# 1. Rx-centered rays，采样点顺序由近到远。
points, directions, lengths = model._rays(rx)  # [B,R,S,3]

# 2. 编码位置并广播频率。
normalized_points = points / model.extent_m
position_features = model._encode(normalized_points)  # [B,R,S,27]
frequency_feature = torch_frequency_feature(frequency_ghz, rx)
frequency_feature = frequency_feature.view(B, 1, 1, 1).expand(B, R, S, 1)

# 3. 每个 Rx-ray 是一条独立 sequence。
sequence = torch.cat((position_features, frequency_feature), dim=-1)
sequence = sequence.reshape(B * R, S, -1)  # [BR,S,28]

# 4. WESN 内部从 h0=0 开始，逐点递归并拼接因果窗口。
field = model.attenuation(sequence)         # [BR,S,33]
field = field.reshape(B, R, S, -1)

# 5. WESN 输出拆分。
density = torch.nn.functional.softplus(field[..., 0])
latent_features = field[..., 1:]           # [B,R,S,32]

# 6. Tx、方向和频率条件化 radiance。
tx_features = (tx / model.extent_m).view(B, 1, 1, 3).expand(B, R, S, 3)
direction_features = directions.unsqueeze(-2).expand(B, R, S, 3)
radiance_input = torch.cat(
    (latent_features, tx_features, direction_features, frequency_feature), dim=-1
)
radiance_output = model.radiance(radiance_input)  # [B,R,S,2]
log_amplitude = radiance_output[..., 0]
phase = torch.pi * torch.tanh(radiance_output[..., 1])

# 后续按第10–13节进行稳定渲染，返回 [B] 的 PL。
# 实际调用直接使用 model(rx, tx, frequency_ghz)。
```

这个片段在 28/27/33/32 的注释上使用当前默认值；更改 `num_freq` 或 `hidden_dim` 后，维度相应变化。

## 20. 可以改写进英文论文的方法描述

以下内容描述当前实现。符号与前文一致，投稿时可根据模板篇幅删减。

### 20.1 Query representation and spatial sequence construction

For each receiver query $\mathbf r_i$, the model predicts scalar path loss conditioned on the transmitter position $\mathbf t$ and carrier frequency $f$. We construct $R$ receiver-centered rays with uniformly spaced azimuth angles in the horizontal plane. Each ray is discretized into $S$ midpoint samples ordered from the receiver toward the rectangular scene boundary. The spatial samples are given by

$$
\mathbf p_{iqj}=\mathbf r_i+
\frac{j-1/2}{S}\ell_{iq}\mathbf d_q.
$$

The normalized sample coordinates are mapped to Fourier features and concatenated with the log-frequency feature:

$$
\mathbf u_{iqj}=
[\gamma(\mathbf p_{iqj}\oslash\mathbf e),\ln(f/f_0)].
$$

Each receiver-ray pair forms an independent spatial sequence. These samples are virtual query locations and do not require path-loss measurements. In our default configuration, $R=8$, $S=16$, and each sequence element has 28 features.

### 20.2 Windowed reservoir attenuation module

A fixed random reservoir processes the ordered spatial inputs according to

$$
\mathbf h_j=(1-\lambda)\mathbf h_{j-1}
+\lambda\tanh(\mathbf u_j\mathbf W_{\mathrm{in}}
+\mathbf h_{j-1}\mathbf W_{\mathrm{res}}+\mathbf b_{\mathrm{res}}),
\qquad\mathbf h_0=\mathbf0.
$$

To preserve recent inputs explicitly, the reservoir state is augmented with a causal input window:

$$
\mathbf v_j=[\mathbf h_j,\mathbf u_j,\ldots,\mathbf u_{j-L_w+1}].
$$

Unavailable historical inputs are zero-padded. A trainable linear readout maps $\mathbf v_j$ to a raw density value and a latent feature vector. The input and recurrent reservoir weights remain fixed throughout source pretraining and target fine-tuning. The recurrent state and input window provide complementary representations of dependencies among ordered samples along each ray.

### 20.3 Conditional radiance and latent power rendering

The density is constrained to be nonnegative using softplus. The latent feature is concatenated with the normalized transmitter coordinates, ray direction, and log-frequency feature, and passed to a trainable radiance MLP. Its two outputs represent log-amplitude and a bounded learned phase. With normalized integration increment $\delta_{iq}=\ell_{iq}/(DS)$, we define

$$
T_{iqj}=\exp\left(-\sum_{m<j}\sigma_{iqm}\delta_{iq}\right),
\qquad
\alpha_{iqj}=1-\exp(-\sigma_{iqj}\delta_{iq}).
$$

Samples are combined coherently within each ray, followed by an average of ray powers:

$$
E_{iq}=\sum_j T_{iqj}\alpha_{iqj}\exp(g_{iqj})\exp(\mathrm i\phi_{iqj}),
\qquad
P_i^{\mathrm{latent}}=\frac1R\sum_q|E_{iq}|^2.
$$

The density, amplitude, and phase serve as latent variables for scalar path-loss regression; they are not independently calibrated channel measurements. Because the attenuation module is recurrent, its per-sample outputs depend on the preceding ray samples as well as the current position and frequency.

### 20.4 Path-loss prediction and optimization

The renderer produces $s_\Theta=-10\log_{10}P_i^{\mathrm{latent}}$. We study direct prediction,

$$
\widehat y_i=s_\Theta,
$$

and prediction with an analytical free-space prior,

$$
\widehat y_i=L_{\mathrm{FSPL},i}+s_\Theta.
$$

Both variants minimize mean squared error against the original path-loss labels in dB. Only the linear readout is trained within the WESN attenuation module, while the downstream radiance MLP is optimized jointly. The default architecture contains 7,187 trainable parameters and 1,952 fixed reservoir parameters. Training uses Adam through the final path-loss loss rather than a closed-form readout solution.

### 20.5 Cross-frequency adaptation

For each scene and seed, the source and target training labels are selected at the same receiver locations. The source model is pretrained at 1.8 GHz and then copied for target fine-tuning at 3.5 GHz. Both the WESN readout and the radiance MLP are updated during fine-tuning, while the reservoir remains fixed. A separate copy of the pretrained model is evaluated at the target frequency without target fine-tuning for the zero-shot experiment. The hidden state is reset for each ray sequence; cross-frequency transfer is carried by the model parameters, not by a persistent recurrent state.

The target-only model, transferred model, and zero-shot model are evaluated on the same held-out target-band receiver locations. Transfer additionally uses source-band labels and pretraining computation.

## 21. 写论文时需要保持准确的几个重点

| 想表达的内容 | 与当前实现一致的表述 |
| --- | --- |
| 使用 sequence | 沿每条 Rx-centered ray 的有序虚拟空间样本序列 |
| 建模点相关性 | 通过储备池递归与因果输入窗口表达沿射线的空间依赖 |
| 随机初始化 | 储备池输入、循环权重随机初始化并固定；readout 和 radiance 权重也初始化，但会训练 |
| 只训练 output layer | 限定为 WESN 子模块；整个模型还训练 radiance MLP |
| attenuation field | 带射线前缀上下文的 latent density/features，尚非唯一标定的物理衰减场 |
| 输入为坐标和频率 | 外部输入确为 Rx/Tx/f；内部还构造几何采样、固定方向、编码和窗口 |
| 三维坐标 | 接受 xyz，但当前射线在固定高度的水平面采样 |
| FSPL-informed | residual 模式显式加入 FSPL；direct 模式没有这项解析加法 |
| 训练 residual | 等价于拟合 PL 减 FSPL，实际在 forward 中加回 FSPL 后与原始 PL 求损失 |
| 跨频 transfer | 迁移网络参数，目标阶段更新 readout 和 radiance；不跨频传递 hidden state |
| 相位建模 | 内部 learned phase 参与相干合成；没有真实 CSI/phase 监督 |
| 训练效率 | 固定储备池减少需要优化的参数；速度优势和精度优势仍需实测 |

此外，当前代码没有显式的点对相关性损失、attention 权重或 Gudmundson 协方差约束。若论文使用这些术语，需要补充相应实现或将其明确限定为背景，而不是本模型已经执行的计算。

当前实验按场景独立训练，每个场景 Tx 固定。虽然模型接受 Tx 条件输入，但这不足以证明已经实现对未见 Tx 位置的泛化。

## 22. 本文公式的核验与推荐使用方式

本文的默认维度和参数量已通过实际模型实例核对：

```text
WESN input:             [BR,16,28]
Reservoir hidden:       32
Window length:          4
Readout:                Linear(144,33)
Radiance:               39 -> 32 -> 32 -> 2
Trainable parameters:   7187
Fixed parameters:       1952
```

对第 18 节的 Scene02 示例，还进行了两项只读数值核对：

1. 手动按状态递归、窗口拼接和线性读出计算的结果，与 `WESN.forward` 的最大绝对差为 0。
2. 在普通、未出现数值极端的初始化例子中，直接按第 10–11 节公式计算的 score 与稳定实现相差约 $9.54\times10^{-7}$ dB。此数值只用于检查公式对应关系，不是预测精度实验结果。

推荐写作安排：

- **System Model**：使用第 3 节及第 16.1 节，定义坐标、频段、标签和训练/测试集合。
- **Proposed Model**：使用第 4–11 节，重点说明射线如何形成 sequence、WESN 的两种记忆以及最终 PL 渲染。
- **Learning and Transfer**：使用第 13–16 节，写清损失、冻结范围和跨频阶段。
- **Implementation Details**：列出 R、S、Fourier 尺度、H、window、谱半径和训练参数；数值稳定处理可简写或放附录。
- **Experiments**：精度、时间、参数量比较和消融结果应使用重新训练后的实际结果。关于窗口、储备池、FSPL 和 transfer 的收益，需要分别由对应实验支持。

如果只用一句话概括当前 NeRF-WESN：

> 对每个 Rx，在周围构造多条有序空间采样序列，用固定随机储备池与短窗口生成上下文相关的逐点特征，再经 Tx/频率条件化的 radiance MLP 和可微功率渲染，得到直接 PL 或加到 FSPL 上的 dB 修正。
