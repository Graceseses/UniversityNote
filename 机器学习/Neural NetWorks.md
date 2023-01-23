# <font color ='red'>**基本概念**</font>
### <font color = 'lightgreen'>**神经元模型**</font>
   - 简称"简单单元"
   - 神经网络中的 **最基本成分**
### <font color = 'lightgreen'>**神经网络**</font>
**定义**：
$$
是\left\{\begin{aligned}
&具有适应性の\\
&简单单元~组成の\\
&广泛~并行互联の\\
\end{aligned}\right.~网络
$$
**作用**：
   - 模拟 **生物神经系统** $\Longrightarrow$ 对 真实世界物体 所做出的反应
### <font color = 'lightgreen'>**兴奋**</font>
当神经元兴奋，将会向其他**相连神经元**发送"化学物质"(信息)
   - 0：神经元 **抑制**
   - 1：神经元 **兴奋**
### <font color = 'lightgreen'>**阈值**</font>
使神经元兴奋的**最小**"电位"(下限值)

### <font color = 'lightgreen'>**MP神经元**</font><font size =2>   **阈值逻辑单元**</font>
**模型构成**：
   - 输入：来自n个其他神经元传递过来的输入信号 $x_i$
   - 连接：n个输入信号通过n个各自对应的权重 $w_i$ 进行连接传递
   - 输出：$y=f(~\underset{i=1}{\stackrel{n}{\sum}}~w_i x_i - \theta~)$ + **激活函数**处理

### <font color = 'lightgreen'>**隐藏层**  *hidden layer*</font>
**定义**：位于输入层与输出层之间的一层神经元

### <font color = 'lightgreen'>**线性可分**</font>
若两类模型线性可分----> 则**一定存在**一个**线性超平面** 可以将它们完全分开
### <font color = 'lightgreen'>**激活函数**</font>
- **阶跃函数**：
   - 优点：
      - **理想** の激活函数$\left\{\begin{aligned}~0~~:抑制神经元\\~1: ~~激活神经元\end{aligned}\right.$
   - 缺点：
      - 不连续
      - 不光滑
- **Sigmoid**函数：
   - 优点 ：
      - 连续
      - 光滑
      - 常用の
# <font color ='red'>**感知机**</font><font size = 3> *Perceptron*</font>
- <font color='green'>**模型构成**：</font>：
   - 两层神经元
   - 输入层：接受外部信号
   - 输出层：M-P神经元

- <font color='green'> **应用**：</font>：实现基本**逻辑运算**
   - **与**$x_1\wedge x_2$：$$令 w_1=w_2=1,\theta=2\\ y = f(1·x_1+1·x_2-2)$$
   - **或**$x_1\vee x_2$：$$令 w_1=w_2=1,\theta=0.5\\ y = f(1·x_1+1·x_2-0.5)$$
   - **非**$\urcorner x_1$：$$令 w_1=-0.6,w_2=0,\theta=-0.5\\ y = f(-0.6·x_1+0·x_2+0.5)$$
- <font color='green'> **参数处理**</font> ：

<font size =3><font color = 'lightgreen'>**梯度下降法**</font></font>
迭代更新参数：
$$
w_i\leftarrow w_i +\bigtriangleup w_i\\
\\
\bigtriangleup w_i\leftarrow\eta(y-\tilde{y})x_i\\
$$
其中：$\eta = 学习率 （超参数）$

- <font color='green'> **应用**：</font>：处理**线性可分**的问题
- <font color='green'> **局限**</font> ：只含有一层 **功能神经元**，无法处理 **非线性可分** 的问题$$\Downarrow$$
   - 单个感知机 无法实现 **异或** 逻辑运算  $\Longrightarrow$ 通常使用2个感知机$$\Downarrow$$
- <font color='green'>**原理**</font>：
   - 模型 **线性可分**$\longrightarrow$ 感知机の 学习过程一定 **收敛**
   - 模型 **非线性可分**$\longrightarrow$ ...   学习过程 **振荡**$$\Downarrow$$
- <font color='green'>**解决方案**</font>：<font size =4><font color = 'lightgreen'>**多层感知机**</font></font>
   - <font color='green'>**特点**</font>：
      - **隐含层**+**输出层** = 具有 激活函数の **功能神经元**
# <font color ='red'>**多层前馈神经网络**</font><font size = 3>*Multi-layer feedforward neural networks*</font>
- <font color='green'> **定义**</font> ：
   - **前馈**：  
      - 输入层 ： 仅接受输入
      - 多层隐藏层 ：对信号进行加工
      - 输出层 ： 输出最终结果
   - **学习**：
      -  训练调整 神经元之间の **连接权** $w_ij$
      -  训练调整 每个 **功能神经元** の**阈值** $\theta_i$
- <font color='green'> **特点** </font> ：
   - 包含 **多层隐藏层**
   - 每层神经元 与 下一层神经元 **全互连**
   - 不存在**同层连接**
   - 不存在**跨层连接**

- <font color='green'> **学习算法**</font>  ：学习神经元之间连接的 **连接权** 与各个神经元的 **阈值** 
## <font color = 'lightgreen'>**误差逆传播算法**  *(标准/累积)BP算法*</font><font size = 3>back-propagation</font>
- 标准BP & 累积BP 的 **区别**：

||标准BP|累积BP|
|:-----:|:-----:|:-----:|
|参数更新|每次只针对 **单个样例**|读取完全部数据( 整个D )之后才进行更新|
|适用|数据集D很大|训练前期|
|

- <font color='green'>**损失函数**</font>：

<font size=4><font color = 'lightgreen'>**基于累计误差最小化**</font></font>
$$
\begin{align*}
&样例(x_k,y_k)的误差：\\
&\qquad \qquad \qquad E_k = \frac{1}{2}\underset{j=1}{\stackrel{l}{\sum}}~(\tilde{y_j^k}-y_j^k)^2\\
&神经网络整体误差：\\
&\qquad \qquad \qquad E = \frac{1}{m}\underset{k=1}{\stackrel{m}{\sum}}~E_k
\end{align*}
$$

- <font color='green'>**参数**</font>：
   - **权值**：|输出层|X|隐藏层| + |隐藏层|X|输出层|
   - **阈值**：|隐藏神经元|+|输出层神经元|

- <font color='green'>**学习策略**</font>：
<font size=4><font color = 'lightgreen'>**梯度下降算法**</font></font>

**定义**：以目标的负梯度方向对参数进行调整

任意**参数更新**方式为：
$$
v\leftarrow v+\triangle v
$$
**隐藏层-输出层** **权重**$w_{hj}$:
$$
\begin{align*}
\triangle w_{hj}&=-\eta\frac{\partial E_k}{\partial w_{hj}}\\
&=\eta~·~(-~\frac{\partial E_k}{\partial \tilde{y_j^k}}~·~\frac{\partial \tilde{y_j^k}}{\partial \beta_j})~(\frac{\partial \beta_j}{\partial w_{hj}})   \\
&= \eta~·~g_j~·~b_h 
\end{align*}
$$
**输入层-隐藏层** **权重**$v_{ih}$:
$$
\begin{align*}
\triangle v_{ih}&=\eta~·~e_h~·~x_i    
\end{align*}
$$
**输出层** **阈值**$\theta_j$:
$$
\begin{align*}
\triangle\theta_j = -\eta~·~g_j
\end{align*}
$$
**隐藏层** **阈值**$\gamma_h$:
$$
\begin{align*}
\triangle \gamma_h~=~-\eta~·~e_h
\end{align*}
$$
**梯度项**:
$$
\begin{align*}
g_i &=-~\frac{\partial E_k}{\partial \tilde{y_j^k}}~·~\frac{\partial \tilde{y_j^k}}{\partial \beta_j}\\
&=-~(\tilde{y_j^k}-y_j^k)f'(\beta_j - \theta_j)\\
&=\tilde{y_j^k}·(1-\tilde{y_j^k})·(y_j^k - \tilde{y_j^k})\\
&\\
e_h &= -~\frac{\partial E_k}{\partial b_h}~·~\frac{\partial b_h}{\partial\alpha_h}\\
&==\underset{j=1}{\stackrel{l}{\sum}}~\frac{\partial E_k}{\partial \beta_j}·\frac{\partial \beta_j}{\partial b_h}~·f'(\alpha_h - \gamma_h)\\
&=\underset{j=1}{\stackrel{l}{\sum}}~w_{hj} g_j~·f'(\alpha_h - \gamma_h)\\
&=b_h(1-b_h)\underset{j=1}{\stackrel{l}{\sum}}w_{hj}g_j
\end{align*}
$$

- <font color='green'>**算法流程**</font>：
   - Step 1：<font color='green'>**初始化参数**</font>：$$在(0,1)中随机初始化网络中的~所有~连接权和阈值$$
   - Step 2：<font color='green'>**迭代更新参数**</font>：$$\begin{align*}&<1. 计算当前 连接权+阈值下的样本输出\tilde{y_k}\\&<2.计算梯度项g_j,e_h\\&<3.根据损失函数\rightarrow更新"权值+阈值"\\&<4.达到停止条件return\\\end{align*}$$
   - Step 3：<font color='green'>获得**阈值+权值确定的多层前馈神经网络**</font>

- <font color='green'>**设置参数个数**</font>：
“”如何设置隐藏神经元的个数？”  $\Longrightarrow$  <font size=4><font color = 'lightgreen'>**试错法**</font></font>

# <font color ='red'>**过拟合**</font><font size = 3></font>
- “如何缓解BP网络的过拟合？”

<font size=4><font color = 'lightgreen'>**早停**</font></font>

   - 数据划分为 **训练集**+**验证集**
   - **训练集**：计算梯度，更新参数
   - **验证集**：估计误差
   - **早停条件**：train上误差降低，valid上误差升高


<font size=4><font color = 'lightgreen'>**正则化**</font></font>
 
   - **正则项**：描述**网络复杂度**的部分：
$$
E = \lambda\frac{1}{m}\underset{k=1}{\stackrel{m}{\sum}}~E_k~+~(1-\lambda)\underset{i}{w_i^2}\qquad,\qquad\lambda\in(0,1)
$$
<font size=4><font color = 'lightgreen'>**Dropout**</font></font>

- <font color='green'>**定义**</font>：
   - 通过定义の **概率** $\longrightarrow$ **随机删除**一些神经元 
   - 保持 输入层与输出层 の个人不变
- <font color='green'>**意义**</font>：
   - 减少神经元之间の **依赖性**
   - 随即删除 $\longrightarrow$ 增加 **模型数量**+**集成**+**共享参数**
   - **随机性**：记忆 随机 抹去
  

# <font color ='red'>**局部极小值**</font><font size = 3></font>
- 如何跳出 **局部最小** 到达 **全局最小**？

<font size=4><font color = 'lightgreen'>**多组参数初始化**</font></font>

- 使用多组 不同的参数 初始化多个神经网络
- 从多个结果中选择最小的（也可能都是局部最小）

<font size=4><font color = 'lightgreen'>**模拟退火**</font></font>*simulated annealing*

- 接受 **次优解** ---> 有助于跳出局部最优

<font size=4><font color = 'lightgreen'>**随机梯度下降**</font></font>

- 计算 **梯度项**时**加入随机项**

<font size=4><font color = 'lightgreen'>**遗传算法**</font></font>*genetic algorithms*

# <font color ='red'>**深度学习**</font><font size = 3> *Deep Learning*</font>
- <font color='green'>**特点**</font>：
   - **很深层**的神经网络
   - **多隐层**神经网络

<font size=4><font color = 'lightgreen'>**卷积神经网络** *CNN*</font></font>

- <font color='green'>**基本组成**</font>：
   - **卷积层**： 包含多个 **特征映射** ，输入信号の加工
   - **采样层**： 输入信号の加工
   - **连接层**： 与输出目标之间の映射
   - **采样层**(汇合层)：基于**局部相关性原理**进行**亚采样**

-----
**其余常见的神经网络**：
# <font color ='red'>**RBF网络**</font><font size = 3> 径向基函数网络    *Radial Basis Function*</font>
- <font color='green'>**定义**</font>：
   - **单隐层** 的**前馈**神经网络
   - 以 **径向基函数**作为**激活函数**
   - 输出 = 隐藏层神经元输出的线性组合

- <font color='green'>**训练过程**</font>：
   - 确定神经元中心$c_i$ (**随机采样**/**聚类**/....)
   - BP算法确定网络参数（权值 $w_i$ + 阈值 $\beta_i$）


<font size=4><font color = 'lightgreen'>**无监督逐层训练**</font></font>

基本思想：
   - **预训练**：上一层隐结点的输出作为下一层隐结点的输入
   - **微调**：预选连全部后在进行微调训练



<font size=4><font color = 'lightgreen'>**高斯径向基函数**</font></font>
$$
\begin{align*}
&\rho(x,c_i) = e^{\beta_i||x-c_i||^2}\\
&\\
&其中：\\
&\qquad x:样本\\
&\qquad c_i:数据中心    
\end{align*}
$$

<font size=4><font color = 'lightgreen'>**RBF网络**</font></font>

$$
\varphi(x) = \underset{i=1}{\stackrel{q}{\sum}}~w_i~\rho(x,c_i)
$$
**应用**：CNN（卷积神经网络）
# <font color ='red'>**卷积神经网络** *CNN*</font><font size = 3></font>
- <font color='green'>**基本思想**</font>：<font size=4><font color = 'lightgreen'>**权共享**</font></font>
   - 一组神经元**共享**相同的**连接权**
- <font color='green'>**优点**</font>：
   - 大幅减少了 需要训练 の **参数数目**
- <font color='green'>**组成**</font>：
   - 输入层
   - <font color = 'lightgreen'>**卷积层**</font>：特征映射 $\Longrightarrow$ 卷积滤波器进行 **特征提取**$$\begin{align*}&作用：\\&\qquad\qquad特征提取\end{align*}\\ \begin{align*}&特点：\\&\qquad\qquad稀疏交互\\&\qquad\qquad参数共享\\&\qquad\qquad等变表示\\ \end{align*}$$
   - <font color = 'lightgreen'>**池化层**</font>---->**采样层**：（汇合层）基于"**局部相关性**"原理进行 亚采样 $\Longrightarrow$ 减少数据量+保留有用信息$$\begin{align*}&作用:\\&\qquad <1>.增大~感受野\\&\qquad <2>.局部平移不变性\\&\qquad <3>.处理不同大小输入\end{align*}$$
   - <font color = 'lightgreen'> **连接层**</font>：（隐藏层）传统神经元连接层 $\Longrightarrow$ 连接完成识别任务
   - <font color = 'lightgreen'>**激活层**</font>：对 **卷积层**の输出做一次 **非线性映射**$$\begin{align*}&常用激活函数：\\&\\&\qquad Sigmoid函数~:~\\&\qquad\qquad sigmoid(x)=\frac{1}{1+e^{-x}}\\&\\&\qquad Tanh函数~:~\\&\qquad\qquad Tanh(x)=\frac{sinh(x)}{cosh(x)}=\frac{e^x~-~e^{-x}}{e^x~+~e^{-x}}\\&\\&\qquad ReLU函数~:~\\&\qquad\qquad ReLU(x)=max(0,x)\\&\\&\qquad Softmax函数：~:~\\&\qquad\qquad \sigma(z)_j~=~\frac{e^z~j}{\sum_{k=1}^k~e^zk}\\&\\&\qquad LeakyReLU函数\\&\qquad...\end{align*}$$
   - <font color = 'lightgreen'>**全连接层**</font> $$\begin{align*}&内容:\\&\qquad 全连接层中の~每个神经元\longrightarrow 与前一层の所有神经元~进行~"全连接"\\&\\&作用~：\\&\qquad~整合~"卷积层"/"池化层"具有~"类别区分性"の~局部信息\end{align*}$$
   - 输出层

- <font color='green'>**激活函数**</font>：**修正的线性函数**
$$
f(x)~=~max(0,x)
$$
# <font color ='red'>**ART网络**</font><font size = 3>自适应谐振理论   *Adaptive Resonance Theory*</font>
   - **无监督** 学习
   - **胜者通吃** 原则：**每一时刻**仅有**一个**竞争胜利的神经元被**激活**，其他神经元状态被抑制
   - **竞争型**学习的重要代表

- <font color='green'>**组成**</font>：
    - 比较层：接受输入样本 + 传递给识别层
    - 识别层：每个**神经元**对应一个**模式类**
    - 识别阈值：
    - 重置模块

- <font color='green'>**组成**</font>：
   - 可以进行 **增量学习**
   - 可以进行 **在线学习**

<font size=4><font color = 'lightgreen'>**增量学习**</font></font>

- **批模式**得在线学习

**定义**：学得模型后，在接收到**训练**样例时，仅需**根据新样例**对模型进行**更新**，无需重新训练整个模型

-----> **先前学得**的**有效信息**不会被**冲掉**
    
<font size=4><font color = 'lightgreen'>**在线学习**</font></font>

   - **增量学习**的特例

**定义**：每获得一个**新的样本**就进行一次**模型更新**
# <font color ='red'>**SOM网络**</font><font size = 3> 自组织映射   *Self-Organizing Map*</font>
   - **竞争学习**型神经网络
   - **无监督**学习
   - **高位输入**数据映射到 **低维空间**
   - **保持**输入数据在高维空间的**拓扑结构**
   - **输出层**：以 **矩阵**方式排列在二维空间

- <font color='green'>**训练目标**</font>：
   - 为每一个 **输出层**神经元找到合适的权向量

- <font color='green'>**训练过程**</font>：
   - 寻找**最佳匹配单元**：**训练样本**与 每个**输出层**神经元自身携带的**权向量**得距离**最近**得神经元
   - 调整**权向量**：仅调整最佳匹配单元及其邻近神经元的
   - 不断**迭代**，直到**收敛**
# <font color ='red'>**级联相关网络**</font><font size = 3> 构造性神经网络 *Cascade-Correlation*</font>
- <font color='green'>**主要成分**</font>：
   - **级联**：建立层次连接的 **层级结构**
   - **相关**：**最大**化新神经元的输出与网络误差之间的**相关性**来**更新**相关的**参数**

- <font color='green'>**特点**</font>：
   - 无需设置 **网络层数**
   - 无需设置 **隐藏层神经元个数**
   - 训练速度较快
   - 将 **网络结构**当作学习的目标之一
# <font color ='red'>**Elman网络**</font><font size = 3>常用の递归神经网络</font>
- <font color='green'>**特点**</font>：
   - 隐层神经元的**输出**被反馈回来，与下一时刻的**输入层**神经元一起提供**隐层的**下一次**输入**
   - **激活函数**：通常为Sigmoid函数
   - 训练算法：通常为 BP算法

<font size=4><font color = 'lightgreen'>**循环神经网络**</font></font>

- <font color='green'>**特点**</font>：
   - 允许网络中出现 **环状结构**
   - 一些神经元的**输出反馈**回来**作为输入信号**
# <font color ='red'>**长短期记忆神经网络**  *LSTM*</font><font size = 3></font>
- <font color='green'>**类别**</font>：<font size=4><font color = 'lightgreen'>**时间循环**神经网络</font></font>
- <font color='green'>**应用**</font>：
   - 处理/预测 **时间序列**中 间隔/长延迟 の事件
      - 手写识别
      - 语言识别
      - ...
# <font color ='red'>**Boltzmann机**</font><font size = 3> *基于能量的模型*</font>
- <font color='green'>**特点**</font>：
   - 为 **网络状态**定义一个 **能量**
   - 神经元为**布尔型**，只能取{0,1}**两种状态**
   - 训练时，每一个**样本**视为一个**状态向量**
   - **全连接图**，训练复杂度很高

- <font color='green'>**学习目标**</font>：
   **能量最小化** -----> 最小化能量函数
$$
\begin{align*}&E(s) = -\underset{i=1}{\stackrel{n-1}{\sum}}\underset{j=i+1}{\stackrel{n}{\sum}}~w_{ij}~s_i~s_j~-~\underset{i=1}{\stackrel{n}{\sum}}\theta_i~s_i\\&\\&s:状态向量\rightarrow s\in\{0,1\}^n~~~n个神经元的状态\\&w_{ij}神经元i与j之间的连接权\\&\theta_i：i神经元的阈值\\&\\&P(s):s的出现概率\rightarrow由 's的能量'+'所有可能状态向量的能量'  确定：\\&P(s)=\frac{e^{-E(s)}}{\sum_te^{-E(t)}}\end{align*}
$$
- <font color='green'>**训练算法**</font>：
  
<font size=4><font color = 'lightgreen'>**对比散度**  *CD*</font></font>*Contrastive Divergence* 

![](2022-12-15094304.jpg)
![](2022-12-15094326.jpg)