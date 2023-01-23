# <font color ='red'>**线性模型基本概念**</font>
1. **基本形式**:
学的一个通过**属性的线性组合**来进行预测的函数，即：
$$
\begin{align*}
f(x) &= w_1x_1+w_2x_2+...+w_dx_d+b\\
&=w^Tx+b    
\end{align*}
\longrightarrow 确定 w=(w_1;s_2;...;w_d)与b
$$
2. **优点**：
   - **简易性**： 形式简单，易于建模
   - **可拓展性**： 可以在此基础上引入 **层次结构** / **高位映射** 形成更加强大的非线性模型
   - **可理解性**：参数w直观的表达了各属性的重要性

3. <font size = 4><font color='lightgreen'>**无序属性连续性**</font></font>：

假设属性之间关系为：
   - **有序关系**：可以 **连续化** 为连续值
      - eg:{高，中，矮}------> {1.0，0.5，0.0}
   - **无序关系**：有k个属性值，通常转化为k维向量
      - eg:{西瓜，黄瓜，南瓜}------>{ (1,0,0) , (0,1,0) , (0,0,1) }

4. <font size = 4><font color='lightgreen'>参数优化方法</font></font>：

|模型|优化方法|
|:----:|:----:|
|Linear Regression|最小二乘法|
|Logistic Regression|牛顿法/凸优化梯度下降/..|
|LDA|广义瑞丽商+矩阵论|

# <font color ='red'>**线性回归**</font><font size = 3> *Linear Regression*</font><font color='blue'>*Regression*</font>
1. <font color='green'>**目标函数**</font>：
$f(x_i)=wx_i + b~ ~,~ ~使得f(x_i)\simeq y_i$  （对于每一个属性Di）

<font color='green'>**处理后的目标函数**</font>：
合并w与b
$$
\hat{w} = (w;b)\\
\begin{align*}
&X向量加1:\\
&X = \left[\begin{matrix}
x_{11}\quad &x_{12}\quad &...\quad x_{1d}&\quad 1\\
x_{21}\quad &x_{22}\quad &...\quad x_{2d}&\quad 1\\
.\quad &.\quad &...\qquad~.&\quad 1\\
x_{m1}\quad &x_{m2}\quad &...\quad x_{md}&\quad 1\\
\end{matrix}\right]_{m\times(d+1)}=
\left[\begin{matrix}
x_1^T&\quad 1\\
x_2^T&\quad 1\\
.&\quad .\\
.&\quad .\\
.&\quad .\\
x_1^T&\quad 1\\
\end{matrix}\right]    
\end{align*}
$$

2. **优化函数**：

<font color='green'>**损失函数**</font>：

$$
\begin{align*}
\mathcal{loss}(y) &= (f(x) - y)^2\\
&=(y - w^Tx-b)^2    
\end{align*}

$$

<font size = 4><font color='lightgreen'>**均方误差最小化**</font></font>
$$
\begin{align*}
(w^*,b^*)~=&~\underset{(w,b)}{argmin}\underset{i=1}{\stackrel{m}{\sum}} (f(x_i) - y_i)^2\\
&=\underset{(w,b)}{argmin}\underset{i=1}{\stackrel{m}{\sum}} (y_i - wx_i-b)^2
\end{align*}
$$

<font color='green'>**处理后的优化函数**</font>：
$$
\hat{w}^* = \underset{\hat{w}}{argmin}(y - X\hat{w})^T(y-X\hat{w})
$$

1. **优化方法**:

<font size = 4><font color='lightgreen'>**最小二乘法**</font></font>

对w与b分别求导可得：
$$
\begin{align*}
\frac{\partial E(w,b)}{\partial w} = 2(w\underset{i=1}{\stackrel{m}{\sum}} x_i^2 - \underset{i=1}{\stackrel{m}{\sum}} (y_i-b)x_i)\\
\frac{\partial E(w,b)}{\partial b} = 2(mb-\underset{i=1}{\stackrel{m}{\sum}} (y_i-wx_i))
\end{align*}
\stackrel{分别等于0}{\longrightarrow}
\begin{align*}
\frac{\partial E(w,b)}{\partial w} = 0\\
\frac{\partial E(w,b)}{\partial b}=0    
\end{align*}
\\
$$
<font color='green'>**参数最优解**</font>：
$$
\begin{align*}
\hat{w}^*=\frac{\underset{i=1}{\stackrel{m}{\sum}} y_i(x_i-\bar{x})}{\underset{i=1}{\stackrel{m}{\sum}} x_i^2 - \frac{1}{m}(\underset{i=1}{\stackrel{m}{\sum}} x_i)^2}\\
\hat{b}^*=\frac{1}{m}\underset{i=1}{\stackrel{m}{\sum}} (y_i - \hat{w^*}x_i)
\end{align*}\\
其中 \bar{x} = \frac{1}{m}\underset{i=1}{\stackrel{m}{\sum}} x_i
$$
<font color='green'>**函数处理后的最优解**</font>：
$$
\hat{w}^* = (X^TX)X^{-1}y
$$

4. <font size = 3><font color='lightgreen'>**广义线性回归**</font></font>：
**定义**：对于 **单调**+**可微**函数g(·)，令：
$$
y = g^{-1}(wX^T+b)\\
g(·)：联系函数
$$

# <font color ='red'>**对数几率回归**</font><font size = 3> *logistic Regression* </font><font color='blue'>*Classification*</font>
**二分类问题**
1. **定义**：
   - 取广义线性模型 => g(·) = ln(·)
   - 将输出实值$z=w^Tx+b$转化为0/1值   （ 输出标记为 y∈{0，1} ）

2. **转换z为 0/1值**

<font color='green'>**目标线性模型の函数表达式**</font>：
$$z = w^Tx+b$$

<font size = 3><font color='lightgreen'>**单位阶跃函数**</font></font>：

$$
y =\left \{ \begin{aligned}
0,&\qquad z<0;\\
0.5,&\qquad z=0;\\
1,&\qquad z>0,
\end{aligned}\right.
=\left \{ \begin{aligned}
反例&\qquad z<0\\
任意判定&\qquad z=0\\
正例&\qquad z>0
\end{aligned}\right.
$$
**缺点**：
   - 不连续
   - 不可导
   - 不能直接用做 g(·)

<font size = 3><font color='lightgreen'>**Sigmoid函数**</font></font>：

$$
\begin{align*}
Sigmoid(z) = y &= \frac{1}{1+e^{-z}}\\
&=\frac{1}{1+e^{-(w^Tx+b)}}    \\
&\\
\end{align*}\\
\begin{align*}
&性质：\\
&\qquad\qquad\qquad f'(x) = f(x)(1-f(x))  
\end{align*}
$$
**优势**：
   - **单调可微**
   - **任意阶可导**
   - 一定程度上近似于 单位阶跃函数


<font size = 3><font color='lightgreen'>**几率**</font></font>：

$$
\frac{y}{1-y}
$$

<font size = 3><font color='lightgreen'>**对数几率**</font></font>：
**定义**：样本 作为 **正例**の **相对可能性**の **对数**
$$
ln\frac{y}{1-y}\\
$$
$$
\longrightarrow ln\frac{y}{1-y}=w^Tx+b=z
$$
3. **Logistic回归优势**：
   - 分类学习算法
   - **直接建模**：对**分类可能性**建模，无需 事先 假设 **数据分布**
   - **概率化**：不仅预测出类别 ，还可得到近似の **概率分布**
   - **目标函数优越性**：目标函数是**任意阶可导**的**凸**函数
   - **易解性** ： 可直接应用 现有の **数值优化算法**求解最优解

<font color='green'>**概率化处理**</font>：
$$
\begin{align*}
记：\\
& \qquad p(y=1~|~x) =\frac{e^{w^Tx+b}}{1+e^{w^Tx+b}},\\
& \qquad p(y=0~|~x) =\frac{1}{1+e^{w^Tx+b}}\\
则：\\
& \qquad ln\frac{y}{1-y}=ln\frac{p(y=1~|~x)}{p(y=0~|~x)}=w^Tx+b
\end{align*}
$$

4. <font color='green'>**损失函数**--**对数似然函数**</font>：

<font size = 3><font color='lightgreen'>**极大似然法**</font></font>

**对数似然函数**：
$$
\mathcal{L(w,b)}~=~\underset{i=1}{\stackrel{m}{\sum}}~ln~p(y_i~|~x_i,w,b)
$$
每个样本属于真实标记的概率越大越好：
$$
\mathcal{loss}(w,b)=~-\underset{i=1}{\stackrel{m}{\sum}}~ln~p(y_i~|~x_i,w,b)~=~-\mathcal{L(w,b)}
$$
**重写似然项**：
$$
\begin{align*}
记 \mathcal{W} = (w,b)\\

\qquad p(y_i~|~x_i,w,b) &= y_ip_1(\hat{x_i} ; \mathcal{W})+(1-y_i)p_0(\hat{x_i};\mathcal{W})\\&=
\left \{ \begin{aligned}
p(y_1=1~|~X,\mathcal{W}) = (y_1=1)·p_1(\hat{X_{y=1};\mathcal{W}})+(1-1)·p_0(\hat{X_{y=0};\mathcal{W}})=p_1(\hat{X_{y=1};\mathcal{W}})\\ 
p(y_0=1~|~X,\mathcal{W}) =0·(p_1(\hat{X_{y=1};\mathcal{W}}))(y_0=1-0)·p_0(\hat{X_{y=0};\mathcal{W}})=p_0(\hat{X_{y=0};\mathcal{W}}) 
\end{aligned}\right.\\
&\\
\end{align*}\\
最小化损失函数=\Updownarrow 最大化似然函数
$$
**损失函数**：
$$
\mathcal{loss}(\mathcal{W})=\underset{i=1}{\stackrel{m}{\sum}}(-y_i\mathcal{W}^T\hat{x_i} +ln~(1+e^{\mathcal{W}^T\hat{x_i}}))\\
\mathcal{W}^* = \underset{\mathcal{W}}{argmin}~~\mathcal{loss}(\mathcal{W})
$$

5. <font color='green'>**求解参数最优解**</font>：

<font size = 3><font color='lightgreen'>**牛顿法** ---迭代更新参数</font></font>：

$$
\begin{align*}
\mathcal{W}^{t+1} &= \mathcal{W}^t - (\frac{\partial^2\mathcal{loss}(\mathcal{W})}{\partial\mathcal{W}\partial\mathcal{W}^T})^{-1}\quad·\frac{\partial\mathcal{loss}(\mathcal{W})}{\partial\mathcal{W}}\\
\\
&=上一代参数 - 二阶导数的逆·一阶导数\\
\end{align*}\\
\\
\begin{align*}
其中：\\    
&一阶导数：~\frac{\partial\mathcal{loss}(\mathcal{W})}{\partial\mathcal{W}} = - \underset{i=1}{\stackrel{m}{\sum}}\hat{x_i}(y_i - p_1(\hat{x_i};\mathcal{W}))\\
&二阶导数：~(\frac{\partial^2\mathcal{loss}(\mathcal{W})}{\partial\mathcal{W}\partial\mathcal{W}^T}) = \underset{i=1}{\stackrel{m}{\sum}}\hat{x_i}\hat{x_i}^T p _1(\hat{x_i};\mathcal{W})(1-p _1(\hat{x_i};\mathcal{W}))
\end{align*}
$$
<font size = 3><font color='lightgreen'>**凸优化梯度下降**</font></font>


# <font color ='red'>**线性判别分析**</font><font size = 3> *LDA*</font>
1. **基本思想**：
   - **样例投影**到同一条**直线**（一维空间）上
   - **同类**样例的投影点尽可能地**接近**
   - **异类**样例地投影点尽可能地**远离**

2. **基本概念**：

<font size = 3><font color='lightgreen'>**类内散度矩阵**   *$S_w$*</font></font>*within-class scatter matrix*：  
**二分类**：
$$
\begin{align*}
S_w& = \Sigma_0 +\Sigma_1\\
&=\underset{x\in X_0}{(x-\mu_0)(x-\mu_0)^T}~+~\underset{x\in X_1}{(x-\mu _1)(x-\mu_1)^T}
\end{align*}
$$
**多分类**：
$$
\begin{align*}
&S_w = \underset{i=1}{\stackrel{N}{\sum}}S_{w_i}\\
&其中：\\
&S_{w_i} = \underset{x\in X_i}{\sum}(x-\mu_i)(x-\mu_i)^T    
\end{align*}
$$

<font size = 3><font color='lightgreen'>**类间散度矩阵**    *$S_b$ *</font></font>*between-class scatter matrix*：
**二分类**：
$$
S_B = (\mu_0-\mu_1)(\mu_0-\mu_1)^T
$$

**多分类**：
$$
S_b = S_t-S_w\\
=\underset{i=1}{\stackrel{N}{\sum}}m_i()
$$

<font size = 3><font color='lightgreen'>**广义瑞利商**</font></font>：

**二分类**:
$$
\frac{w^TS_bw}{w^TS_ww}
$$
**多分类**：
$$
\frac{tr(W^TS_bW)}{tr(W^TS_wW)}\\
$$
tr(·):矩阵的迹

<font size = 3><font color='lightgreen'>**全局散度矩阵** *$S_t$*</font></font>：
$$
\begin{align*}
S_t &= S_b +S_w\\
&=\underset{i=1}{\stackrel{m}{\sum}}(x_i-\mu)(x_i-\mu)^T    
\end{align*}
$$

3. <font color='green'>**目标函数**/**代价函数 $\mathcal{J(w)}$**</font>：

**二分类**：
$$
\begin{align*}
\mathcal{J(W)}&= \frac{||w^T\mu_0 - w^T\mu_1||^2_2}{w^T\Sigma_0w+w^T\Sigma_1w}\\
&=\frac{w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^Tw}{w^T(\Sigma_0+\Sigma_1)w}\\
&=\frac{w^TS_bw}{w^TS_ww}\qquad\qquad 广义瑞利商\\
\end{align*}
$$
前提假设：$w^TS_ww=1$   ----> 解与w长度无关，至于方向有关 （一维直线）
$\therefore$优化目标：
$$
\begin{align*}
&\left\{\begin{aligned}
\underset{w}{max} ~~~\mathcal{J(W)}\\
s.t. w^TS_ww = 1  
\end{aligned}\right.\\
\\
\longrightarrow&\left\{\begin{aligned}
\underset{w}{min}~-w^TS_bw\\
s.t. w^TS_ww = 1     
\end{aligned}\right.
\end{align*}
$$

**多分类**：
$$
\mathcal{J(W)}=\underset{\mathcal{W}}{max}\frac{tr(\mathcal{W}^TS_b\mathcal{W})}{tr(\mathcal{W^TS_w\mathcal{W}})}
$$

4. <font color='green'>**求解最优解  $w^*$**</font>：

<font size = 3><font color='lightgreen'>**拉格朗日乘子法**</font></font>：

**二分类最优解**：
$$
w^* = S_w^{-1}(\mu_0 - \mu_1)
$$

**多分类最优解**：
$$
\mathcal{W}^* = S_w^{-1}S_b的~d'个~最大非零广义特征值的特征向量构成的~矩阵\\
其中：d'\leq N-1
$$
5. <font color='green'>**特点**</font>：
- **多分类LDA**：
   - **投影**：将样本投影到 N-1维 空间
   - $\qquad\Updownarrow$
   - **监督降维**技术


# <font color ='red'>**多分类问题**</font><font size = 3></font>
**拆分策略**：
   - **多分类** = 多个**二分类**任务求解
   - 每个二分类任务都需要训练一个分类器

<font size = 3><font color='lightgreen'>**一对一**拆分策略  *OvO*</font></font>  One vs One：

**拆分策略** ($\frac{N(N-1)}{2}$个分类器) ：
   - Step 1：N个类别两两分别作为正反类进行配对----->  $\frac{N(N-1)}{2}$个二分类任务
   - Step 2：对应产生$\frac{N(N-1)}{2}$个分类分类器 --> 各自的分类结果
   - Step 3：**投票选举**最终结果 ---->**预测个数最多**的类别作为最终的分类结果

<font size = 3><font color='lightgreen'>**一对其余**拆分策略  *OvR*</font></font>  One vs Rest：

**拆分策略**  ($N$个分类器)：
   - Step 1：每次将**一类**作为**正例**，**其他(N-1)类**作为**反例**
   - Step 2：设置各分类器的**预测置信度**
   - Step 3：选择**置信度最大**的类别标记作为分类结果

<font size = 3><font color='lightgreen'>**多对多**拆分策略  *MvM*</font></font>  Many vs Many

**拆分策略**：
   - Step 1 ：将m类作为正类，其余(N-m)作为反类

**如何选择m？**：

**MvM技术**：<font size = 3><font color='lightgreen'>**纠错输出码**   *EOOC*</font></font>

1. 思想：
   - 将**编码**的思想引入 **类别拆分**
   - 尽可能地在**解码**过程中具有**容错性**

2. 主要工作步骤：  
- **编码**(M个分类器)：
   - 将N个类别做M次划分
   - 每一次划分将一部分(m类)作为正类，其余(N-m)类作为反类
   - 每一次划分形成一个二分类训练集
- **解码**：
   - M个分类器分别对test进行预测
   - M次划分 分别得到的 预测标记形成一个编码：$M\times 1$的**预测编码**
   - 测试样例+各个类别各自的编码钩沉**编码矩阵**
   - 将 **预测编码**与各个类别各自的编码进行比较
   - 选择**距离最小的类别**作为最终预测结果

3. <font size =2><font color='lightgreen'>**编码矩阵**  *coding matrix*</font></font>：

**主要形式**：
   - **二元码**：{ 正类， 反类 }
   - **三元码**：{ 正类， 停用类 ，反类 }

4. **特点**：
   - 具有一定的 **容错力** +**修正力**
   - EOOC编码越长， **纠错**能力越强 ，但是需要训练的分类器越多，**计算/存储开销**增大
   - 相同长度的EOOC编码，任意两类之间的**编码距离**越远，**纠错**能力越强

# <font color ='red'>**类别不平衡问题**</font><font size = 3>*class imbalance*</font>
1. **定义**：分类任务中 不同的训练样例 **数目差别很大** 的情况
2. **处理策略**： <font color='lightgreen'>**再缩放**</font>

<font size = 3><font color='lightgreen'>**欠采样**   *undersampling*</font></font>**去除**一些**多余**的样例，使得正反样例数目**相近**

<font size = 3><font color='lightgreen'>**过采样**   *oversampling*</font></font>**增加**一些**少数**的样例样本，使得正反样例数目**相近**

<font size = 3><font color='lightgreen'>**阈值移动**   *threshold-moving*</font></font>

- 直接基于原始数据集进行学习
- 预测时，将下式嵌入到决策过程中：
$$
\frac{y'}{1-y'} = \frac{y}{1-y}\times\frac{m^-}{m^+}
$$