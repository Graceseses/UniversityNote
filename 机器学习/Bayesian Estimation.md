# <font color ='red'>**贝叶斯决策论**</font>
- <font color='green'>**原理**</font>：对于分类问题，如何基于 **概率**+**误判损失**来选择最优类别

- <font color='green'>**目标**</font>：寻找一个**判定准则 $h:X\mapsto Y$**以最小化**总体风险**

- <font color='green'>**基本概念**</font>：
### <font color = 'lightgreen'>**(误判)损失  $\lambda_{ij}$**</font>
**定义**：将真实标记为 $c_j$ 的样本 **误分类**为 $c_i$ 的损失，记  $\lambda_{ij}$
$$
\lambda_{ij}=\left\{\begin{aligned}
~~0~,&\qquad if~i=j\\
~~1~,&\qquad otherwise
\end{aligned}\right.
$$

### <font color = 'lightgreen'>**条件风险**(期望损失)  $R(h(x)|x)$</font>
**定义**：基于 **后验概率**$~P(c_i|x)~$获得的 将样本x分类为$c_j$所产生的期望误差/在样本上的 **条件风险**
$$
R(c_i|x)~=~\underset{i=1}{\stackrel{N}{\sum}}~\lambda_{ij}P(c_j|x)
$$
### <font color = 'lightgreen'>**总体风险**  $R(h)$</font>
**定义**：基于 **目标**所寻找的判定准则h
$$
R(h) = E_x~[~R(h(x)|x)~]\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad P(c_j|x):后验概率\rightarrow将样本x标记为c_j的概率
\end{align*}
$$
- <font color='green'>**定理准则**</font>：
### <font color = 'lightgreen'>**贝叶斯判定准则**  $h^*$</font>
**定义**：在 **每个样本** 上选择那个能使 **条件风险$~R(c|x)~$最小**的 **类别标记**，从而最小化总体风险
$$
h^*~=~\underset{c\in y}{argmin}~R(c|x)\\
\begin{align*}
&\\
&其中：\\
&\qquad \qquad h^*~:~贝叶斯最优分类器\\
&\qquad \qquad R(h^*)~:~贝叶斯风险\\
&\qquad \qquad 1-R(h^*)~:~分类器能达到的最好性能，模型精度上限
\end{align*}
$$
- <font color='green'>准确估计**后验概率$P(c|x)$**</font>：
### <font color = 'lightgreen'>**判别式模型**</font>
**定义**：给行样本x，通过对后验概率P(c|x)**直接建模**来预测c

**eg**： 
   - 决策树
   - BP神经网络
   - 支持向量机
   - ...


### <font color = 'lightgreen'>**生成式模型**</font>
**定义**：
   - 先对 **联合概率分布**P(c,x)进行建模
   - 再利用下列公式获得P(c|x)$$\begin{align*}~P(c|x)~&=~\frac{P(x,c)}{P(x)}~\\&\\&=\frac{P(c)P(x|c)}{P(x)} \end{align*}\\\begin{align*}其中：\\&P(c):先验概率\longrightarrow样本空间中各类样本所占的比例\\&P(x|c):类条件概率~/~似然概率\longrightarrow 标记c相对于样本x的\\&P(x):证据因子\longrightarrow与标记无关\end{align*}$$
### <font color = 'lightgreen'>**概率模型训练过程**</font>
- <font color='green'>**原理**</font>：
   -  **概率模型**の **训练过程**  = **参数估计**过程

||**频率主义**学派|**贝叶斯**学派|
|:----:|:----:|:----:|
|概率定义|<font color='lightgreen'>**大数定律** </font>: 通过 **大量**+**独立**实验$\Longrightarrow$概率=**统计均值**|概率 = <font color='lightgreen'>**信念度**  （无需大量实验）|
|参数定义|未知参数 = **普通变量**（固定值）|一切变量（参数/样本） = **随机变量**|
||仅 利用 **抽样数据**|**过去知识**+**抽样数据**|

# <font color ='red'>**朴素贝叶斯分类器**</font>
- <font color='green'>**模型假设**</font>：
<font size =4><font color = 'lightgreen'>**属性条件独立性假设**</font></font>
**定义**：假设所有属性相互独立$\longrightarrow$假设每个属性独立的对分类结果发生影响
$$
\begin{align*}
P(c|x)&=\frac{P(c)P(x|c)}{P(x)}\\
&\\
&=\frac{P(c)}{P(x)}\underset{i=1}{\stackrel{d}{\prod}}P(x_i|c)\\
\end{align*}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad d:属性数目\\
&\qquad\qquad x_i:样本x在属性i上的取值
\end{align*}
$$
- <font color='green'>**朴素贝叶斯分类器**</font>：
$$
h_{nb}(x)=\underset{c\in\mathcal{Y}}{argmax}~P(c)\underset{i=1}{\stackrel{d}{\prod}}~P(x_i|c)\\
$$

- <font color='green'>**模型训练**</font>：
   - **P(c)**：基于训练集D估计$$P(c)=\frac{|D_c|}{|D|}$$
   - **$P(x_i|c)$** : 估计每个属性的类条件概率$$\Updownarrow$$

- <font color='green'>估计 **类条件概率P(x|c)**</font>：
**求解方法**：
   - Step 1 ：假定其服从某种 **概率分布**形式
   - Step 2 ：基于训练样本对概率分布进行 **参数估计**

<1. **离散属性**：
$$
P(x_i|c)~=~\frac{|D_{c,x_i}|}{|D_c|}\\
\begin{align*}
&其中：\\
&\qquad\qquad D_c~：训练集D中的第c类样本集合\\
&\qquad\qquad D_{c,x_i}~:~在D_c中第i个属性上取值为x_i的样本集合
\end{align*}
$$

**P=0**？：属性信息未在训练集中出现$\longrightarrow$未出现的属性值被抹去

<font size =4><font color = 'lightgreen'>平滑处理------**拉普拉斯修正** *MLE*</font></font>

   - **P(c)**：基于训练集D估计$$P(c)=\frac{|D_c|~+~1}{|D|~+~N}\\  \begin{align*}&\\&其中：\\&\qquad\quad N:训练集D中的种类数\end{align*}$$
   - **$P(x_i|c)$** : 估计每个属性的类条件概率$$P(x_i|c)~=~\frac{|D_{c,x_i}|~+~1}{|D_c|~+~N_i}\\ \begin{align*}&\\&其中：\\&\qquad\quad N_i:第i个属性的所有取值可能数\end{align*}$$
   - 
<2. **连续属性**：考虑概率密度函数
$$
\begin{align*}
&假定类条件概率符合正态分布：P(x_i|c)\sim \mathcal{N}(\mu_{c,i},\sigma^2_{c,i})\\  
&\\
&\qquad\qquad P(x_i|c)~=~\frac{1}{\sqrt{2\pi}\sigma_{c,i}}exp(-\frac{(x_i-\mu_{c,i})^2}{2\sigma^2_{c,j}})
\end{align*}
$$

**求解$\mu_{c,i}与\sigma_{c,j}$**? <font size =4><font color = 'lightgreen'>**极大似然估计** *MLE*</font></font>
**原理**：在参数$\theta$中

**特点**：
   - 属于**频率主义学派**
   - 根据**数据采样**来估计**概率分布参数**的经典方法
   - **假设**：独立同分布假设

- <font color='green'>**模型基础**</font>：
   - 结果の正确性 **严重依赖**于 对 所假设的 **概率分布** 是否符合 潜在的真实数据分布
- <font color='green'>**步骤**</font>：
$$
假设样本集D中的所有样本相互 独立同分布 ， 则概率分布参数对数据集D的似然为：\\
\begin{align*}
&\\
p(D|\theta)~=~\underset{x\in D_c}{\prod}~P(x|\theta)
\end{align*}\\

\begin{align*}
&连乘操作\longrightarrow '下溢'\\
&\qquad\qquad\qquad\qquad\qquad\qquad  \Updownarrow\\
&对数似然函数: \\
&\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad  
\end{align*}\\
\begin{align*}
LL(\theta_c)~&=~log~P(D|\theta)\\
&\\
&=~\underset{x\in D}{\sum}~logP(x|\theta)\\
\Updownarrow
\end{align*}\\
\begin{align*}
&\\
&参数\theta的极大似然估计\hat{\theta}:\\
&\qquad\qquad\qquad  \hat{\theta}    ~=~\underset{\theta}{argmax}~LL(\theta_c)
\end{align*}
$$

- <font color='green'>**应用**</font>：
   - **懒惰学习**：任务数据更换频繁
   - **增量学习**：数据不断增加

# <font color ='red'>**半朴素贝叶斯分类器**</font>

- <font color='green'>**模型假设**</font>：
<font size =4><font color = 'lightgreen'>**独依赖估计** *ODE*</font></font>   One-Dependent Estimator

**定义**：假设每个属性在类别之外最多仅依赖于一个其他属性
   - 适当考虑一部分属性间的相互依赖关系
   - 无需进行完全联合概率的计算
$$
P(c|x)\propto P(c)~\underset{i=1}{\stackrel{d}{\prod}}~P(x_i|c,pa_i)\\
\begin{align*}
&\\
&其中：\\
&\qquad\quad pa_i：属性x_i的父属性（属性x_i以来的属性）
\end{align*}
$$
- <font color='green'>**不同假设对应的算法**</font>：

<font size =4><font color = 'lightgreen'>**超父**独依赖估计 *SPODE*</font></font>   Super-Parent ODE

- <font color='green'> **假设**</font> ：所有属性都依赖于同一个属性$\longrightarrow$"**超父**"

- <font color='green'>**超父选择**</font>：
   - <font color='lightgreen'>**交叉验证**</font>：选择超父属性

<font size =4><font color = 'lightgreen'>**树增强の朴素贝叶斯** *TAN*</font></font>   Tree Augmented $na\ddot{i}ve$ Bayes

**基础**：**最大带全生成树**算法

**步骤**：
   - 计算任意属性之间的 <font color = 'lightgreen'>**条件互信息**</font>$$I(x_i,x_j|y)=\underset{x_i,x_j;c\in \mathcal{Y}}{\sum}~P(x_i,x_j|c)~log~\frac{P(x_i,x_j|c)}{P(x_i|c)P(x_j|c)}$$
   - 以属性为结点$\longrightarrow$建立完全图，权重设为$I(x_i,x_j|y)$
   - 构建完全图の **最大带全生成树**
   - 挑选根变量，将边置为有向
   - 加入类别结点y，增加y到各个属性的有向边

**特点**：
   - 条件互信息$I(x_i,x_j|y)$刻画了属性在已知类别的情况下的相关性
   - 保留了 **强相关**属性之间的依赖性

<font size =4><font color = 'lightgreen'>**平均**独依赖估计 *AODE*</font></font>   Averaged ODE

**基础**：
   - **集成学习**
   - **SPODE**

**思想**：
   - 将每个属性作为 **超父**来构造SPODE
   - 将**具有足够训练数据支撑**（$|D_{x_i}|≥m'$）的SPODE集成作为最终结果$$P(c|x)~\propto~\underset{\underset{|D_{x_i}|≥m'}{i=1}}{\stackrel{d}{\sum}}~P(c,x_i)\prod~P(x_j|c,x_i)$$

**优点**：
   - 无需进行模型选择（假设 **概率分布**）
   - **预计算**节约预测时间
   - 采用 **懒惰学习**的学习方式
   - 易于实现增量学习

# <font color ='red'>**贝叶斯网**</font>
- <font color='green'>**假设**</font>：给定**父结集**，每个属性 与他的 **非后裔属性** 独立

- <font color='green'>**模型基础**</font>：
   - <font size =4><font color = 'lightgreen'>**有向无环图** *DAG*</font></font>    Directed Acyclic Graph
      - 刻画属性之间的 **依赖关系**  
   - <font size =4><font color = 'lightgreen'>**条件概率表** *CPT*</font></font>    Conditional Probability Table
      - 描述属性的 **联合概率分布** 

- <font color='green'>**DAG**---分析变量的典型**依赖关系**</font>：
   - <font color = 'lightgreen'>**同父结构**</font>$$父变量值完全未知，子变量之间不独立\\父变量值给定，子变量之间条件独立\\$$
   - <font color = 'lightgreen'>**V型结构**（冲撞结构）</font>$$子变量值给定，父变量之间必不独立\\子变量值完全未知，父变量之间相互独立$$
   - <font color = 'lightgreen'>**顺序结构**</font>$$给定中间变量，左右变量之间条件独立$$

- <font color='green'><font color = 'lightgreen'><font size= 4>**有向分离**</font></font>---分析变量间的 **条件独立性**</font>：将 **有向无环图ADG**$\longrightarrow$**无向图（道德图）**
   -  <font color = 'lightgreen'>**道德化**</font>：
      - 找出图中所有的 **V型结构**  ，在两个父节点之间加上一条**无向边**
      - 所有 **有向边**$\longrightarrow$**无向边**

- <font color='green'>**目标**</font>：根据训练集D找出结构 **最恰当**的 **贝叶斯网**

$$\Updownarrow$$

- <font color='green'><font color = 'lightgreen'><font size= 4>**评分函数$S(·)$**</font></font>----评估贝叶斯网&训练数据 的 **默契程度**</font>：

<font color = 'lightgreen'><font size= 4>**最小描述长度准则**  *MDL*</font></font>Minimal Description Length

**原理**：**最短**综合编码长度的贝叶斯网
$$
\begin{align*}
&评分函数：\\
&\qquad\qquad\qquad S(B|D)~=~f(\theta)|B|~-~LL(B|D)\\
&\qquad\qquad\qquad LL(B|D)~=~\underset{i=1}{\stackrel{m}{\sum}}logP_B(x_i)\\
&其中：\\
&\qquad\qquad |B|:贝叶斯网的参数个数\\
&\qquad\qquad f(\theta):描述每个参数\theta所需要的字节数\\
&\qquad\qquad D：数据集\\
&\qquad\qquad LL(B|D):贝叶斯网の对数似然函数
\end{align*}
$$
- $f(\theta)$:
   - <font color = 'lightgreen'>**AIC评分函数**</font>$$\begin{align*}&取f(\theta)=1:\\&\qquad\qquad AIC(B|D)~=~1·|B|-LL(B|D)\end{align*}$$
   - <font color = 'lightgreen'>**BIC评分函数**</font>$$\begin{align*}&取f(\theta)=\frac{1}{2}log m:\\&\qquad\qquad BIC(B|D)~=~\frac{1}{2}log m·|B|-LL(B|D)\end{align*}$$
   - <font color = 'lightgreen'>**'极大似然估计'评估函数**</font>$$\begin{align*}&取f(\theta)=0:\\&\qquad\qquad LL(B|D)~=~-LL(B|D)\end{align*}$$

- <font color='green'><font color = 'lightgreen'><font size= 4>**近似推断**</font></font>----通过部分属性变量 **预测** **其他的属性**变量取值</font>：

<font color = 'lightgreen'>**推断**</font>：通过**已知变量观测值**来推测 **待查询变量**的过程

<font color = 'lightgreen'>**证据**</font>：推断使用的 **已知变量观测值**

<font color = 'lightgreen'>**近似推断**</font>：降低精度要求，在有限的时间内求得 **近似解**即可

**求解方法**：<font color = 'lightgreen'><font size= 4>**吉布斯采样** *Gibbs Sampling*</font></font>

近似估算的后验概率：
$$
\begin{align*}
&\qquad\qquad\qquad p(Q=q|E=e)\simeq \frac{n_q}{T}\\
&\\
&其中：\\
&\qquad\qquad\qquad T:采样次数\\
&\qquad\qquad\qquad E:证据变量\{E_1,E_2,...,E_k\}，其取值为e=\{e_1,e_2,...,e_k\}\\
&\qquad\qquad\qquad Q:待查询变量
\end{align*}
$$
<font color = 'lightgreen'>**马尔科夫链**</font>：每一步仅仅依赖于前一步的状态
# <font color ='red'>**EM算法**</font><font size =3>Expectation-Maximization</font>
<font color = 'lightgreen'>**隐变量**</font>：未观测变量集

<font color = 'lightgreen'>**$Z$**</font> : **隐变量**集合

<font color = 'lightgreen'>**$X$**</font> : **已观测**变量 集合

<font color = 'lightgreen'>**$LL(\Theta|X)$**</font> :已观测数据的对数 **边际似然**

<font color = 'lightgreen'>**$\mathcal{\theta}$**</font> : 模型参数

- <font color='green'>**作用**</font>：
   - 常用的 **估计参数隐变量**的利器
   - 迭代式算法
   - ----

- <font color='green'>**基本思想**</font>：基于参数$\theta^t$计算隐变量Z的概率分布$P(Z|X,\theta^t)$
   - <font color='green'>**E步**</font>：**参数$\theta$已知**，则可以根据训练数据推断出 **最优隐变量Z**的值/期望

   - <font color='green'>**M步**</font>：**若Z已知**，则可以对参数$\theta$做 **极大似然估计**
   - ---

- <font color='green'>**算法实现**</font>：迭代进行
   - <font color='green'>**E步** (期望步) </font>：已知以前参数 $\Theta^t$ 推断 **隐变量分布**  $P(Z|X,\Theta^t)\longrightarrow$计算对数似然 $LL(\Theta|X,Z)$ 对Z的期望$$Q(\Theta|\Theta^t)~=~E_{Z|X,\Theta^t}~LL(\theta|X,Z)$$  $$\begin{align*}&其中：\\&\qquad\qquad\qquad LL(\Theta)~=~ln~P(X,Z|\Theta)\end{align*}$$
   - <font color='green'>**M步** (最大化步)</font>：寻找 **参数最大化期望似然**$$\Theta^{t+1}~=~\underset{\Theta}{argmax}~Q(\Theta|\Theta^t)$$
   - ---