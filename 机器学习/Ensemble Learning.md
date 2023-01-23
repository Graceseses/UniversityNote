# <font color ='red'>**基本概念**</font>
### <font color = 'lightgreen'>**集成学习** *ensemble learning*</font>：
- <font color='green'>**定义**</font>：**构建**+**结合**多个学习器来完成学习任务的系统

- <font color='green'>**别名**</font>：
   - 多分类器系统 *multi-classifier system*
   - 基于委员会的学习 *committee-based learning*

- <font color='green'>**结构**</font>：
   - 先产生 **一组** **个体学习器**
   - 使用 **某种结合策略** 结合起来

- <font color='green'>**要求**</font>：<font color = 'lightgreen'>**好而不同**</font>="准确性"+"多样性"
- <font color='green'>**核心要求**</font>：如何产生并结合 **好而不同**得个体学习器
- <font color='green'>**种类**</font>：
   - <font color = 'lightgreen'>**串行**生成的**序列化**方法</font> ：个体学习器之间存在 **强依赖**关系
   - <font color = 'lightgreen'>**同时**生成的 **并行化**方法</font> ：可同时生成得并行化方法
- <font color='green'>**特点**</font>：
   - 实际计算开销 **并不比**使用单一学习器**大**很多 
### <font color = 'lightgreen'>**同质**</font>
**定义**：集成中只包含**同种类型**的个体学习器
### <font color = 'lightgreen'>**异质**</font>
**定义**：集成中由**不同的**学习算法生成
### <font color = 'lightgreen'>**基学习器**</font>
**定义**：同质集成中的 **个体学习器**
### <font color = 'lightgreen'>**基学习算法**</font>
**定义**：基学习器对应的基学习算法
### <font color = 'lightgreen'>**组件学习器**</font>
**定义**：**异质**集成中的个体学习器
### <font color = 'lightgreen'>**弱学习器**</font>
**定义**：泛化性能 **略优于** **随机**猜测的学习器

# <font color ='red'>好而不同</font>
<font color = 'green'><font size =4></font>**学习器集合策略**</font>
- <font color='green'>**组合策略**</font>：
   - **统计**方面：减小 "**误选假设空间**$~~~\underset{\longrightarrow}{导致}~~~$**泛化性能不佳**" の **几率** 
   - **计算**方面：降低 "陷入坏的 **局部极小**$~~~\underset{\longrightarrow}{影响}~~~$ **泛化性能**" の **风险**
   - **表示**方面：扩大 " **假设空间学习** 对于 **真实空间** " の 更好**近似**

- <font color='green'>**常用方法**</font>：

**平均法**：
### <font color = 'lightgreen'>**简单平均法**</font>
$$
H(x)~=~\frac{1}{T}\underset{i=1}{\stackrel{T}{\sum}h_i(x)}
$$
**应用**：个体学习器 **性能相近**
### <font color = 'lightgreen'>**加权平均法**</font>
$$
H(x)~=~\underset{i=1}{\stackrel{T}{\sum}~w_i·h_i(x)}\\
\begin{align*}
&\\
&其中：\\
&\qquad\quad w_i:个体学习器h_i的权重~，~w_i\geq 0 ,\underset{i=1}{\stackrel{T}{\sum}}w_i~=~1
\end{align*}
$$
**应用**：个体学习器 **性能迥异**

**投票法**：---"标签型输出"の最常见结合策略
### <font color = 'lightgreen'>**绝对多数 投票法**</font>
**定义**：选择 **票数过半** 的进行标记
$$
H(x)~=~\left\{\begin{align*}
类别标记~c_j~,&\qquad if \underset{i=1}{\stackrel{T}{\sum}}h_i^j(x)>0.5\underset{k=1}{\stackrel{N}{\sum}}\underset{i=1}{\stackrel{T}{\sum}}h_i^k(x)（c_j票数过半）\\
reject（拒绝预测）~，&\qquad otherwise
\end{align*}\right.
$$
### <font color = 'lightgreen'>**相对多数 投票法**</font>
**定义**：选择 **票数最高** 的 标记
$$
H(x)~=~c_{\underset{j}{argmax}\sum_{i=1}^Th_i^j(x)}
$$
### <font color = 'lightgreen'>**加权 投票法**</font>
**定义**：选择 **加权后票数最多** 的 标记
$$
H(x)~=~c_{\underset{j}{argmax}\sum_{i=1}^T~w_ih_i^j(x)}
$$

 - <font color = 'green'>**标记输出类型**</font>：
    -  <font color = 'lightgreen'><font size =4>**硬投票**</font></font>$$h_i^j(x)\in\{~0~,~1~\}$$
    -  <font color = 'lightgreen'><font size =4>**软投票**</font></font>$$h_i^j(x)~\in~[~0~,~1~]$$

**学习法**：
### <font color = 'lightgreen'>**Stacking**</font>
**定义**：通过另一个学习器进行结合$~~~~~~\underset{\Longrightarrow}{被结合的学习器}~~~$ <font color = 'lightgreen'><font size =4>**元/次级 学习器**</font></font>

----

<font color = 'green'><font size =4></font>**多样性**</font>

<font color = 'lightgreen'><font size =4></font>**分歧  $A(h|x)$**</font>：
   - 反映 个体学习器 在样本x上的 **不一致性**
   - 反映 个体学习器的 **多样性**
$$
\begin{align*}
&个体学习器~h_i~的分歧：\\
&\qquad\qquad\qquad\qquad A(~h_i|x~)~=~(h_i(x)~-~H(x))^2  \\
&\\
&集成学习の分歧：\\ 
&\qquad\qquad\qquad\qquad \bar{A}(~h|x~)~=~\sum_{i=1}^T~w_iA(h_i|x)\\
&\qquad\qquad\qquad\qquad\qquad\quad~~~=\sum_{i=1}^Tw_i(h_i(x)-H(x))^2
\end{align*}
$$

<font color = 'lightgreen'><font size =4></font>**泛化误差  $E(h|x)$**</font>：
$$
\begin{align*}
&个体学习器~h_i~的泛化误差：\\
&\qquad\qquad\qquad\qquad E_i~=~{\displaystyle\int E(~h_i|x~)~p(x),dx } \\
&\\
&集成学习の分歧：\\
&\qquad\qquad\qquad\qquad E~=~={\displaystyle\int E(~H|x~)p(x),dx}\\
\end{align*}
$$

<font color = 'lightgreen'><font size =4></font>**误差-分歧分解** *E*</font>：
$$
E~=~\bar{E}~-~\bar{A}\\
\begin{align*}
&其中：\\
&\qquad\qquad\qquad\hat{E}~=~\sum_{i=1}^T~w_iE_i\longrightarrow E_i的加权均值\\
&\qquad\qquad\qquad\hat{A}~=~\sum_{i=1}^T~w_iA_i\longrightarrow A_i的加权~歧值
\end{align*}
$$
- **难以求解**：
   - 定义在 **整个样本空间**
   - $\hat{A}$ 不可直接操作
   - 推导只适用于 **Regression**学习 ， 难以直接推广到 **分类任务**

- <font color = 'green'>**多样性度量**</font>：

**定义**：估计个体学习器的 **多样化程度**$\longrightarrow$考虑个体分类器的 **两两相似性**/**不相似性**

<font color = 'lightgreen'><font size =4></font>**预测结果列联表**</font>：

||$h_i~=~+1$|$h_i~=~-1$|
|:----:|:----:|:----:|
|$h_j~=~+1$|a|c|
|$h_j~=~-1$|b|d|
$$
\begin{align*}
&其中：\\
&a~:~h_i与h_j~均预测为~正类~~（一致）\\
&b~:~h_i正类~,~h_j负类~~（不一致）\\
&c~:~h_i负类~,~h_j正类~~（不一致）\\
&d~:~h_i与h_j~均预测为~负类~~（一致）\\
\end{align*}
$$
<font color = 'lightgreen'><font size =4></font>**不合度量** *$dis_{ij}$*</font>：
$$
dis_{ij}~=~\frac{b+c}{m}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad dis_{ij}\in[0,1]\Longrightarrow dis_{ij}\uparrow，多样性\uparrow
\end{align*}
$$
<font color = 'lightgreen'><font size =4></font>**相关系数** *$\rho_{ij}$*</font>：
$$
\rho~=~\frac{ad~-~bc}{\sqrt{(a+b)(a+c)(c+d)(b+d)}}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad \rho_{ij}\in[-1,1]\Longrightarrow \left\{\begin{aligned}0~,&\qquad 无关\\(0,1]~,&\qquad 正相关\\ [-1,0)~,&\qquad 负相关\end{aligned}\right.\end{align*}
$$

<font color = 'lightgreen'><font size =4></font>**Q-统计量** *$Q_{ij}$*</font>：
$$
Q_{ij}~=~\frac{ad-bc}{ad+bc}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad Q_{ij}\geq\rho_{ij} ~,~与\rho_{ij}同号
\end{align*}
$$
<font color = 'lightgreen'><font size =4></font>**$\mathcal{k}$-统计量** </font>：
$$
\mathcal{k}~=~\frac{p_1-p_2}{1-p_2}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad p_1~:~两分类器~"一致"~的概率\Longrightarrow p_1=\frac{a+d}{m}\\
&\qquad\qquad P_2~:~两分类器~"偶然达成一致"~的概率\Longrightarrow p_2=\frac{(a+b)(a+c)+(c+d)(b+d)}{m^2}\\
\end{align*}
$$

<font color = 'lightgreen'><font size =4></font>**$\mathcal{k}$-误差图** </font>：

- **定义**：
   - **横坐标** ：学习器的$\mathcal{k}$值
   - **纵坐标** ：平均误差

- **特点**：
   - 数据点云位置 $\uparrow$ ,个体学习器准确性 $\downarrow$
   - 数据点云位置 $\rightarrow$ ,个体学习器多样性 $\downarrow$
   - 数据云 **偏左下** 更 好而不同！！
   - ---
- <font color = 'green'>**多样性增强**</font>：
在学习过程中引入 **随机性**，可以同时使用多个！

<font color = 'lightgreen'><font size =3></font>**数据样本扰动** 
</font>：

**定义**： 从给定的数据集D中 **产生** **不同的** 数据子集

方法：
   - <font color = 'lightgreen'>**采样法**</font>：
      - **自主采样法**：Bagging
      - **序列采样法**：AdaBoost

适用：**不稳定**の基学习器  （对数据扰动敏感）

<font color = 'lightgreen'><font size =3></font>**输入属性扰动** </font>：

**定义**：从不同的 **子空间**（属性空间）训练个体学习器

方法：
   - <font color = 'lightgreen'>**随即子空间算法**</font>

适用：包含大量 **冗余属性** 数据

<font color = 'lightgreen'><font size =3></font>**输出表示扰动** </font>：

**定义**：对输出表示 进行操作操作 $\longrightarrow$ 对训练样本的类标记稍作变动

方法：
   - <font color = 'lightgreen'>**翻转法**</font>$$~随机改变~一些训练样本的标记$$
   - <font color = 'lightgreen'>**输出调制法**</font>$$将分类输出~~\underset{\longrightarrow}{转换}~ ~回归输出~后构造个体学习器$$
   - <font color = 'lightgreen'>**ECOC法**</font>$$利用~"纠错输出码"~\Longrightarrow '~多分类任务~=~\underset{i=1}{\stackrel{m}{\sum}}二分类任务~'$$


<font color = 'lightgreen'><font size =3></font>**算法参数扰动** </font>：

**定义**：随即设置不同的参数/环节

方法：
   - <font color = 'lightgreen'>**负相关法**</font>：
      - **多参数** 时：**显示**通过**正则化项** 强制参数个数
      - **少参数** 时：将**属性选择机制**替换为其他方法

# <font color ='red'>**AdaBoost(Boosting族)**</font><font size = 3>"串行"代表</font>
<font size =4><font color ='red'>**Boosting族**</font></font>
- <font color='green'>**要求**</font>：基学习器能对特定的 **数据分布** 进行学习

- <font color='green'>**特点**</font>：
   - 使用 **贪心法**最小化损失函数
   - 每个模型：弱模型 + **偏差高** + 方差低（1）
   - $\Updownarrow$
   - 模型 **相关性**强$\longrightarrow$不能显著降低 **方差**
   - 注重 **降低偏差**$\longrightarrow$基于泛化性能较弱的**弱学习器**构造出 **很强的**集成

<font size=4><font color = 'lightgreen'>**赋权法** *re-weighting*</font></font>

- **思想**：在训练过程的 **每一轮**中，根据 **样本分布**为每一个**训练样本**重新赋予一个 **权重**$$\Updownarrow \begin{align*}&对于无法接受\\&带权样本的基学习算法\end{align*}$$

<font size=4><font color = 'lightgreen'>**重采样法** *re-sampling*</font></font>

- **思想**：
   - 在训练过程的 **每一轮**中，根据样本分布对训练集重新进行采样
   - 使用 重采样所得的样本集对基学习器进行训练

- **优点**：
   - 可获得 **重启动** 的机会 $\longrightarrow$ 避免训练过程中 **过早停止**
- <font color='green'>**指数损失函数（优化目标）**</font>：
$$
\begin{align*}
 \mathcal{l_{exp}(H~|~D)}~&=~\mathbb{E}_{\mathcal{x\thicksim D}}~[e^{-~f(x)~H(x)}]\\
&=\underset{x\in D}{\sum}~\mathcal{D(x)}~e^{-~f(x)H(x)}\\
&=\underset{i=1}{\stackrel{|D|}{\sum}}~\mathcal{D(x)}~·(~e^{H(x_i)}~\mathbb{I}(f(x_i)~=~1)+e^{H(x_i)}\mathbb{I}(f(x_i)=-1)~)\\
&=\underset{i=1}{\stackrel{|D|}{\sum}}~(~e^{H(x_i)}P(f(x_i)=1~|~x_i)+e^{H(x_i)}P(f(x_i)~=~-1~|~x_i))
\end{align*}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad f(x)~:~真实函数   \\
&\qquad\qquad \mathcal{D(x_i)}:在数据集D中进行一次随机抽样，样本x_i被抽中的概率\\
&\qquad\qquad \mathbb{I}(f(x_i)=1):使得f(x_i)=1的样本x_i
\end{align*}

$$
**二分类问题**：
$$
\begin{align*}
分类错误&率\epsilon：\\
&\epsilon ~=~ P(h_i(x)~\neq~f(x))  
\end{align*}
$$
若基分类器之间**相互独立**：
$$
\begin{align*}
\epsilon~&=~\underset{k=0}{\stackrel{\llcorner\frac{T}{2}\lrcorner}{\sum}}~\left(\begin{array}{c}T\\ k\end{array}\right)~(1-\epsilon)^k\epsilon^{T-k}\\
&\leq exp~(-\frac{1}{2}~T~(1-2\epsilon)^2)
\end{align*}\longrightarrow \begin{align*}&当T不断增大,\\ &\epsilon将以指数级下降,\\&最终趋于零
    
\end{align*}
$$
<font size =4><font color ='red'>**AdaBoost**---二分类</font></font>
- <font color='green'>**模型构成**</font>：

<font size=4><font color = 'lightgreen'>**加性模型** *a dditive model*</font></font>
$$
\begin{align*}
&基学习器的线性组合：\\
&\qquad\qquad\qquad H(x)~=~\underset{t=1}{\stackrel{T}{\sum}}~\alpha_t h_t(x)\\
\end{align*}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad T~:~训练迭代次数   \\
&\qquad\qquad h_t(x)~:~t代产生的~基学习器\\
&\qquad\qquad \alpha_t~:~h_t(x)的~加权系数  \\
\end{align*}
$$

- <font color='green'>最小化**指数损失函数（优化目标）**</font>：
$$
\begin{align*}
 \mathcal{l_{exp}(H~|~D^t)}~&=~\mathbb{E}_{\mathcal{x\thicksim D_t}}~[e^{-~f(x)~H(x)}]\\
&=~\mathbb{E}_{\mathcal{x\thicksim D_t}}~[e^{-~f(x)~\alpha_t h_t(x)}]\\
&=~\mathbb{E}_{\mathcal{x\thicksim D_t}}~[e^{-\alpha_t}\mathbb{I}(f(t)~=~h_t(x))~+~e^{\alpha_t}\mathbb{I}(f(x)\neq h_t(x))]\\
&=~e^{\alpha_t}~P_{x\thicksim D_t}(f(x)=h_t(x))~+~e^{\alpha_t}P_{x\thicksim D_t}(f(x)\neq h_t(x))\\
&=~e^{-\alpha_t}(1-\epsilon_t)~+~e^{\alpha_t}\epsilon_t
\end{align*}\\
\Updownarrow
$$
**指数损失函数** = 替代损失函数 （达到 <font color='lightgreen'>**贝叶斯最有错误率**</font>）
- <font color='green'>**参数 $\alpha_t$ 求解**</font>：
$$
\begin{align*}
&令导数\frac{\partial~\mathcal{l_{exp}(~\alpha_th_t~|~D_t~)}}{\partial\alpha_t}=0:\\    
&\\
&\qquad\qquad -e^{\alpha_t}(1-\epsilon_t)~+~e^{\alpha_t}\epsilon_t~=~0\\
&\\
&\qquad\qquad\Longrightarrow\alpha_t~=~\frac{1}{2}ln(\frac{1-\epsilon_t}{\epsilon_t})\qquad\quad(对数几率)
\end{align*}
$$
- <font color='green'>**算法流程**</font>：
   - Step 1 ：**初始化** 样本分布$D_0(x)$ $$\mathcal{D_1(x)}=\frac{1}{m}$$
   - Step 2：循环**迭代 T 代**$$\begin{align*}for t=1,2,..T then:\\&<1.~基于分布D_t从数据集D中训练出分类器h_t\\&<2.~估计h_t误差~\epsilon_t : \longrightarrow h_t是否为弱分类器\\&\qquad\qquad\qquad\qquad\qquad 如果~\epsilon_t\geq0.5\rightarrow 舍弃h_t\qquad return\\&\\&<3.确定分类器~h_t~の~权重:\\&\qquad\qquad\alpha_t~=~\frac{1}{2}~ln(\frac{1-\epsilon_t}{\epsilon})\\&\\&<4.~更新样本分布D_{t+1}(x) :\\&\qquad D_{t+1}(x)~=~\frac{D_t(x)}{Z_t}\times\left\{\begin{aligned}exp(-\alpha_t)~,&\qquad if h_t(x)~=~f(x)\\exp(\alpha_t)~,&\qquad if h_t(x)\neq f(x)\end{aligned}\right. \end{align*}$$

   - Step 3:输出 **基分类器**の **线性组合**：$$H(x)~=~sign(~\sum_{t=1}^T\alpha_th_t(x)~)$$

# <font color ='red'>**Bagging**</font></font><font size = 3>"并行"代表</font>
- <font color='green'>**特点**</font>：
   - 个体学习器之间 **不存在  强依赖关系**$\longrightarrow$可以降低 **方差**
   - **时间复杂度** 低
   - 可直接用于 **多分类 Multi-Classifier** / **回归Regression**等任务

- <font color='green'>**方法**</font>：
   - **样本**：<font color = 'lightgreen'>**自助采样法**</font>：
   - **分类选择**：
      - Classifier : <font color = 'lightgreen'>**投票法**</font>
      - Regression : <font color = 'lightgreen'>**平均法**</font>

- <font color='green'>**计算复杂度** *O(·)*</font>：
$$
\begin{align*}
&基学习器の计算复杂度:\\
&\qquad\qquad\qquad\qquad\qquad O(m)\\
&Baggingの计算复杂度:\\
&\qquad\qquad\qquad\qquad\qquad T(O(m)~+~O(s))\qquad\qquad(O(s)一般很小)
\end{align*}

$$

- <font color='green'>**特点**</font>：

<font size=4><font color = 'lightgreen'>**包外估计** *out-of-bag estimate*</font></font>

**定义**：对于样本数据集 **留下** 一定的样本(约36.8%)作为 **验证集**$\longrightarrow$进行包外估计

**作用**：
   - Decision Tree:
      - 辅助剪枝
      - 估计各结点的后验概率
   - Neural Networks：
      - 辅助早期停止$\longrightarrow$  减小**过拟合**风险
# <font color ='red'>**随机森林** *RF*</font></font><font size = 3>"并行"代表   *Radom Forest*</font>
- <font color='green'>**特点**</font>：
   - Bagging的 **拓展**：
      - Bagging ：**确定性型**DecisionTree
      - RF ：**随机型**DecisionTree $\longrightarrow$ 效率更优
   - 算法**简单** + **易实现** + 计算**开销小**
   - 性能强大$\longrightarrow$  "**代表 集成学习** 技术水平 的方法"
   - 基学习器の **多样性** 通过 **样本扰动**+**属性扰动**来实现

- <font color='green'>**模型构成**</font>：
   - **基学习器**：决策树 
   - **集成方法**：Bagging集成
   - **特点**：DecisionTree训练过程中引入 **随机属性**の选择