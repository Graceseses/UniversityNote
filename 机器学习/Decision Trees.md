# <font color ='red'>**基本概念**</font>

<font size = 4><font color='lightgreen'>**纯度**</font></font>

结点**定义**：决策树的分支节点所包含的**样本**尽可能地**来自**于**同一类别**

<font size = 4><font color='lightgreen'>**信息熵**   *Ent(D)*</font></font>

**定义**：样本集合D的信息熵
$$
Ent(D) = - \underset{k=1}{\stackrel{|\mathcal{Y}|}{\sum}}p_k~log_2p_k\\
\begin{align*}
&\\
&其中：\\
&\qquad D ：样本集合\\
&\qquad p_k: D中第k类样本的比例   
\end{align*}
$$


**特点**：
   - Ent(D)越小，纯度越高
   - **度量**样本集合**纯度**最常用的一个**指标**
   - 对 **取值数目较多**的属性有**偏好**

**结论**：
   - $Ent(D)\downarrow~,~纯度\uparrow$
   - $0\leq Ent(D)\leq log_2|\mathcal{Y}|$

<font size = 4><font color='lightgreen'>**信息增益**  *Gain(D;a)*</font></font>

**定义**：使用属性a进行样本划分时所获得的纯度提升
$$
Gain(D,a) = Ent(D) - \underset{v=1}{\stackrel{\mathcal{V}}{\sum}}\frac{|D^v|}{|D|}Ent(D^v)
$$
其中：
   - a :  **属性**a
   - $a^v : 离散属性的\mathcal{V}$ 种**可能取值**${a^1,a^2,...,a^V}$
   - $\mathcal{Y}$: 样本**类别数**
   - $D^v$:属性a上取值为$a^v$的**样本个数**

**特点**：
   - 信息增益越大，属性a划分纯度提升的越大
   - 可作为 决策树**属性划分**的准则

<font size = 4><font color='lightgreen'>**增益率**  *Grain_ratio(D;a)*</font></font>

**定义**：属性a 在 数据集D 上的增益率
$$
Gain\_ ratio(D,a) = \frac{Gain(D,a)}{IV(a)}
$$
其中<font size = 4><font color='lightgreen'>属性a的 **固有值**     *IV(D,a)*</font></font>：
$$
IV(a) = - \underset{v=1}{\stackrel{\mathcal{V}}{\sum}}\frac{|D^v|}{|D|}log_2\frac{|D^v|}{|D|}
$$
**特点**：
   - a属性 の 取值数目 越多，IV(a)越大
   - $\qquad\qquad\qquad\Updownarrow$
   - 增益率 对**取值数目较少**的属性有所**偏好**
<font size = 4><font color='lightgreen'>**基尼值**  *Gini(D)*</font></font>

**定义**：数据集D的基尼值（纯度）
$$
\begin{align*}
Gini(D) &= \underset{k=1}{\stackrel{|\mathcal{Y}|}{\sum}}\underset{k'\neq k}{\sum}~p_k~p_{k'}\\
&= 1-\underset{k=1}{\stackrel{|\mathcal{Y}|}{\sum}}~p_k^2
\end{align*}
$$
<font size = 4><font color='lightgreen'>**基尼指数**  *Gini_index(D;a)*</font></font>

**定义**：属性a的基尼指数
$$
Gini\_ index(D,a)=\underset{v=1}{\stackrel{\mathcal{V}}{\sum}}\frac{|D^v|}{|D|}Gini(D^v)
$$
**特点**：
   - $Geni(D)\downarrow~,~Dの纯度\uparrow$

<font size = 4><font color='lightgreen'>**决策树桩**</font></font>

**定义**：一颗仅含有**一层划分**的决策树

# <font color ='red'>**基本算法流程**</font>
1. <font color='lightgreen'>**分而治之**</font>策略：

输入：
$$
\begin{align*}
&训练集 ： D = {(x_1,y_1) , (x_2 , y_2) ,....,(x_m,y_m)}   \\
&属性集 ： A = {a_1,a_2,...,a_d}
\end{align*}
$$
Step 1：<font color='green'>**生成节点node**</font>

Step 2：<font color='green'>两个**if判断**</font>
$$
\begin{align*}
&if D中地样本全部属于同一类别C\equiv"无需划分" then\\
&\qquad 将node标记为类别C\\    
&\\
&if (~A=\phi~~or~~D中样本在A上地所有属性划分中取值相同~)\equiv "无法划分" then\\
&\qquad 将node标记为叶节点，node类别标记为D中样本数最多的C_{max-number}
\end{align*}
$$
Step 3:<font color='green'>从A中选择**最有划分属性$a_*$**</font>

Step 4：<font color='green'>**递归返回**</font>
$$
\begin{align*}
&for~a_*^v\in a_*:\\
&\qquad 为node生成一个分支\\
&\qquad if D_v =\phi ('无法划分')~~then\\
&\qquad\qquad\qquad node标记为叶节点，node类别标记为D_v中最多的类\\
&\qquad\qquad\qquad return  '递归返回'\\
&\qquad else\\
&\qquad\qquad\qquad 从属性集A中去除~a_*\\
&end！\\

&\\
&其中：\\
&D_v : D中在a_*上取值为~a_*^v~の样本子集
\end{align*}
$$
Step 5：<font color='green'>**输出决策树**</font>

<font color='green'>**递归返回**条件</font>：

   - **无需划分**:当前结点所包含的样本完全来自于同一类别
   - **无法划分**：
       - 当前属性集为空
       - 所有样本在所有属性上取值相同
   - **不能划分**：当前结点包含的样本集为空

<font color='green'>递归返回**标记**</font>：
   
   - **无需划分**：属于哪类归哪类
   - **无法划分**：归为**该结点**所包含**样本最多的类别**
   - **不能划分**：**父结点**所含**样本最多**地类别

# <font color ='red'>**属性划分原则**</font>
**离散属性**：

<font size = 4><font color='lightgreen'>**ID3决策树**</font></font>

划分准则：**信息增益**
$$
a_* = \underset{a\in A}{argmax}~Gain(D,a)
$$

<font size = 4><font color='lightgreen'>**C4.5决策树**</font></font>

划分准则：
   - Step 1：从候选划分属性A中找出**信息增益**水平高于平均水平地属性，记为$\hat{A}$
   - Step 2:从$\hat{A}$中选出**增益率**最高的
$$
a_*= \underset{a\in \hat{A}}{argmax}~Gain\_ ratio(D,a)
$$

<font size = 4><font color='lightgreen'>**CART决策树**</font></font>

划分准则：**基尼指数**
$$
a_* = \underset{a\in A}{argmax}~Gini\_ index(D,a)
$$

- ---

**连续属性**：

<font size = 4><font color='lightgreen'>**C4.5决策树**</font></font>

划分准则：**二分法**

# <font color ='red'>**剪枝处理**</font>

1. **本质** : <font color='red'>**贪心**</font>本质

2. **意义**：主动剪枝以降低 **过拟合** の风险

3. **基本策略**：

<font size = 4><font color='lightgreen'>**预剪枝**</font></font>

**定义**：
   - 决策树的**生成过程中**
   - 对每个结点在**划分前**先进行剪枝**前后**Tree**泛化性能估计**

**优点**：
   - 降低了 **过拟合**的风险
   - 显著减少Tree的训练与测试**时间开销**

**缺点**：
   - 具有 **欠拟合**的风险

<font size = 4><font color='lightgreen'>**后剪枝**</font></font>

**定义**：
   - 先生成一颗**完整**の**决策树**
   - **自底向上**对于 **非叶结点**  **逐一** 进行**剪枝前后**Tree**泛化性能评估**

**优点**：
   - 降低了 **过拟合**の风险
   - **欠拟合**风险小$ \longrightarrow$  泛化性能往往优于 **预剪枝**の

**缺点**：
   - 训练**时间开销**加大

1. **估计标准**：
$$
是否剪枝=\left\{\begin{aligned}
剪枝&\qquad 决策树泛化性能提高\\
不剪枝&\qquad 决策树泛化性能不变/降低
\end{aligned}\right.
$$
判断 **泛化性**：是否提高？：

<font size = 4><font color='lightgreen'>**留出法**</font></font> : 预留一部分数据用作 **验证集**

# <font color ='red'>**连续值处理**</font>
<font size = 4><font color='lightgreen'>**二分法**---离散化</font></font>

首先假设：
$$
\begin{align*}
连续属性a &: \{a^1,a^2,...,a^n\}  \longrightarrow 从小到大の顺序排序集合！\\
t&:划分点\\
D_t^+ &：包含在a上 且\leq t的样本\\
D_t^- &: 包含在a上 且 >t 的样本\\
T_a &:包含n-1个元素的候选划分点集合\\
&\qquad T_a=\{\frac{a^i+a^{i+1}}{2}|1\leq i\leq n-1 \} \longrightarrow 区间中位点作为 候选划分点
\end{align*}
$$

<font color='green'>**属性划分**----</font><font size = 4><font color='lightgreen'>**改良信息增益**</font></font>
：

**逐一**~对a中的属性点带入计算Gain(D,$a_t$),**比较**得出取得 max 时的  $\hat{t}$
$$
\begin{align*}
Gain(D,a) &= \underset{t\in Ta}{max}Gain(D,a,t)\\
&=\underset{t\in T_a}{max} (Ent(D)- \underset{\lambda\in\{-,+\}}{\sum}\frac{|D_t^\lambda|}{|D|}Ent(D_t^\lambda))
\end{align*}
$$
**特点**：
  - 属性可**继续**作为 后代结点の划分属性

# <font color ='red'>**缺失值处理**</font>
1. **处理思想**：
   - 如何在 **属性值缺失**的情况下进行**划分属性选择**？
   - 给定划分属性，若**样本**在该属性上**值缺失**，如何对**样本**进行**划分**
2. **不处理的后果**：
   - 造成数据信息极大浪费

3. <font color='green'>**属性划分**----</font><font size = 4><font color='lightgreen'>**改良信息增益**</font></font>：
为每一个样本x **赋予权重**$w_x$
$$
假设:\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\\
\begin{align*}
&\\
\mathcal{Y}&:样本类别\\
D &:训练集\\
a &:属性集 , 有\mathcal{V}个取值 \{a^1,a^2,...,a^{\mathcal{V}}\}\\
\tilde{D} &:在属性a上没有缺失值的样本子集\\
\tilde{D^v}&:\tilde{D}在属性a上取值为a^v的样本子集\\
\tilde{D_k} &: 属于第k类的样本子集\\
w_x&:每个样本x的权重系数\\
\rho&:无缺失值样本所占比例\\
\tilde{r_v}&：无缺失值样本在属性a上取值为a^v的样本所占比例\\
\tilde{p_k}&:无缺失值样本中第k类所占比例\\
&\\
\end{align*}
\\
\begin{align*}
&\rho = \frac{\sum_{x\in\tilde{D}}·w_x}{\sum_{x\in D}·w_x}\\
&\\
&\tilde{r_v} = \frac{\sum_{x\in\tilde{D_k}}·w_x}{\sum_{x\in\tilde{D}}·w_x}\qquad,~\sum_{v=1}^{|\mathcal{V}|}~\tilde{r_v}=1~~~(1\leq v \leq |\mathcal{V}|)\\
&\\
&\tilde{p_k}=\frac{\sum_{x\in\tilde{D_k}}·w_x}{\sum_{x\in \tilde{D}}·w_x}\qquad,~\sum_{k=1}^{|\mathcal{Y}|}~\tilde{p_k}=1~~~(1\leq k \leq |\mathcal{Y}|)\\
\end{align*}
$$
</font><font size = 4><font color='lightgreen'>**改良信息增益**</font></font>：
$$
\begin{align*}
Gain(D,a)&=\rho\times Gain(\tilde{D},a)\\
&=\rho\times(~Ent(\tilde{D})-\underset{v=1}{\stackrel{\mathcal{V}}{\sum}}~\tilde{r_v}~Ent(\tilde{D^v})~)\\
\end{align*}\\
\begin{align*}
&其中:\\
&Ent(\tilde{D})=-\underset{k=1}{\stackrel{|\mathcal{Y}|}{\sum}}    
\end{align*}
~\tilde{p_k}~log_2\tilde{p_k}
$$

4. <font color='green'>**类别化分**</font>：
   - 样本取值 **已知**：
      - x $~\longrightarrow~$ 取值对应の子节点
      - 样本权值$w_x$不变
   - 样本取值 **未知**：
      - x $~\longrightarrow~$ 划入所有子节点   $\Longrightarrow$让同一个样本x以 **不同概率**划入不同の子节点中去
      - 样本权值 $w_x$ 调整为 $\tilde{r_v}·w_x$ 

# <font color ='red'>**多变量决策树**</font>

1.**特点**：
||单变量决策树|多变量决策树|
|:----:|:----:|:----:|
|分类边界|**轴平行**|**倾斜划分**|
|划分属性|单属性|属性的**线性组合**|
|学习目的|寻找**最优划分属性**|建立一个合适的**线性分类器**|
|
2. **线性分类器**：
$$
分类边界：
-a\leq \underset{i}{\stackrel{d}{\sum}}w_i~a_i \leq b\\
线性分类器： \underset{i=1}{d}w_i~a_i~=~t\\
\begin{align*}
&\\
&其中：\\  
&\qquad w_i:属性a_i的权重\\
&\qquad t  :常数\\
&\qquad a,b为上下限\\ 
&\\
&w_i,t均由模型学习
\end{align*}
$$

3. **经典算法**：

</font><font size = 4><font color='lightgreen'>**OC1算法**</font></font>