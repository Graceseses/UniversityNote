# <font color ='red'>**基本概念**</font>
### <font color = 'lightgreen'>**聚类** *Clustering*</font>：
- <font color='green'>**归属**</font>： **无监督**学习$\longrightarrow$**自动** 形成 **簇结构**
- <font color='green'>**目标**</font>：将数据集中的样本 划分为 若干 **通常不相交** の子集
- <font color='green'>**特点**</font>：
   -  **独立性**：可做为一个 单独の过程
   -  **复合性**：可以作为其他分类任务の **前驱过程**

- <font color='green'>**原则**</font>：<font color = 'lightgreen'><font size =4>**物以类聚**</font></font>$$物以类聚=\left\{\begin{align*}&"簇内相似度"~高\\&"簇间相似度"~低\end{align*}\right.$$


### <font color = 'lightgreen'>**簇** *Cluster*</font>：
**定义**： 不相交 の子集
### <font color = 'lightgreen'>**有效性指标** *validity index*</font>：
- **定义**：聚类的 **性能度量** ， 评估模型好坏
- **种类**：
   - <font color = 'lightgreen'><font size =4>**外部指标**：</font></font> 与某个 **参考模型**进行比较
   - <font color = 'lightgreen'><font size =4>**内部指标**：</font></font> 直接考察 **聚类结果** ，不利用任何参考模型

- <font color='green'>**外部指标**</font>：

规定：
$$
\begin{align*}
&D~=\{x_1,x_2,...,x_m\}~:~数据集\\
&\mathcal{C}=\{C_1,C_2,...,C_k\}:聚类の簇化分\\
&\mathcal{C^*}=\{C_1^*,C_2^*,....,C_k^*\}：参考模型の簇化分\\
&\\
&\qquad\qquad\qquad ~a~=~|SS|,\qquad SS=\{(x_i,x_j)~|~\lambda_i=\lambda_j~,\lambda_i^*=\lambda_j^*~,i<j\}\\
&\qquad\qquad\qquad ~b~=~|SD|,\qquad SD=\{(x_i,x_j)~|~\lambda_i=\lambda_j~,\lambda_i^*\neq\lambda_j^*~,i<j\}\\
&\qquad\qquad\qquad ~c~=~|DS|,\qquad DS=\{(x_i,x_j)~|~\lambda_i\neq\lambda_j~,\lambda_i^*=\lambda_j^*~,i<j\}\\
&\qquad\qquad\qquad ~d~=~|DD|,\qquad DD=\{(x_i,x_j)~|~\lambda_i\neq\lambda_j~,\lambda_i^*\neq\lambda_j^*~,i<j\}
\end{align*}
$$
### <font color = 'lightgreen'>**Jaccard系数** *JC*  </font>  Jaccard Coefficient：
$$
JC~=~\frac{a}{a+b+c}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad JC\in[0,1]~\And~JC\uparrow,性能\uparrow (正比关系)
\end{align*}
$$
### <font color = 'lightgreen'>**FM指数** *FMI*  </font>  Fowlkes and Mallows Index：
$$
FMI~=~\sqrt{\frac{a}{a+b}·\frac{a}{a+c}}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad FMI\in[0,1]~\And~FMI\uparrow,性能\uparrow (正比关系)
\end{align*}
$$
### <font color = 'lightgreen'>**Rand指数** *RI*</font>   Rand Index：
$$
RI~=~\frac{2(a+d)}{m(m-1)}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad RI\in[0,1]~\And~RI\uparrow,性能\uparrow (正比关系)
\end{align*}
$$

- <font color='green'>**内部指标**</font>：

规定：
$$
\begin{align*}
&D~=\{x_1,x_2,...,x_m\}~:~数据集\\
&\mathcal{C}=\{C_1,C_2,...,C_k\}:聚类の簇化分\\
&\mu=\frac{1}{|C|}\sum_{1\leq i<j\leq|C|}dist(x_i,x_j)~:~簇C的中心点
&\\
&\\
&\qquad\qquad\qquad 簇C的平均距离~：~~~~~~~~~~~~~~~~~~~avg(C)~=~\frac{2}{|C|(|C|-1)}\sum_{1\leq ~~i<j\leq|C|}dist(x_i,x_j)\\
&\qquad\qquad\qquad 簇C的最远距离~：~~~~~~~~~~~~~~~~~~~diam(C)~=~max_{1\leq i<j\leq|C|}dist(x_i,x_j)\\
&\qquad\qquad\qquad 簇C_i与C_j之间的最近距离~：~~~d_{min}(C_i,C_j)~=~min_{1\leq i<j\leq|C|}dist(x_i,x_j)\\
&\qquad\qquad\qquad 簇C_i与C_j中心点间的距离~：~~~d{cen}(C_i,C_j)~=~dist(\mu_i,\mu_j)\\
\end{align*}
$$
### <font color = 'lightgreen'>**DB指数** *DBI*</font>  Davies-Bouldin Index：
$$
DBI~=~\frac{1}{k}\underset{i=1}{\stackrel{k}{\sum}}\underset{i\neq j}{max}(\frac{avg(C_i)+avg(C_j)}{d_{cen}(C_i,C_j)})\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad~DBI\downarrow , 性能\uparrow (反比关系)
\end{align*}
$$
### <font color = 'lightgreen'>**Dunn指数** *DI*</font>   DunnIndex：
$$
DI~=~\underset{1\leq i\leq k}{min}\{~\underset{i\neq j}{min}~(~\frac{d_{min}(C_i,C_j)}{max_{1\leq l\leq k}diam(C_l)}~)~\}\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad ~DI\uparrow,性能\uparrow (正比关系)
\end{align*}
$$
### <font color = 'lightgreen'>**连续属性** *continuous attribute*</font>：
**定义**：在定义域上有 **无穷多个可能取值**の属性集
### <font color = 'lightgreen'>**离散属性** *categorical attribute*</font>：
**定义**:在定义域上有 **有限个可能取值**の属性集
### <font color = 'lightgreen'>**有序属性** *ordinal attribute*</font>：
**定义**：可以**直接**在**属性值**上进行**数值计算**

### <font color = 'lightgreen'>**无序属性** *non-ordinal attribute*</font>：
**定义**：**无法直接**在属性值上进行数值运算


### <font color = 'lightgreen'>**距离**   *dist(·,·)* </font> *distance*：
- <font color='green'>**定义**</font>：描述数据集两点之间的差距/关系
- <font color='green'>**性质**</font>：
   - **非负性**：$$dist(x_i,x_j)\geq0$$
   - **同一性**：$$dist(x_i,x_j)=0~,~~当且仅当~x_i=x_j时$$
   - **对称性**:$$dist(x_i,x_j)~=~dist(x_j,x_i)$$
   - **直递性**：$$dist(x_i,x_j)\leq dist(x_i,x_k)+dist(x_k,x_j)$$

<font color = 'lightgreen'>**非度量距离**</font> : 不满足度量**基本性质**的**距离**


- <font color='green'>**距离度量**</font>：
**有序属性**
### <font color = 'lightgreen'>**闵可夫斯基距离** *$dist_{mk}(x_i,x_j)$*</font> 
$$
\begin{align*}
&假设~x~为~n维向量：\\
&\qquad\qquad\qquad dist_{mk}(x_i,x_j)~=~(~\underset{u=1}{\stackrel{n}{\sum}}|x_{iu}~-~x_{ju}|^p~)^\frac{1}{p}\\
&\qquad\qquad\qquad\qquad\qquad\qquad\Updownarrow   
\end{align*}

$$
- <font color = 'lightgreen'>**欧氏距离** *Euclidean Distance*</font>：

当 **p=2** 时：
$$
dist_{ed}(x_i,x_j)~=~||x_i-x_j||_2~=~\sqrt{\underset{u=1}{\stackrel{n}{\sum}}|x_{iu}-x_{ju}|^2}
$$
- <font color = 'lightgreen'>**曼哈顿距离** *Manhattan Distance*</font>：

当 **p=1** 时：
$$
dist_{man}(x_i,x_j)~=~||x_i-x_j||_1~=~\underset{u=1}{\stackrel{n}{\sum}}|x_{iu}-x_{ju}|
$$
**无序属性**:
### <font color = 'lightgreen'>**VDM距离** *$dist_{p}(a,b)$*</font> 
$$
VDM_p(a,b)~=~\underset{i=1}{\stackrel{k}{\sum}}|~\frac{m_{u,a,i}}{m_{u,a}}~-~\frac{m_{u,b,i}}{m_{u,b}}~|^p
$$
**混合属性**
### <font color = 'lightgreen'>**加权距离** *$MinkovDM_{p}(x_i,x_j)$*</font> 
$$
\begin{align*}
&假设：\\
&\qquad\quad [0,n_c]  ~:~有序属性\\
&\qquad\quad (n_c,n]~:~无序属性\\
&\\
&\quad\quad\quad\quad MinkovDM_p(x_i,x_j)~=~(\underset{u=1}{\stackrel{n_c}{\sum}}~|x_{iu}-x_{ju}|^p~+~\underset{u=n_c+1}{\stackrel{n}{\sum}}~VDM_p(x_{iu},x_{ju}))^\frac{1}{p}
\end{align*}
$$
- <font color='green'>**作用**</font>："距离" $\longrightarrow$ 定义数据之间の **相似度**
# <font color ='red'>**原型聚类 - K均值算法** </font><font size =3>*K-means*</font>
- <font color='green'>**算法思想**</font>：对簇$\mathcal{C}={C_1,C_2,...,C_k}$**最小化均方误差** 
$$
min\{~E~\}~=~min \{ ~\underset{i=1}{\stackrel{k}{\sum}}\underset{x\in C_i}{\sum}~||x-\mu_i||^2 ~\}
$$
E值越高，**相似度**越高
- <font color='green'>**算法流程**</font>：
   - Step 1：**初始化**----随机选取 k个初始均值向量$\{\mu_1,\mu_2,...,\mu_k\}$
   - Step 2：**迭代更新簇**
$$
\begin{align*}
&<1>.计算各样本距离dist_{ji}，确定样本标记\lambda_j\\
&\qquad\qquad\qquad \lambda_j~=~argmin_{i\in\{1,2,...,k\}}~d_{ji}\\
&<2>.~将每个样本分入~距离最近~の簇：\\
&\qquad\qquad\qquad C_{\lambda_j}~=~C_{\lambda_j}∪{x_j};\\
&<3>.~计算各个簇の~新均值向量：\\
&\qquad\qquad\qquad \mu_i = \frac{1}{|C_i|}~\sum_{x\in C_i}x\\
&\\
&<4>.~停止条件：\left\{\begin{align*}
&最大迭代次数T\\
&均值向量 均不再改变or变化极小
\end{align*}\right.
\end{align*}
$$
- <font color='green'>**k值选择**</font>：<font size=4><font color = 'lightgreen'>**Elbow法（手肘法）**</font></font>

**定义**： **迭代优化** ----  画图选取 **近似最优k**
$$= \left\{\begin{align*} &横坐标~：~k\\& 纵坐标~：~Objective~ Function~values~of~k\end{align*}\right.$$

# <font color ='red'>**学习向量量化** *LVQ*</font><font size =3>*Learning Vector Quantization*</font>
- <font color='green'>**模型假设**</font>：强行 **监督学习**
   - 假设 数据样本带有 **类别标记**
   - 学习过程中 利用样本的这些 **监督信息** 来 **辅助** 聚类

- <font color='green'>**算法流程**</font>：
$$
\begin{align*}
&假设：\\
&\qquad\qquad D~=~\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\}\\
&\qquad\qquad P~=~\{p_1,p_2,...,p_q\}\\
&\qquad\qquad T~=~\{t_1,t_2,...,t_q\}\rightarrow 各原型向量的类别标记
\end{align*}
$$
   - Step 1：**初始化**----确定一组 **原型向量**$\{p_1,p_2,...,p_q\}$  $$p_q~:~从标记为~t_q~的样本中~随机选择~一个作为原型向量$$
   - Step 2：**迭代更新簇**$$\begin{align*}&<1>.计算各样本x_i与各个原型向量p_j 之间的距离dist_{ij}，确定样本标记\lambda_j\\&\qquad\qquad\qquad d_{ij}~=~||x_i~-~p_j||_2\\&<2>.~寻找离x_i最近的原型向量p_{j^*}：\\&\qquad\qquad\qquad j^*~=~argmin_{i\in\{1,2,...,q\}}d_{ij}\\&<3>.~更新~原型向量：（⭐）\\&\qquad\qquad\qquad p'~=~\left\{\begin{align*}p_{j^*}~+~\eta·(x_i~-~p_j),&\qquad 向x_i靠拢\\p_{j^*}~-~\eta·(x_i~-~p_j),&\qquad向x_i远离\\\end{align*}\right.\Longrightarrow\eta:学习率，\eta\in(0,1)\\&\\&<4>.~停止条件：\left\{\begin{align*}&最大迭代次数~T\\&原型向量 ~不再更新~or~变化极小~\end{align*}\right.\end{align*}$$
   - Step 3:输出$\longrightarrow$ **最终の原型向量**
   - Step 4:对于任意样本x$~\Longrightarrow\in$**距离最近**の原型向量所代表的簇中

<font size=4><font color = 'lightgreen'>**Voronoi剖分**</font></font>

**定义**：对于算法每个**最终的原型向量**$p'_i\longrightarrow$都定义了一个与之相关的区域$R_i$：
$$
R_i~=~\{x\in\mathcal{X}~|~||x-p_i||_2\leq||x-p;_i||_2~,~i'\neq i\}
$$

# <font color ='red'>**高斯混合聚类** </font><font size =3>*Mixture-of-Gaussian*</font>
- <font color='green'>**模型特点**</font>：采用 **概率模型**来表达**聚类原型**

概率函数---<font size=4><font color = 'lightgreen'>**n维高斯分布**</font></font>：$$\begin{align*}x\thicksim \mathcal{N}(\mu,\Sigma):\\&\quad\quad p(x)~=~\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}~·e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}\\&\\&其中：\\&\qquad\qquad\qquad \mu~:~n维均值向量\\&\qquad\qquad\qquad \Sigma~:~n\times n对称正定矩阵\end{align*}$$

概率模型----<font size=4><font color = 'lightgreen'> **高斯混合分布**</font></font>：
$$
p_{\mathcal{M}}(x)~=~\underset{i=1}{\stackrel{k}{\sum}}\alpha_i·p(x|\mu_i,\Sigma_i)
$$
$$
\begin{align*}
&\\
&其中：\\
&\qquad\qquad\qquad \alpha_i~:~混合系数\Longrightarrow\alpha_i>0~\And~\sum_{i=1}^k\alpha_i=1
\end{align*}
$$

- <font color='green'>**算法流程**</font>：
   - Step 1：。。。


# <font color ='red'>**密度聚类 - DBSCAN** </font><font size =2>*Denstiy-Based Spatial Clustering of Applications with Noise*</font>
- <font color='green'>**特点**</font>：
   - 著名的 **密度聚类**算法
   - 基于一组 **领域参数**$\longrightarrow$ 刻画**样本分布**の**紧密程度**

- <font color='green'>**名词定义**</font>：
   - <font size=4><font color = 'lightgreen'> **$\epsilon$-领域**</font></font>：$$\begin{align*}&对于x_i\in D~,~其\epsilon -领域包含样本集D中与x_i距离大于\epsilon の样本集合:\\&\qquad\qquad\qquad\qquad\qquad N_{\epsilon}(x_i)~=~\{x_j\in D|dist(x_i,x_j)\}\end{align*}$$
   - <font size=4><font color = 'lightgreen'> **核心对象**</font></font>：$$\begin{align*}&x_i为核心对象：\\&\qquad\qquad x_i的\epsilon-领域中至少含有MinPts个样本\\&\qquad\qquad\qquad\qquad\qquad\qquad\qquad\Updownarrow\\&\qquad\qquad\qquad\qquad\qquad |N_{\epsilon}(x_i)|\geq MinPts\end{align*}$$
   - <font size=4><font color = 'lightgreen'> **密度直达**</font></font>：$$\begin{align*}&<1>.x_j\in E_{\epsilon}(x_i)\\&<2>.x_i为核心对象\end{align*}\Longrightarrow x_j由x_i密度直达$$
   - <font size=4><font color = 'lightgreen'> **密度可达**</font></font>：$$\begin{align*}&<1>.存在样本序列~{p_1,p_2,...,p_n}\left\{\begin{aligned}&p_1~=~x_i\\&p_n~=~x_j\end{aligned}\right.\\&\\&<2>.p_{i+1}由p_i~"密度直达"\end{align*}\Longrightarrow ~~~~~x_j由x_i密度可达$$
   - <font size=4><font color = 'lightgreen'> **密度相连**</font></font>：$$\begin{align*}存在x_k~~~\underset{\longrightarrow}{满足}~~~x_i与x_j均由x_k~"密度可达"\Longrightarrow x_i与x_j密度相连\end{align*}$$
   - <font size=4><font color = 'lightgreen'> **簇**</font></font>：（重新定义）$$由~"密度可达"~关系导出の~"最大"の~"密度相连"の样本集合C~(~C\in D~)\\\begin{align*}&\\&领域参数:(\epsilon~，~MinPts)\\&\\&基本性质：\\&\qquad\qquad<1>.连接性~:~ \left\{\begin{aligned}x_i\in C\\x_j\in C\end{aligned}\right.\Longrightarrow x_i与x_j密度相连\\&\\&\qquad\qquad<2>.最大性~:~\left\{\begin{aligned}&x_i\in C\\ &x_j由x_i密度可达\end{aligned}\right.\Longrightarrow x_j\in C\\\end{align*}$$
   - <font size=4><font color = 'lightgreen'> **噪声(异常样本)**</font></font>：$$D中~不属于~任何簇~の~样本$$

# <font color ='red'>**层次聚类 — AGNES** </font><font size =3>*Hierarchical clustering — AGglomerative NESting*</font>
<font color ='red'><font size =4> **层次聚类** </font></font>

- <font color='green'>**定义**</font>：
在不同层次对数据集进行划分$\longrightarrow$形成树形の聚类结构
- <font color='green'>**数据集 化分**</font>：
   -  **自底向上**-- 聚合策略
   -  **自顶向下**-- 分拆策略

<font color ='red'><font size =4> **AGNES** </font></font>

- <font color='green'>**定义**</font>：采用 **自底向上**聚合策略の层次聚类算法
- <font color='green'>**别名**</font>：
   - **单链接**算法
   - **全链接**算法
   - **均链接**算法
- <font color='green'>**算法思路**</font>：
   - Step 1 ：每个样本看作一个 **初始聚类簇** 
   - Step 2 ：迭代 —— 寻找**距离最近**の两个聚类簇，进行**合并**
   - Step 3 ：达到预设の聚类簇个数 —— 停止
- <font color='green'>**距离计算**</font>：
   - **最小距离**：$$d_{min}(C_i,C_j)~=~\underset{x\in C_i~,z\in C_j}{min}~dist(x,z)\qquad\qquad 两簇中最近两个样本决定$$
   - **最大距离**：$$d_{max}(C_i,C_j)~=~\underset{x\in C_i~,z\in C_j}{max}~dist(x,z)\qquad\qquad 两簇中最远两个样本决定$$
   - **平均距离**：$$d_{avg}(C_i,C_j)~=~\frac{1}{|C_i||C_j|}\underset{x\in C_i}{\sum}\underset{z\in C_j}{\sum}~dist(x,z)\qquad\qquad 两簇所有样本决定$$