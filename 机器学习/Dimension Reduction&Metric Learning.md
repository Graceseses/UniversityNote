# <font color ='red'>**基本概念**</font>
### <font color = 'lightgreen'>**懒惰学习** *lazy learning*</font>：
**定义**：
   - 训练过程中仅仅将样本保存起来
   - 训练时间开销为零

### <font color = 'lightgreen'>**急切学习** *eager learning*</font>：
**定义**：
   - 训练阶段就对样本进行学习处理


# <font color ='red'>**KNN**</font></font> <font size =3>*K-Nearest Neighbor*</font>
- <font color='green'>**特点**</font>：
   - **监督学习**
   - **懒惰学习**：没有显示的训练过程，仅保存数据样本
- <font color='green'>**算法流程**</font>：
   - 确定 **D**（训练样本）+ **k**（重要参数） + **dist**（距离度量方式）
   - 预测：寻找训练集D中距离 测试样本**最近**の **k个样本**，使用 **投票法**/**平均法** 确定 **分类结果**
- <font color='green'>**k参数の选取**</font>：**作图法**
   - **横坐标**： kの连续取值  
   - **纵坐标**：训练误差error-rate

# <font color ='red'>**多维缩放算法(低维嵌入)** *MDS*</font> <font size =3>*Multiple Dimensional Scaling*</font>
</font> <font size =4><font color ='red'>**低维嵌入**</font></font>
### <font color = 'lightgreen'>**密采样**  *dense sample*</font>：
- <font color='green'>**定义**</font>：
**训练样本**の采样密度足够大  $\longrightarrow$  任意 **测试样本x**附近任意小的$\delta$距离范围内总能找到一个 训练样本

### <font color = 'lightgreen'>**维数灾难**  *curse of dimensionality*</font>：
- <font color='green'>**定义**</font>：高维情况下导致の
   - 数据**样本稀疏**
   - 距离**计算困难**
   - ...
### <font color = 'lightgreen'>**降维 (维数约简)**  *dimension reduction*</font>：
- <font color='green'>**定义**</font>：缓解 **维数灾难**の重要途径之一
- <font color='green'>**做法**</font>：
   - 原始高维属性空间   $~~~~~\underset{\Longrightarrow}{某种数学变换}~~$  **低维子空间**
- <font color='green'>**原因**</font>：
   - 与学习任务**密切相关**の仅仅为某个 **低维分布**  $\longrightarrow$ 高维空间の <font color = 'lightgreen'>**低维嵌入**</font>

</font> <font size =4><font color ='red'>**MDS**--多维缩放</font></font>

- <font color='green'>**优点**</font>：
    - 在低维空间依然**保持**样本在原始空间中的**距离**$$\Updownarrow$$
- <font color='green'>**目标**</font>：
    - 获得样本在 d'维空间の表示$$Z\in\mathbb{R}^{d'\times m},\quad d'\leq d$$
    - 任意两样本在 d'维空间の**欧氏距离**=原始空间的距离$$||z_i-z_j||~=~dist=_{ij}$$
- <font color='green'>**算法求解**</font>：
    - **原始空间** D： $$m个样本在原始空间的 距离矩阵:\\\begin{align*}&\\&D\in\mathbb{R}^{m\times m}\\&dist_{ij}:x_i到x_jの距离\end{align*}$$ 
    - **低维d'空间** Z：$$\begin{align*}&保持欧氏距离不变：\\&\qquad\qquad\quad||z_i-z_j||~=~dist_{ij}\end{align*}$$
    - <font color= 'lightgreen'>**内积矩阵**  B</font>:$$B=Z^TZ\longrightarrow 降维后样本的内积矩阵$$
**三者关系**：
$$\begin{align*}dist_{ij}^2~&=~||z_i||^2+||z_j||^2-2z_i^Tz_j\\&=~b_{ii}+b_{jj}+-2b_{ij}\end{align*}\\\Updownarrow\\
\left\{\begin{aligned}
&\\
&\underset{i=1}{\stackrel{m}{\sum}}dist_{ij}^2~=~tr(B) +m·b_{jj}\\
&\underset{j=1}{\stackrel{m}{\sum}}dist_{ij}^2~=~tr(B) +m·b_{ii}\\
&\underset{i=1}{\stackrel{m}{\sum}}\underset{j=1}{\stackrel{m}{\sum}}~dist_{ij}^2~=~2m·tr(B)\\
&\\
\end{aligned}\right.\\
\\\Updownarrow\\
\left\{\begin{aligned}
&\\
&dist_{i·}^2~=~\frac{1}{m}\underset{j=1}{\stackrel{m}{\sum}}dist_{ij}^2\\
&dist_{·j}^2~=~\frac{1}{m}\underset{i=1}{\stackrel{m}{\sum}}dist_{ij}^2\\
&dist_{··}^2~=~\frac{1}{m^2}\underset{i=1}{\stackrel{m}{\sum}}\underset{j=1}{\stackrel{m}{\sum}}~dist_{ij}^2\\
&\\
\end{aligned}\right.\\
\Updownarrow\\
b_{ij}~=~-\frac{1}{2}(~dist_{ij}^2-dist_{i·}^2-dist_{·j}^2+dist_{··}^2~)
$$
<font color = 'lightgreen'><font size =4>**特征值分解**  </font>*dimension reduction*</font>：
$$
B~=~V\Lambda V^T\\
\begin{align*}
&其中：\\
&\qquad\qquad B~:~内积矩阵\\
&\qquad\qquad \Lambda~:~特征值构成的对角矩阵 ,~\Lambda=diag(\lambda_1,\lambda_2,...,\lambda_d)\\
&\qquad\qquad V~:~特征向量矩阵
\end{align*}\\
\Updownarrow
$$

$$
\begin{align*}
&假设：\\   
&\qquad\qquad <1.\lambda_1\geq\lambda_2\geq....\geq\lambda_d\\
&\qquad\qquad <2.有d^*个 非零特征值~\longrightarrow 
\left\{\begin{aligned}
&\Lambda_*~=~diag(\lambda_1,\lambda_2,...,\lambda_{d^*})\\
&V_*~:~对应的 特征矩阵
\end{aligned}\right.\\
\end{align*}\\
\Updownarrow\\
\begin{align*}
&\\
Z~&=~\tilde{\Lambda}^{\frac{1}{2}}\tilde{V}^T~\in\mathbb{R}^{d'\times m}\\    
&=~\tilde{\Lambda_*}^{\frac{1}{2}}\tilde{V}_*^T~\in\mathbb{R}^{d^*\times m}\\    
\end{align*}

$$
# <font color ='red'>**线性降维**</font>
- <font color='green'>**定义**</font>：对原始高维空间进行高位变化
$$
Z=W^TX\\
\begin{align*}
&其中：\\
&\qquad\qquad X:d维空间的样本~,~X=(x_1,x_2,...,x_m)\in\mathbb{R}^{d\times m}\\
&\qquad\qquad W~:~变换矩阵   ~，W\in\mathbb{R}^{d\times d'}\\
&\qquad\qquad Z~:~X在新空间表达式~，Z\in\mathbb{R}^{d'\times m}
\end{align*}
$$
- <font color='green'>**特点**</font>：**最简单**の降维方法
# <font color ='red'>**主成分分析** *PCA*</font><font size =3> *Principal Component Analysis*</font>
- <font color='green'>**特点**</font>：
   - **最常用**の降维方法
- <font color='green'>**推到方向**</font>：
   - <font color='lightgreen'>**最近重构性**</font>：样本点距离超平面の **距离足够近**$$\begin{align*}&优化目标：\\&\qquad\qquad \underset{W}{min}~~-~tr(W^TXX^TW)\\&\qquad\qquad s.t.~W^TW~=~I\end{align*}\\\begin{align*}&其中：\\&\qquad\qquad W=(w_1,w_2,...,w_d)~,~投影变换后的新坐标系~\\ &\qquad\qquad\qquad\qquad\qquad\qquad\qquad 为 "标准正交基向量"=\left\{\begin{aligned}&||w_i||_2~=~1\\&w_i^Tw_j~=~0\end{aligned}\right.\\&\qquad\qquad X~:~样本集~（数据样本已经过 ~"中心化"\longrightarrow \sum_ix_i=0）\end{align*}$$
   - <font color='lightgreen'>**最大可分性**</font>：样本点在超平面上の **投影**$W^Tx_i$  能 **尽可能分开**$$\begin{align*}&优化目标：\\&\qquad\qquad \underset{W}{max}~~tr(W^TXX^TW)\\&\qquad\qquad s.t.~W^TW~=~I\end{align*}$$

- <font color='green'>**算法流程**</font>：
   - Step 1：对所有样本进行 **中心化**$$x_i\longleftarrow x_i~-~\frac{1}{m}\sum_{i=1}^mx_i$$
   - Step 2: 计算样本の **协方差矩阵**：$$协方差矩阵~=~X^TX$$
   - Step 3：**特征值分解**$$X^TX~=~V\Lambda^TV$$
   - Step 4：选取 **最大の** d'个特征值对应的特征向量$w_i$构成特征矩阵作为 **投影矩阵**$$W~=~(w_1,w_2,...,w_{d'})$$
# <font color ='red'>**流形学习** </font>
<font color ='red'><font size=4>**流形学习**</font></font>
- <font color='green'>**基础**</font>：
   - 借鉴了 **拓扑流形**概念作为 降维方法

- <font size =4><font color='lightgreen'>**流行**</font></font>：
   - 在**局部** 与 **欧氏空间** 同胚の空间$$\Updownarrow$$
   - **局部具有**欧氏空间的性质  $\longrightarrow$ 可以进行**距离计算**
- <font color='green'>**应用**</font>：
   - 可视化 

<font color ='red'><font size=5>**等度量映射** *IsoMAap*</font></font>

- <font size =4><font color='lightgreen'>**测地线**距离</font></font>：低维嵌入流形上两点の**本真距离**
- <font color='green'>**算法流程**</font>：
   - Step 1：计算每个样本$x_i$の **k近邻**：$$dist_{ik}~=~\left\{\begin{aligned} 欧氏距离:\underset{d=1}{\stackrel{D} {\sum}}||x_i^d-x_k^d||^{\frac{1}{2}},&\qquad\qquad x_k为x_iのk近邻\\ \infty\qquad\qquad\qquad,&\qquad\qquad otherwise\\ \end{aligned}\right.$$ 
   - Step 2：使用 **最短路径算法**计算任意样本点之间的距离$dist(x_i,x_j)$  $$最短路径算法~=~\left\{\begin{align*}&~~ Dijkstra~ 算法\\&~~Floyd~算法\end{align*}\right.$$
   - Step 3：将$dist_(x_i,x_j)$作为 **MDS算法**の输入
   - Step 4：输出MDS算法の结果:样本集D在 **低维空间の投影Z**$$Z=\{z_1,z_2,..,z_m\}$$

<font color ='red'><font size=5>**局部线性嵌入** *LLE*</font></font>Locally Linear Embedding 

- <font color='green'>**算法思想**</font>：
   - 保持 **邻域**内の线性关系$$\Updownarrow$$
   - **线性关系**在降维后的空间中继续保持$$\Updownarrow$$
   - 
$$
\begin{align*}
&假设：\\
&\qquad\qquad x_i~:~ 原始空间中の~样本i\\
&\qquad\qquad w_i~:~ 对样本x_i进行线性重构の系数\\
&\qquad\qquad Q_i~:~样本x_i~近邻样本~下标の集合   \\
&\qquad\qquad z_i~:~ 样本x_i在低维空间中の坐标
&\\
&\\
\end{align*}\\
\begin{align*}
&优化目标:\\
&\qquad\qquad\underset{w_1,w_2,..,w_m}{min}~\underset{i=1}{\stackrel{m}{\sum}}~||x_i-\underset{j\in Q_i}{\sum}~w_{ij}x_j||_2^2\\
&\\
&\qquad\qquad\qquad s.t.~~\underset{j\in Q_i}{\sum}~w_{ij}~=~1\\
\end{align*}\Longrightarrow ~~~w_{ij}~=~\frac{\underset{k\in Q_i}{\sum}~C_{jk}^{-1}}{\underset{l,s\in Q_i}{\sum}~C_{ls}^{-1}}\\
\begin{align*}
&\\
&\qquad 保持低维空间中~~\Updownarrow ~w_i~不变\\
&\\
&\qquad\underset{z_1,z_2,...,z_m}{\sum}~\underset{i=1}{\stackrel{m}{\sum}}~||z_i~-~\underset{j\in Q_i}{\sum}~w_{ij}z_j||_2^2\\
&\\
&\qquad\qquad\qquad\qquad\Updownarrow\\
\end{align*}\\
\begin{align*}
&令：\\
&\qquad\qquad Z=(z_1,z_2,...,z_m)~\in~\mathbb{R}^{d'\times m}\\
&\qquad\qquad W = (w_{ij}) \\
&\\
&\qquad\qquad M=(I-W)^T(I-W)
\end{align*}\\
\begin{align*}
&\\
&优化目标：\\
&\qquad\qquad \underset{Z}{min} tr(ZMZ^T)\\
&\qquad\qquad~~s.t.~ZZ^T~=~I
\end{align*}
$$


# <font color ='red'>**核化线性降维**</font>



# <font color ='red'>**度量学习**</font>

