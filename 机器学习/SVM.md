# <font color ='red'>**基本概念**</font>
### <font color = 'lightgreen'>**划分超平面(w,b)**</font>
**超平面模型**：
$$
w^Tx+b=0\\
\begin{align*}
&\\
&其中：\\
&\qquad\qquad w = (w_1;w_2,...;w_d)~，平面法向量\\
&\qquad\qquad b : 平面位移项
\end{align*}
$$
点 $x$ 到面の**距离d**：
$$
d = \frac{|w^Tx+b|}{||w||}
$$
### <font color = 'lightgreen'>**支持向量**</font>
**定义**：离超平面 (w,b)**距离最近**的几个**训练样本点**
$$
支持向量\hat{x}= \underset{x}{min}(d_x~|~x\in X)
$$
### <font color = 'lightgreen'>**异类**</font>
**分类规则**:
$$
\left\{\begin{aligned}
w^Tx+b\geq +1&\qquad,~y_i=+1\\
w^Tx+b\leq -1&\qquad,~y_i=-1
\end{aligned}\right.
$$
其中两个 $y_i$ 不同(在平面的正反面) 的支持向量 $\hat{x}$ 互为异类
### <font color = 'lightgreen'>**间隔$\gamma$**</font>
两个**异类** $\hat{x}$ 到超平面(w,b)的**距离之和**
$$
\gamma = \frac{2}{||w||}
$$

### <font color = 'lightgreen'>**软/硬间隔**</font>
- <font color='green'>**软间隔**</font>：所有样本都正确化分
- <font color='green'>**硬间隔**</font>：允许某些不满足约束

### <font color = 'lightgreen'>**KKT条件**</font>*Karush-Kuhn-Tucker条件*
SVM对偶问题当中：
$$
\left\{\begin{align*}
&\alpha_i~(y_if(x_1)-1)=0\\
&\alpha_i\geq0\\
&y_if(x_i)-1\geq0
\end{align*}\right.
$$
**特点**：最优问题有解 的 **必要条件**
### <font color = 'lightgreen'>**核函数  $\mathcal{k(·,·)}$**</font>
$$
\begin{align*}
&k(x_i,x_j) = \phi(x_i)^T\phi(x_j)\qquad\qquad,x_i,x_j\in\mathcal{X}\\
&\\
&\mathcal{X}:输入空间\\
&\mathcal{k( · , · )}:定义在\mathcal{X}\times\mathcal{X}上的对称函数\\  
\end{align*}\\
\Updownarrow\\
$$
- <font color='green'>**常用核函数**</font>：
   - **线性**核：$$k(x_i,x_j)=x_i^Tx_j$$
   - **多项式**核:$$k(x_i,x_j)=(x_i^Tx_j)^d\qquad\qquad\qquad d\geq1(多项式次数)$$
   - **高斯**核：$$k(x_i,x_j)=exp(-\frac{||x_i-x_j||^2}{2\sigma^2})\qquad\qquad \sigma>0(高斯核带宽)$$
   - **拉普拉斯**核:$$k(x_i,x_j)=exp(-\frac{||x_i-x_j||}{\sigma})\qquad\qquad \sigma>0$$
   - **Sigmoid核**：
  $$\begin{align*}
    &k(x_i,x_j)=tanh(\beta x_i^Tx_j+\theta)\\
   &\\
   &\qquad \beta>0,\theta<0,tanh(·)双曲正切函数 
    \end{align*}$$
 
- <font color='green'>**其他核函数**</font>：
    - **线性组合**：$$\gamma_1k_1(·,·)+\gamma_2k_2(·,·)$$
    - **直积**：$$k_1\otimes k_2(x,z) = k_1(x,z)\otimes k_2(x,z)$$
    - **函数**：$$k(x,z)=g(x)k(x,z)g(z)\qquad ，其中g(x)为任意函数$$

### <font color = 'lightgreen'>**核矩阵K**</font>
$$
\begin{align*}
&对于任意数据D={x_1,x_2,...,x_m}   \\
&\\
&K=\left[\begin{matrix}
k(x_1,x_1)&~...~&~k(x_1,x_j)&~...~&~k(x_1,x_m)\\
.&~...~&~.&~...~&~.\\
k(x_i,x_1)&~...~&~k(x_i,x_j)&~...~&~k(x_i,x_m)\\
.&~...~&~.&~...~&~.\\
k(x_m,x_1)&~...~&~k(x_m,x_j)&~...~&~k(x_m,x_m)\\
\end{matrix}\right]
\end{align*}
$$
**特点**：
   - **半正定** 矩阵
   -  核矩阵K 与 核函数k(·,·) 一一对应
  
  $\Updownarrow$

### <font color = 'lightgreen'>**Mercer定理**:</font>
(**充分非必要**条件)： 若一个对称函数所对应的核矩阵K半正定，则该函数可以作为核函数 k( · , · )


### <font color = 'lightgreen'>**再生核希尔伯特空间**  *RKHS空间*        $\mathbb{H}$</font>
对于 **任何一个**核函数 都**隐式定义**了一个 再生和希尔伯特空间

### <font color = 'lightgreen'>**表示定理**:</font>
$$
\begin{align*}
&\mathbb{H}~:~核函数k(·,·)对应的再生核希尔伯特空间\\
&||h||_{\mathbb{H}}~:~在\mathbb{H}空间中 关于h的范式\\
&\Omega~:~任意 单调递增的函数\\
&l~:~任意非负损失函数~~,~l:\mathbb{R}^m \mapsto [0,\infin] \\
&\\
&对于任意优化问题：\\
&\qquad\qquad\underset{h\in\mathbb{H}}{min}~F(h)~=~\Omega(||h||_{\mathbb{H}})~+~l(h(x_1),h(x_2),...,h(x_m))\\
&\\
&其最优解为：\\
&\qquad\qquad\qquad\qquad h^*(x)~=~\underset{i=1}{\stackrel{m}{\sum}}\alpha_ik(x,x_i)
\end{align*}
$$

**应用**: 解决 **线性不可分**的问题

# <font color ='red'>**算法流程**</font>
- <font color='green'>**模型构成**</font>：
**定义**：寻找一个超平面(w,b)，能将数据**线性划分**----> 能将 异类全部线性化分
$$f(x)=w^Tx+b$$

- <font color='green'>**目标函数**</font>：
<font color = 'lightgreen'>**最大化间隔**</font>

<font color='lightblue'>**硬间隔SVM**</font>：
$$
\left\{\begin{align*}
&\underset{w,b}{max}~\frac{2}{||w||}\\
&\\
&s.t. y_i(w^Tx+b)\geq 1\quad,i=1,2,...,m   
&\\ 
\end{align*}\right.\\
\Updownarrow\\
\left\{\begin{align*}
&\\
&\underset{w,b}~{min}||w||^2\\
&\\
&s.t. y_i(w^Tx+b)\geq 1\quad,i=1,2,...,m    
\end{align*}\right.\\
$$

<font color='lightblue'>**软间隔SVM**</font>：

**约束条件** 引入**松弛变量 **$\xi\geq0$ :
$$
\left\{\begin{align*}
\underset{w,b,\xi_i}{min}~\frac{1}{2}||w||^2~+~C\underset{i=1}{\stackrel{m}{\sum}}\xi_i\\
s.t.~~y_i(w^Tx_i+b)\geq 1-\xi_i\\
\qquad\xi_i\geq0~,~i=1,2,...,m
\end{align*}\right.

$$

- <font color='green'>**模型求解**</font>：
   - **凸二次规划**问题求解：
      - 使用现成的 **优化计算包** 
      - **拉格朗日乘子法** 转化为 **对偶问题**

<font size=4><font color = 'lightgreen'>**拉格朗日乘子法**</font></font>

- <font color='green'>**目标函数$\longrightarrow$拉格朗日函数**</font>：

**定义**：对于模型(w,b)的每条 **约束**添加拉格朗日乘子$\alpha_i\geq0$

<font color='lightblue'>**硬间隔SVM**</font>：
$$
\begin{align*}
&对于每个样本x_i都有一个约束条件：y_i(1-w^Tx_i+b)\geq 1\\
&\therefore 目标函数：\\
&\qquad\qquad\mathcal{Loss(w.b.\alpha)} = \frac{1}{2}||w||^2~+~\underset{i=1}{\stackrel{m}{\sum}}~\alpha_i(1-y_i(w^Tx_i+b))    
\end{align*}
$$

<font color='lightblue'>**核函数映射处理的SVM**</font>：
$\phi(x)$:向量x经过核函数映射后的对应的向量
$$
\qquad\qquad\mathcal{Loss(w.b.\alpha)} = \frac{1}{2}||w||^2~+~\underset{i=1}{\stackrel{m}{\sum}}~\alpha_i(1-y_i(w^T\phi(x_i)+b))  
$$

<font color='lightblue'>**软间隔SVM**</font>：
$$
\begin{align*}
&\mathcal{Loss(w,b,\alpha,\xi,\mu)}= \frac{1}{2}||w||^2~+~\underset{i=1}{\stackrel{m}{\sum}}~\alpha_i(1-y_i(w^Tx_i+b)-\xi_i) -\underset{i=1}{\stackrel{m}{\sum}}\mu_i\xi_i      \\
&其中：\\
&\qquad\alpha_i:y_i(1-w^Tx_i+b)\geq 1~~约束条件的拉格朗日乘子\\
&\qquad\mu_i~:~\xi_i\geq0~~约束条件的拉格朗日乘子
\end{align*}
$$


- <font color='green'>**参数求解**（偏导为零）</font>：
$$
\left\{\begin{align*}
&\frac{\partial\mathcal{Loss(w,b,\alpha)}}{\partial w}=0\\
&\frac{\partial\mathcal{Loss(w,b,\alpha)}}{\partial b}=0\\
&\\
\end{align*}\right.\\
\Updownarrow\\
\left\{\begin{align*}
&\\
&w~=~\underset{i=1}{\stackrel{m}{\sum}}~\alpha_iy_ix_i\\
&0~=~\underset{i=1}{\stackrel{m}{\alpha_iy_i}}
\end{align*}\right.
$$
- <font color='green'>**模型转换**（回代）</font>：
<font color = 'lightgreen'>**模型的对偶问题**</font>

<font color='lightblue'>**硬间隔SVM**</font>：
$$
\underset{\alpha}{max}~\underset{i=1}{\stackrel{m}{\sum}}~\alpha_i - \frac{1}{2}\underset{i=1}{\stackrel{m}{\sum}}\underset{j=1}{\stackrel{m}{\sum}}~\alpha_i\alpha_j~y_iy_j~x_i^Tx_j
$$

<font color='lightblue'>**核函数映射处理的SVM**</font>：
$$
\underset{\alpha}{max}~\underset{i=1}{\stackrel{m}{\sum}}~\alpha_i - \frac{1}{2}\underset{i=1}{\stackrel{m}{\sum}}\underset{j=1}{\stackrel{m}{\sum}}~\alpha_i\alpha_j~y_iy_j~\phi(x_i)^T\phi(x_j)
$$
$其中：\phi(x_i)^T\phi(x_j)为向量内积$

- <font color='green'>**KKT条件**要求：</font>：
   - 若 $\alpha_i = 0,则~样本不参与计算（不出现）$
   - 若 $\alpha_i > 0,则~y_if(x_i)=1$
   - SVM<font size=4><font color = 'lightgreen'>解の'**稀疏性**'</font></font>$$\downarrow$$
$$\begin{align*}
训练完成后，&大部分训练模型无需保留\\&最终模型仅与支持向量有关   
\end{align*}$$


- <font color='green'>**参数求解**</font>：<font size=4><font color = 'lightgreen'>**SMO算法**</font></font>
   - **算法思想**:
      - 每次 选取一对需要更新的参数$(\alpha_i,\alpha_j)$ 
      - 固定$(\alpha_i,\alpha_j)$ 以外的参数$\alpha_k~,(k\neq i,j)$ :$$\begin{align*}&\alpha_iy_i+\alpha_jy_j = c~~~~,~\alpha_i,\alpha_j\geq0\\&\\&c = -\underset{k\neq i,j}{\sum}\alpha_ky_k~~~~,\alpha_k\geq0\end{align*}$$
      - 求解 **模型对偶式** ----> 更新参数$(\alpha_i,\alpha_j)$ 
   - 求解**b**：$$b=\frac{1}{|S|}\underset{s\in S}{\sum}(\frac{1}{y_s}-\underset{i\in S}{\sum}\alpha_iy_ix_i^Tx_i)\\\begin{align*}&\\&其中：S={i|\alpha_i>0,i=1,2,..,m}为所有支持向量机的 下标集\end{align*}$$

**特点**：将 **对偶问题**   $\underset{\longrightarrow}{转换}$   **单变量**的 **二次规划**问题


# <font color ='red'>**正则化项**</font>
**一般形式**：
$$
\underset{f}{min}~\Omega(f)~+~C~\underset{i=1}{\stackrel{m}{\sum}}~l(f(x_i),y_i)\\
\begin{align*}
&\\
&其中：\\
&\qquad \Omega(f)~:~正则化项，结构风险\\
&\qquad \underset{i=1}{\stackrel{m}{\sum}}~l(f(x_i),y_i)~:~经验风险\\
&\qquad C~:~正则化常数
\end{align*}
$$

<font size=4><font color = 'lightgreen'>**结构风险**</font></font>

**定义**：描述**模型**f的某些**性质**

<font size=4><font color = 'lightgreen'>**经验风险**</font></font>

**定义**：描述 **模型**与 **训练数据**的 **契合程度**

<font size=4><font color = 'lightgreen'>**经验风险最小化**</font></font>：

**好处**：
   - $\Omega(f)$:表达了希望获得何种性质的模型
   - 有利于消减**假设空间**---->降低 **最小化训练误差**的 **过拟合**风险



# <font color ='red'>**软间隔SVM**</font>
**特点**：
   - 结果仅与 **支持向量**有关
   - 加入$l_{hinge}(z)$损失函数依然保持** 稀疏性**

**约束条件** 引入**松弛变量 **$\xi\geq0$ :
$$
\left\{\begin{align*}
\underset{w,b,\xi_i}{min}~\frac{1}{2}||w||^2~+~C\underset{i=1}{\stackrel{m}{\sum}}\xi_i\\
s.t.~~y_i(w^Tx_i+b)\geq 1-\xi_i\\
\qquad\xi_i\geq0~,~i=1,2,...,m
\end{align*}\right.
\\\Updownarrow
$$
**约束条件**：加入正则化项 "**损失函数**"(松弛变量 $\xi_i$ )：
$$
\underset{w,b,\xi_i}{min}~\frac{1}{2}||w||^2~+~C\underset{i=1}{\stackrel{m}{\sum}}~l_{损失函数}(·)\\
$$
<font size=4><font color = 'lightgreen'>**损失函数**</font></font>

<font color = 'lightgreen'>**0/1损失函数**</font>
$$l_{0/1}(z)=\left\{\begin{align*}1~,~&\qquad if z<0\\0~,~&\qquad otherwise\end{align*}\right.$$

**缺点**：
   - 数学性质不太好：非凸，非连续

<font color = 'lightgreen'>**替代损失**</font>

**定义**：凸的连续函数，是 $l_{0/1}(z)$ 损失函数的上限

   - **hinge**损失函数：$$l_{hinge}(z)~=~max(0,1-z)$$
   - **指数**损失函数：$$l_{exp}(z)=~exp(-z)$$
   - **对率**损失函数：$$l_{log}(z)~=~log(1+exp(-z))$$



# <font color ='red'>**SVM回归** *SVR*</font>
- <font color='green'>**模型**</font>：
$$
f(x) = w^Tx~+~b\\
 ~\\
其中：模型输出f(x)与实际输出y之间存在2\epsilon
$$

- <font color='green'>**损失函数**</font>：
$$
l(z)=\left\{\begin{align*}
0\qquad\qquad&if |z|\leq\epsilon\\
~|z|-\epsilon\qquad\qquad&otherwise
\end{align*}\right.
$$


