<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>


# 并行结算结课论文

高斯消元以及图像卷积的CUDA实现 

## 说明
要运行请使用Gauss/a.sh或Convolution/z.sh

## 公式备份

$$
\begin{pmatrix}
\begin{array}{ccc|c|cccccc}
{a_{11}}	&{a_{12}}	&{\cdots}	&{a_{1k}}	&{a_{1,k+1}}	&{\cdots}	&{a_{1j}}	&{\cdots}	&{a_{1n}}	&{a_{1,n+1}}\\

{0}	&{a_{22}}	&{\cdots}	&{a_{2k}}	&{a_{2,k+1}}	&{\cdots}	&{a_{2j}}	&{\cdots}	&{a_{2n}}	&{a_{2,n+1}}\\

{\vdots}	&{\vdots}	&{}	&{\vdots}	&{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}	&{\vdots}\\

\hline
{0}	&{0}	&{\cdots}	&\boxed{a_{kk}}	&{a_{k,k+1}}	&{\cdots}	&\boxed{a_{kj}}	&{\cdots}	&{a_{kn}}	&{a_{k,n+1}}\\

\hline
{0}	&{0}	&{\cdots}	&{a_{k+1,k}}	&{a_{k+1,k+1}}	&{\cdots}	&{a_{k+1,j}}	&{\cdots}	&{a_{k+1,n}}	&{a_{k+1,n+1}}\\

{\vdots}	&{\vdots}	&{}	&{\vdots}	&{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}	&{\vdots}\\

{0}	&{0}	&{\cdots}	&\boxed{a_{ik}}	&{a_{i,k+1}}	&{\cdots}	&\boxed{a_{ij}}	&{\cdots}	&{a_{in}}	&{a_{i,n+1}}\\

{\vdots}	&{\vdots}	&{}	&{\vdots}	&{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}	&{\vdots}\\

{0}	&{0}	&{\cdots}	&{a_{nk}}	&{a_{n,k+1}}	&{\cdots}	&{a_{nj}}	&{\cdots}	&{a_{nn}}	&{a_{n,n+1}}\\

\end{array}
\end{pmatrix}\rightarrow B
$$

$$
\begin{cases}
a_{ij}=&a_{ij}-\frac{a_{ik}}{a_{kk}}{a_{kj}} &,k+1\le i\le n, k+1\le j\le n+1\\
a_{ik}=&0			&,k+1\le i\le n
\end{cases}
$$

$$
\begin{matrix}
\begin{array}{|cccc|cccc|cc}
\hline
{a_0} &\boxed{a_2} &{\cdots} &{a_{len-1}}  &{a_{len}} &\boxed{a_{len+1}}   &{\cdots}    &{a_{2len-1}} &{a_{2len}}  &\boxed{a_{2len+1}} &{\cdots}\\
\hline
\end{array}
\end{matrix}
$$

$$
\text{0号线程负责规约:\quad}
\begin{matrix}
\begin{array}{|}
\hline
{b_{1}} &{b_{2}} &{\cdots} &{b_{len-1}}  \\
\hline
\end{array}
\end{matrix}
$$


$$
\begin{matrix}
\begin{array}{|ccccccc|c}

\hline
{a_{k,k+1}} &{\cdots} &{a_{k,j}} &{\cdots} &{a_{k,n-1}}  &{\cdots} &{a_{k,n}}   &\boxed{为第k行}\\
\hline


\hline
{a_{k+1,k+1}} &{\cdots} &{a_{k+1,j}} &{\cdots} &{a_{k+1,n-1}}  &{\cdots} &{a_{k+1,n}}   &\boxed{-\frac{a_{k+1,k}}{a_{kk}}\times 第k行}\\
\hline

{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}\\


\hline
{a_{i,k+1}} &{\cdots} &{a_{i,j}} &{\cdots} &{a_{i,n-1}}  &{\cdots} &{a_{i,n}}	&\boxed{-\frac{a_{i,k}}{a_{kk}}\times 第k行}\\
\hline

{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}\\


\hline
{a_{n-1,k+1}} &{\cdots} &{a_{n-1,j}} &{\cdots} &{a_{n-1,n-1}}  &{\cdots} &{a_{n-1,n}}   &\boxed{-\frac{a_{n-1,k}}{a_{kk}}\times 第k行}\\
\hline

\end{array}
\end{matrix}
$$



$$
\begin{matrix}
\begin{array}{c|c|}

\hline
{a_{k+1,k}}	&{a_{k+1,k+1}} &{\cdots} &{a_{k+1,j}} &{\cdots} &{a_{k+1,n-1}}  &{\cdots} &{a_{k+1,n}}\\


{\vdots}	&{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}\\



{a_{k+1,k}}	&{a_{i,k+1}} &{\cdots} &{a_{i,j}} &{\cdots} &{a_{i,n-1}}  &{\cdots} &{a_{i,n}}\\


{\vdots}	&{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}	&{}	&{\vdots}\\



{a_{k+1,k}}	&{a_{n-1,k+1}} &{\cdots} &{a_{n-1,j}} &{\cdots} &{a_{n-1,n-1}}  &{\cdots} &{a_{n-1,n}}\\
\hline

\boxed{为第k列}	&\boxed{-\frac{a_{k,k+1}}{a_{kk}}\times 第k列}	&{}	&\boxed{-\frac{a_{kj}}{a_{kk}}\times 第k列}	&{} &\boxed{-\frac{a_{kj}}{a_{kk}}\times 第k列}	  &{} &\boxed{-\frac{a_{kj}}{a_{kk}}\times 第k列}\\


\end{array}
\end{matrix}
$$

$$
\begin{matrix}
\begin{array}{c|c}
{} &{\vdots} &{\vdots} &{\vdots} &{} \\
\hline
{\cdots} &{a_{i-1,j-1}}	&{a_{i-1,j}}	&{a_{i-1,j+1}}	&{\cdots}\\
\hline
{\cdots}  &{a_{i,j-1}}	&\boxed{a_{ij}}	&{a_{i,j+1}}	&{\cdots}\\
\hline
{\cdots}  &{a_{i+1,j-1}}	&{a_{i+1,j}}	&{a_{i+1,j+1}}&{\cdots}\\
\hline
{}&{\vdots} &{\vdots} &{\vdots} &{} \\
\end{array}
\end{matrix}
\\
$$

$$
a_{ij}=\sum_{k,l=-1}^{1}a_{i+k,j+l}b_{2+k,2+l}
$$

$$
\begin{matrix}
\begin{array}{|c|c|}
\hline
4 &6 &6 &6 &4\\
\hline
6 &9 &9 &9 &6\\
\hline
6 &9 &9 &9 &6\\
\hline
6 &9 &9 &9 &6\\
\hline
4 &6 &6 &6 &4\\
\hline
\end{array}
\end{matrix}
$$


