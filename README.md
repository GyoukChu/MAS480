# MAS480
 Final Project for Introduction to Scientific Machine Learning course by Prof. Yeonjong Shin  

 * Topic: South Korea COVID prediction model by using PINN, SIRVD model  

### SIRVD Model
$$
\begin{cases}
{dS \over dt} = -{\beta \over N}IS + \sigma R - \alpha S \\
{dI \over dt} = +{\beta \over N}IS - \gamma I - \delta I\\
{dR \over dt} = +\gamma I -\sigma R\\
{dV \over dt} = +\alpha S\\
{dD \over dt} = +\delta I \\
\end{cases}
$$

### Datasets
From [6], we get V(t) from totalSecondCnt.  
From [5], we get I(t), R(t), D(t), and S(t) from decideCnt-clearCnt-deathCnt,  
clearCnt, deathCnt, 5174e+4-I(t)-R(t)-D(t)-V(t), respectively.  
  
Training data: training_data.csv  
Data From February 2nd, 2020 to October 31st, 2021  
Test data: testing_data.csv  
From November 1st, 2021 to December 2nd, 2021  
Both Data are composed of 6 lines and represent t, S, I, R, V, D in order.  

### References
[1] Schiassi, E., De Florio, M., D’ambrosio, A., Mortari, D., & Furfaro, R. (2021). Physics-informed neural networks and functional interpolation for data-driven parameters discovery of epidemiological compartmental models. Mathematics, 9(17), 2069.  
[2] Long, J., Khaliq, A. Q. M., & Furati, K. M. (2021). Identification and prediction of time-varying parameters of COVID-19 model: a data-driven deep learning approach. International Journal of Computer Mathematics, 98(8), 1617-1632.  
[3] Gai, C., Iron, D., & Kolokolnikov, T. (2020). Localized outbreaks in an SIR model with diffusion. Journal of Mathematical Biology, 80(5), 1389-1411.  
[4] Liao, Z., Lan, P., Fan, X., Kelly, B., Innes, A., & Liao, Z. (2021). SIRVD-DL: A COVID-19 deep learning prediction model based on time-dependent SIRVD. Computers in Biology and Medicine, 138, 104868.  
[5] Accumulated data on COVID-19 status. KDX, Korea Data Exchange. (2022, December 14). Retrieved December 15, 2022, from https://kdx.kr/data/view/25918  
[6] Corona (COVID-19) vaccine vaccination status cumulative data. KDX, Korea Data Exchange. (2022, November 9). Retrieved December 16, 2022, from https://kdx.kr/data/view/30239  
[7] Shaier, S., Raissi, M., & Seshaiyer, P. (2022). Data-driven approaches for predicting spread of infectious diseases through DINNs: Disease Informed Neural Networks. Letters in Biomathematics, 9(1), 71-105.  
[8] Loshchilov, I., & Hutter, F. (2016). Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983.  
