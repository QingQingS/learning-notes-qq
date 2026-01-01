# Learning Notes (Work in Progress)  

This repository contains my personal learning notes. 
- Not tutorials 
- Not guaranteed to be correct 
- Updated continuously as my understanding evolves

# 20251230
 **在完成SAC作业时写下的笔记
 许多内容是在理解过程中记录的，后续可能会进行修改**

学习过程流水线：
1. 创建分布时，Independent与Normal
2. 为什么要将log_prob在最后一维相加
3. TransformedDistribution的作用
4. critic网络label获取:
   - 根据采样数据中下一时刻状态获取当前策略下一时刻动作
   - 使用target critic 获得下一时刻Q-value
   - 下一时刻Q-value的backup strategy（Double-Q swap, mean, clip）
   - 增加entropy_bonus
   - 期望符号去哪了
5. double Q loss
6. actor 网络更新：
   - 熵正则部分，梯度打开了
   - 优势函数 * 对数概率 部份梯度关闭了，为什么，反向的时候只从熵正则部分对actor 网络进行参数更新
   - 上面的问题理解了，但是有点绕，前一项的梯度没有关闭，只是将action当常量，该项的梯度是同过ation分布的log_prob部分传递的
   - actor 重参数化  loss = -q(s,重参数化样本) 重参数化样本 = .rsample() (z = mean() + std * ε)
   - target critic更新策略


中断时重启点：
   1. >> 作业文档里找 critic 网络 添加熵正则的公式
   2. >> 为什么公式里都是用期望，在实际update时没有期望
   3. >> 是均方差loss本身是在求样本的期望，所以等价了？
   4. >> double 网络 loss
   5. >> num_actor_samples 是什么意思，一个obs采样多个action?
   6. >> 重参数化
   7. >> 使用自己的语言和理解描述sac算法


回忆soft-actor-critic 算法流程：  
SAC是off-policy RL，replay buffer里的数据来自旧策略，（以前一直以为off-policy 就是当前要更新的动作策略与生成训练数据的使用的策略不是同一个策略，  
在完成这个作业时才发现，SAC里actor网络更新时使用动是当前策略中采样的，但是状态来自replay buffer）
1. 运行当前策略，生成一批数据加入到replay buffer里 （状态，动作，收益，下一个状态，轨迹是否结束）
2. 从replay buffer中采样一批数据D，在(当前策略+过去策略)中随机采样
3. critic网络训练：
    - 根据下一时刻状态s_t+1,运行actor网络采样一个动作a_t+1 ,这里动作来自当前策略
    - 使用target_critic网络计算q(s_t+1,a_t+1),
    - 如果使用的双Q网络，可以选择1.交换两者的q_t+1结果 2.求两个q_t+1的均值 3.两个q_t+1选最小值
    - 计算entropy bonus，下一时刻动作的概率似然，-log[pi(a_t+1)] 熵应该是它的期望值,这里是使用的采样样本
    - 计算target Q-value = r(st,at) + discount * (轨迹是否终止) * (q(s_t+1,a_t+1)-temperature * -log[pi(a_t+1)])
    - 使用critic网络估计q(s_t,a_t)与 target Q-value 进行均方差得到loss，反向梯度更新critic网络
4. actor网络训练：
    - 根据当前时刻状态s_t，运行actor网络采样新动作a1_t
    - 使用critic网络估计q(s_t,a1_t)，做为策略梯度的权重
    - 计算当前actor网络a_t的负对数似然然后乘以q(s_t,a1_t)，做为loss的第一项
    - 计算entropy bonus，同样根据当前时刻状态s_t，运行actor网络采样新动作a2_t，计算该动作的负对数似然乘以temperature，做为loss的第二项
    - 两项相减得到最终的loss，更新actor模型梯度


# 20251231
复习昨天关于SAC的部分（发现以问自己问题的形式来学习高效且有趣）
1. 训练critic网络目的是希望能对当前策略产生的（状态，动作）能带来的未来累积收益进行估计
    - 那对应的groundtruth就得是当前针对策略的Q-function，记为target_Q_valule
2. target_Q_valule在SAC算法中的计算方法是：  
    - target_Q_valule = r(s_t,a_t) + discount * Q(s_t+1, a_t+1)
    - 这里的s_t, a_t, s_t+1, a_t+1,还有Q(...)都指什么？
3. Q-function的贝尔曼公式一步展开后应该是：  
    - 当前状态动作的收益 + 下一时刻状态的value-funtion的期望，
    - 而value-funtion又等于该状态下所有动作的Q-function的期望，
    - target_Q_valule（groundtruth）为什么能使用的单个轨迹来计算Q_value ?
4. 为什么target_Q_value能作为当前策略的Q-value ？
5. 计算target_Q_value时使用的Q与要训练的Q为什么不是同一个？
6. 为什么使用double-Q？
7. 为什么在target_Q_value上也加了熵？   
    - 在这个算法中熵正则的目的是为了增加动作的探索空间，
    - 为什么在critic网络这部分也加而且是加在了groundtruth上？
    - wait a minute ,让我思考下，嗯，
    - 这里下一时刻的动作是从当前策略里采样出来的，它的Q_value是衡量未来收益，
    - 它的熵是表示这个动作概率意义？比如这个动作的Q指很大，但熵很小（在动作空间分布成簇状）， 
    - 或者这个动作Q很大同时熵也很大（在动作空间分布很散），
    - 那对训练critic有什么帮助呢？
    - 这样组合不就让训练出来的critic网络输出的结果不单纯是估计未来收益了，还包括了熵的信息，
    - 这样合理么？目标就不单一了呢
8. 均方差loss 或 L2真是个好东西，发现在很多地方用到了它预测的是均值/期望，无偏估计特性，
    - 比如在NCSN的Denoising score matching
9. actor网络训练部分其实就是on-policy的吧
10. 损失函数里有优势函数加权的策略梯度项和熵正则项，
    - 作业代码里这两项里求log_prob时用的动作都是来自当前策略，
    - 但为什么是不同的动作，不理解，极大的不理解？
    - 作业代码里：torch.mean(self.entropy(action_distribution))
    - 这里算熵的时候传了策略的分布，而不是采样好的动作

关键点：
- SAC 的 critic 学的是 soft Q
- Bellman target 是 Monte Carlo + bootstrapping 的无偏回归
- actorloss 里使用解析熵（Gaussian），直接用 distribution，不依赖具体 sampled action，是“对分布的期望”；而 target_Q_value 需要的是“期望里的一个样本”，手动进行负对数似然计算 ，而不是解析熵

SAC的论文还没看  

今日完成：  
 - cs224r lecture9 收尾，lecture10补全，
    - 这两节课换了讲师，课堂内容有点散，吸收率低了很多，没有想记录的知识点
 - 看了篇4D World Model论文，没看完，因为涉及到3DGS的内容，不懂，
    - 让chatgpt给讲了下，竟然没有很枯燥，比这篇论文有趣


NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos
1. pose-free, 不需要依赖相机位姿，这点很有吸引力，看到还要用相机位姿的就头大，这样数据简单了
2. 4D是啥？3D物体/场景+时间维度？
3. 既能重建也能生成

# 20260101
DPO
- 使用人类偏好数据对来调整模型输出
- 不用训练一个真实reward模型
- kl约束新旧策略
- 本质上还是对比学习，有监督学习

GRPO
- PPO的简化版（训练目标相同，重要性采样+裁剪）
- 去掉PPO的critic部分,不用训练value-function模型
- 那优势函数怎么求？
- 优势函数目的是衡量采取当前动作比平均动作好多少或差多少
- GRPO使用同一prompt采样多个样本，用这个多个样本的收益来模拟平均收益，即组内归一化（group-relative）
- 优势函数=（单个样本收益-减均值）/标准差


FlowMatching
- 拟合一个确定性ODE速度场
- 加噪不是扩散，而是插值采样
- 从x0加噪到xt，在扩散模型中，xt服从在给定x0的条件概率分布，从这个分布中使用重参数化采样一个xt，同一个x0同一个t得到的xt可能是不同的，是有随机性在的；
- flowmatching中是在数据分布和高斯噪声分布之间做插值采样，给定x0和噪声端x1,中间状态xt是确定的，随机性来自对噪声的采样
- 推理阶段，标准DDPM扩散模型依然是一个条件分布p(xt-1｜xt)，即使固定初始噪声xT，每次得到的路径也是不同的，因为每一步都要重新采样噪声,（ps:DDIM可以做成确定性的）；flowmatch，从同一个初始噪声出发，解ODE，路径是唯一的，轨迹是完全确定的

问：
- flowmatching有什么优势
- flowmatching又付出了什么代价

Flow-GRPO
- 好了，flow的优势在rl面前成了劣势，flow是确定性生成，而rl一大亮点是它通过探索更多的空间动作来寻找更优策略
- online rl需要高效采样数据，flow要multi-step才能生成一个完整的样本
- 在看具体的算法前，先脑海里思考下，如何在flow-matching训练过程中加入grpo？
    - 网络输出的是向量场，根据向量场怎么得到似然函数/概率密度p(x_t-1)|x_t,t,c)？
    - 论文看完了，这个概率分布是一长串公式推导
    - ODE找一个等价的SDE
    - 去看看代码
    - 看train_wan2_1.py x_t-1部分怎么跟论文公式对不上？？
    - 仔细核对了，一样的，做了同类项提取
    - x_t+Δt​∣x_t​∼N(μ(x_t​,t)Δt,(σ_t)**2*​Δt*I)--> 概率密度是多维高斯分布