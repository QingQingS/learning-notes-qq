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
   >> 作业文档里找 critic 网络 添加熵正则的公式
   >> 为什么公式里都是用期望，在实际update时没有期望
   >> 是均方差loss本身是在求样本的期望，所以等价了？
   >> double 网络 loss
   >> num_actor_samples 是什么意思，一个obs采样多个action?
   >> 重参数化
   >> 使用自己的语言和理解描述sac算法

今日回忆
soft-actor-critic 算法描述：
SAC是off-policy RL，replay buffer里的数据来自旧策略，（以前一直以为off-policy 就是当前要更新的动作策略与生成训练数据的使用的策略不是同一个策略，在完成这个作业时才发现，SAC里actor网络更新时使用动是当前策略中采样的，但是状态来自replay buffer）
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