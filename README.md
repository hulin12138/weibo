# weibo
## 分工
- 数据处理
    - 分词
    - 可以考虑采用词向量的形式
- 算法实现
    - 根据分词的结果实现简单的Bayes算法
    - 评估
## git-usages
- 首先要fork代码到自己的github的帐号上,然后自己上传到自己的github的repo,最后在添加pull request才能合并到原来的owner的代码中.
- git checkout -b branchname
- git commit -am "name date description" __在push前一定要commit__
- git push origin/branchname
- git pull __从git上下载全部的branches__
- git merge....

## 代码运行依赖包
### Yan Yongfei
- tqdm 4.11.2
- python 2.7

## 接下来的目标
- 完成一个感知机算法
- 记录每个算法的训练准确率, 测试准确率, top2准确率, 
- 完成一个集成算法
- 调节神经网络的参数进行训练
- 对数据进行预处理, 之后比较结果的好坏,包括去除高频词,重采样等 