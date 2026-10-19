# TMax CPU rollout 接入

此整理分支保留正式实验使用的 Vanillux2Agent、OpenSandbox 模型传输、
每 task 镜像解析、SDK 0.4.1 和相关回归测试。原 weiqi 分支未改变。

完整运行入口位于配套 AvaTrain 整理分支的 experiments/tmax/README.md。
请通过其中的 run.sh 配置并启动指定远端 CPU；不要另起第二套 listener。

发布到个人fork的tmax-reproduce-compat-20261020分支，不向sii-system默认分支直接推送：

- CPU接入：[weiqiFan123/agent-fleet](https://github.com/weiqiFan123/agent-fleet/tree/tmax-reproduce-compat-20261020)。
- 训练与完整说明：[weiqiFan123/weiqi](https://github.com/weiqiFan123/weiqi/blob/tmax-reproduce-compat-20261020/experiments/tmax/README.md)。

当前本地分支为tmax-reproduce-compat-20261020，基于已验证训练基线f801de1；
远端Tmax为38c47ae、尚无OpenSandbox，不能直接覆盖几个文件便称已经兼容。
配套AvaTrain克隆为weiqi-Tmax-local，使用同名本地分支。
配套版本见AvaTrain的experiments/tmax/versions.lock.json。

先从配套README执行clone、依赖准备、Ray/NCCL、Nginx及sandbox probe，再启动CPU入口。
端到端验收要求真实8×32批次完成optimizer update、metrics与完整checkpoint落盘，
不以sandbox运行或模型HTTP200代替训练通过；见配套docs/e2e.md。

本地改动覆盖的是当前正式版本，上游 #235 异步命令路径及新版 verifier 的
隔离验证版本没有混入这个分支。保留其已知限制，见 tasks issue #204。

凭据放在本 checkout 独立的 config.local.env，不复制原开发机的整份配置。
提交给通用框架的补丁需与 TMax 专用接入分别审查。
