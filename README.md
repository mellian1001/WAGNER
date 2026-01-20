## 项目说明


### 主要文件

1. `model.py`

主要包含了PPO模型的定义

2. `train.py`

主要包含单个模型的训练脚本

3. `test.py`

测试单个模型的脚本，可以在config字典中设置`render=True`可视化显示测试过程（远程连接服务器时可视化可能得设置端口转发）

4. `human_demonstrations.py`

人手动玩游戏，让LLM总结规则

5. `planning.py`

LLM根据规则做出任务规划，生成两个文件，分别是用自然语言表示和环境中物体表示的规划列表，用于后续提出子任务

6. `propose_subRLs.py`

根据上一步的规划和总结到的规则生成一个sub RL models的说明性json字典以及一个reward wrapper文件，用于训练模型

7. `train_sub_models.py`

训练sub RL models

8. `llm_agent.py`

整合所有好训练的子模型，让LLM根据规则调用模型，效果比baseline要好很多

9. `temp_result`

这个路径下包含所有LLM生成的中间结果

10. `utils`

这个路径下包含了一些实用工具，包括大模型的封装以及一些实用的环境wrapper，其中

* `InitWrapper`: 用于对装备栏初始化，如：

```python
env = InitWrapper(env, ["stone_pickaxe", "wood"], [1, 2])
```
表示初始环境装备栏中加入一个木头镐和两个木头，可以在`train.py`以及`test.py`的config字典中进行设置

11. `example_info.txt`

环境step后返回的内存信息的一个示例

12. `crafter`

包含crafter环境的实现

13. `RL_models`

包含训练好的子强化学习模型

14. 其它

包含最终模型测试结果以及指标分析脚本


### 使用方法

由于LLM输出的内容不稳定，直接写个脚本运行可能导致文件解析错误，因此现阶段分步实现了各个模块，以方便调试。后续会针对LLM特殊的输出做处理。

1. 人类演示玩游戏，让LLM总结规则：
```bash
python human_demonstrations.py
```
2. 根据总结出的规则，做规划：
```bash
python planning.py
```
3. 根据规划提出子任务：
```bash
python propose_subRLs.py
```
4. 训练子任务：
```bash
python train_sub_models.py
```
5. LLM根据subRL、游戏规则、规划以及当前状态做规划
```bash
python llm_agent.py
```

### 注意事项

1. 本项目在原始的crafter环境中做了一些修改，所以与通过`pip`安装的crafter环境可能会不兼容，建议新建一个环境，然后安装相关依赖：

```bash
pip install -r requirements.txt
```
2. 本项目使用的线上模型是deepseek671B，请在运行前设置环境变量`DEEPSEEK_API_KEY`为您的api_key

验证方法：

```bash
echo $DEEPSEEK_API_KEY
```
如果输出为您的api_key则代表环境变量成功设置
