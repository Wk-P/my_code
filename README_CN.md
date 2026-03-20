# Reinforcement Learning VS Optimization

本 README 聚焦仓库内各类脚本的实际用法，方便直接执行训练、汇总和清理流程。

英文版请见 README.md。

## 1. 项目结构

### 核心问题目录

<table>
  <thead>
    <tr>
      <th>目录</th>
      <th>说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>problem1</td>
      <td>基础环境定义与早期问题描述</td>
    </tr>
    <tr>
      <td>problem2_ilp</td>
      <td>ILP 配置生成与最优解求解</td>
    </tr>
    <tr>
      <td>problem3_ppo</td>
      <td>无约束 PPO</td>
    </tr>
    <tr>
      <td>problem4_ppo_mask</td>
      <td>带动作掩码的 PPO</td>
    </tr>
    <tr>
      <td>problem5_ppo_lagrangian</td>
      <td>拉格朗日约束 PPO</td>
    </tr>
    <tr>
      <td>problem6_ppo_opt</td>
      <td>带修补优化逻辑的 PPO</td>
    </tr>
    <tr>
      <td>dqn</td>
      <td>DQN 基线</td>
    </tr>
    <tr>
      <td>run_scripts</td>
      <td>一键运行、结果复制、汇总作图、结果清理</td>
    </tr>
  </tbody>
</table>

### 结果目录

每个问题目录下通常包含一个 results 子目录。单次运行会在其中创建一个时间戳目录，例如：

- 20260321_071733

标准输出通常包括：

- results.json
- summary.csv
- training_curve.png
- comparison.png

## 2. 环境准备

推荐使用仓库内虚拟环境：

    .venv/bin/python

如果还没有安装依赖：

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Linux 下推荐统一使用：

    /home/soar009/github/my_code/.venv/bin/python

现在所有可训练问题的默认训练步数统一由这个文件管理：

    training_steps_config.py

如果你想修改所有问题的默认 TOTAL_STEPS，直接改这个文件即可，不需要再分别修改每个问题目录下的 config.py。

## 3. 最常用入口

### 3.1 一键跑全部问题

脚本：run_scripts/all_run_shell_script.sh

用途：

- 顺序运行 P3、P4、P5、P6、DQN
- 自动复制最新 summary.csv 到汇总目录
- 自动生成 combined_results.png
- 可选在结束后自动清理结果目录

最常用命令：

    source run_scripts/all_run_shell_script.sh --total-timesteps 500000

快速测试：

    source run_scripts/all_run_shell_script.sh --total-timesteps 1000

训练后顺手清理今天结果和不完整记录：

    source run_scripts/all_run_shell_script.sh --total-timesteps 1000 --cleanup-results

只执行清理，不跑训练：

    source run_scripts/all_run_shell_script.sh --cleanup-only --cleanup-today --cleanup-date 20260321

查看帮助：

    bash run_scripts/all_run_shell_script.sh --help

支持的清理参数：

- --cleanup-results
- --cleanup-only
- --cleanup-today
- --cleanup-incomplete
- --cleanup-date YYYYMMDD
- --cleanup-dry-run

### 3.2 用 Python 顺序跑全部问题

脚本：run_scripts/run_all_problems.py

用途：

- 只负责顺序执行五个 run_all.py
- 不负责复制 summary.csv
- 不负责生成 combined 图

命令：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/run_all_problems.py

快速测试：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/run_all_problems.py --total-timesteps 1000

查看帮助：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/run_all_problems.py --help

## 4. 单独运行每个问题

下面这些脚本都会在对应目录的 results 中生成一个新的时间戳目录。

### 4.1 P3：无约束 PPO

脚本：problem3_ppo/run_all.py

命令：

    /home/soar009/github/my_code/.venv/bin/python problem3_ppo/run_all.py

快速测试：

    /home/soar009/github/my_code/.venv/bin/python problem3_ppo/run_all.py --total-timesteps 1000

### 4.2 P4：动作掩码 PPO

脚本：problem4_ppo_mask/run_all.py

命令：

    /home/soar009/github/my_code/.venv/bin/python problem4_ppo_mask/run_all.py

快速测试：

    /home/soar009/github/my_code/.venv/bin/python problem4_ppo_mask/run_all.py --total-timesteps 1000

### 4.3 P5：拉格朗日 PPO

脚本：problem5_ppo_lagrangian/run_all.py

命令：

    /home/soar009/github/my_code/.venv/bin/python problem5_ppo_lagrangian/run_all.py

快速测试：

    /home/soar009/github/my_code/.venv/bin/python problem5_ppo_lagrangian/run_all.py --total-timesteps 1000

### 4.4 P6：修补优化 PPO

脚本：problem6_ppo_opt/run_all.py

命令：

    /home/soar009/github/my_code/.venv/bin/python problem6_ppo_opt/run_all.py

快速测试：

    /home/soar009/github/my_code/.venv/bin/python problem6_ppo_opt/run_all.py --total-timesteps 1000

### 4.5 DQN

脚本：dqn/run_all.py

命令：

    /home/soar009/github/my_code/.venv/bin/python dqn/run_all.py

快速测试：

    /home/soar009/github/my_code/.venv/bin/python dqn/run_all.py --total-timesteps 1000

### 4.6 所有 run_all.py 的共同参数

当前五个训练入口都支持：

- --total-timesteps TOTAL_TIMESTEPS

用途：覆盖对应配置中的 TOTAL_STEPS，适合 smoke test 和调试。

默认值统一从这个文件读取：

    training_steps_config.py

## 5. ILP 相关脚本

### 5.1 生成 ILP 配置

脚本：problem2_ilp/config/generate_config.py

用途：

- 随机生成多个 scenario
- 输出到 problem2_ilp/config/config_YYYYMMDD_HHMMSS.yaml

命令：

    /home/soar009/github/my_code/.venv/bin/python problem2_ilp/config/generate_config.py

说明：

- 这个脚本目前没有命令行参数
- 场景数量写死在脚本里，默认是 200

### 5.2 求 ILP 最优解并生成汇总

脚本：problem2_ilp/optimal_solution/main.py

用途：

- 读取 problem2_ilp/config 下的 YAML 配置
- 对所有 scenario 求最优解
- 生成共享缓存和统计图

命令：

    /home/soar009/github/my_code/.venv/bin/python problem2_ilp/optimal_solution/main.py

说明：

- 这个脚本当前没有标准命令行参数
- 使用哪个配置文件，取决于脚本底部的 config_filename 默认值
- 如果要切换配置文件，需要修改脚本中的默认文件名

## 6. 汇总与复制脚本

### 6.1 复制最新 summary.csv 到汇总目录

脚本：run_scripts/copy_files.py

用途：

- 从每个问题目录的最新有效结果目录中复制 summary.csv
- 复制到 run_scripts/results_combined 对应子目录

命令：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/copy_files.py

输出文件：

- run_scripts/results_combined/problem3_ppo/summary.csv
- run_scripts/results_combined/problem4_ppo_mask/summary.csv
- run_scripts/results_combined/problem5_ppo_lagrangian/summary.csv
- run_scripts/results_combined/problem6_ppo_opt/summary.csv
- run_scripts/results_combined/dqn/summary.csv

### 6.2 生成综合对比图

脚本：run_scripts/results_combined/combined.py

用途：

- 读取汇总目录中的五份 summary.csv
- 生成综合对比图 combined_results.png

命令：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/results_combined/combined.py

输出文件：

- run_scripts/results_combined/combined_results.png

当前 combined 图展示三类信息：

- AR 对比
- 违约率对比
- 放置完成度对比

## 7. 清理脚本

脚本：run_scripts/cleanup_results.py

用途：

- 删除指定日期生成的结果目录
- 删除不完整记录

命令示例：

预览将删除什么：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/cleanup_results.py --dry-run

只删除今天结果：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/cleanup_results.py --today

只删除不完整记录：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/cleanup_results.py --incomplete

删除指定日期结果：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/cleanup_results.py --today --date 20260321

同时处理日期结果和不完整记录：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/cleanup_results.py --today --incomplete --date 20260321

如果不传 --today 和 --incomplete：

- 脚本默认两者都执行

## 8. 推荐执行顺序

### 方案 A：最省事

适合日常训练和出图：

    source run_scripts/all_run_shell_script.sh --total-timesteps 500000

### 方案 B：先单独调一个方法

适合调试某个问题：

    /home/soar009/github/my_code/.venv/bin/python problem5_ppo_lagrangian/run_all.py --total-timesteps 1000

然后如果想刷新汇总图：

    /home/soar009/github/my_code/.venv/bin/python run_scripts/copy_files.py
    /home/soar009/github/my_code/.venv/bin/python run_scripts/results_combined/combined.py

### 方案 C：跑完后清理

    source run_scripts/all_run_shell_script.sh --total-timesteps 1000 --cleanup-results --cleanup-today --cleanup-incomplete

## 9. 常见输出位置

### 单问题输出

- problem3_ppo/results/时间戳目录
- problem4_ppo_mask/results/时间戳目录
- problem5_ppo_lagrangian/results/时间戳目录
- problem6_ppo_opt/results/时间戳目录
- dqn/results/时间戳目录

### 汇总输出

- run_scripts/results_combined/problem\*/summary.csv
- run_scripts/results_combined/dqn/summary.csv
- run_scripts/results_combined/combined_results.png

## 10. 注意事项

1. all_run_shell_script.sh 推荐用 source 执行，这样与当前仓库里的使用方式一致。
2. run_all.py 的快速测试步数只适合检查流程，不代表最终训练效果。
3. problem2_ilp/optimal_solution/main.py 目前仍然依赖脚本内默认配置文件名，不是完整参数化入口。
4. cleanup_results.py 只会删除时间戳目录，不会删除模型 zip 或 ilp_cache.json。
