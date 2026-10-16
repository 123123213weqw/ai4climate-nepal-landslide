# DeepSeek Club submission post (paste into /community/post)

Title:
【AI4Climate】喜马拉雅降雨滑坡易发性制图：全开源数据 + 混合CNN 的尼泊尔山地气候韧性方案

Body:

项目名称：Rainfall-Driven Landslide Susceptibility Mapping for the Nepal Himalaya（喜马拉雅降雨型滑坡易发性制图）

选题方向：推荐方向 02 · 尼泊尔山地气候韧性（山地社区：山洪 / 滑坡 / 冰湖溃决）

项目阶段：可运行原型 + 完整研究基线

团队：Yue Wang、Chenfeng Zhang

一句话介绍：用 NASA 开放滑坡目录和全开源地形、气候、河流与活动断层数据，训练"地形图像 + 气候融合"的混合 CNN，在严格的空间交叉验证下达到 AUC 0.898，并输出 200 米分辨率、可交付社区使用的滑坡易发性地图。

一、真实气候问题

尼泊尔喜马拉雅山地社区每年季风期（6–9 月，约占全年降水 80%）因暴雨触发大量滑坡，威胁村庄、徒步路线与县乡公路。我们对 NASA 全球滑坡目录中尼泊尔的 479 条事件做了解析：84% 由降雨触发（downpour / continuous rain / monsoon），地震触发仅 9 条 —— 这是一个典型的"气候驱动"灾害问题。受影响与将使用本方案的，是县乡灾害管理办公室（我们的目标验证伙伴是 Gandaki 省的 2–3 个市）与社区减灾工作者。

二、AI 方案与技术有效性

- 11 个条件因子：8 个地形因子（SRTM 衍生：高程、坡度、坡向、曲率、粗糙度、局部起伏、湿度指数）+ 距河流 / 距活动断层 / 距道路距离；
- 3 个气候协变量：年降水、季风期（6–9 月）降水、降水的月际变率（NASA POWER）；
- 四模型基准（0.5° 空间分块交叉验证，杜绝随机划分的空间泄漏）：逻辑回归 0.852 / 梯度提升 0.893 / MLP 0.877 / 混合 CNN 0.898 ± 0.008（最优且方差最小）；
- 三个方法学发现：① 消融实验显示气候+距离融合头贡献 +0.013 AUC 并使折间方差减半；② 同数据同模型的"随机划分 vs 空间划分"配对实验量化了基准虚高（+0.002~0.008 AUC）；③ 置换重要性显示断层距离与河流距离是高程之外最强的因子；
- 推理：对 Gorkha–Trishuli–Langtang 走廊滑窗打分 37.6 万个 8×33×33 地形窗口，生成 200 米分辨率易发性地图，高易发通道与水系及目录事件一致。

三、可运行（Demo / 代码仓库）

- 代码仓库（含论文 PDF、训练好的模型、地图、全部指标）：https://github.com/123123213weqw/ai4climate-nepal-landslide
- 技术报告 PDF（8 页）：https://github.com/123123213weqw/ai4climate-nepal-landslide/blob/main/paper/paper.pdf
- 一键复现：bash scripts/run_all.sh（数据自动下载，无需任何 API key）

四、数据来源与 AI 使用说明

全部使用无密钥开放数据：NASA Global Landslide Catalog、AWS Open Data Terrarium 地形瓦片（SRTM）、NASA POWER 降水气候态、Natural Earth 河流、GEM 全球活动断层库、OpenStreetMap 道路。逐项来源与许可见仓库 README 与论文第 2 节。模型用途为易发性排序（灾害预警辅助研判），误判、不确定性与安全边界在论文 Limitations 一节中如实讨论。

五、目标用户、应用场景与验证计划

- 目标用户：市政灾害管理办公室、社区减灾 NGO、道路与徒步路线管护方；
- 场景：季风前的巡护重点区排序、安置点与路线的风险沟通底图；
- 90 天验证计划：将走廊地图交给 Gandaki 省 2–3 个市的风险管理部门试用，用下一季风季的事件报告做命中率回测；同步推进概率校准与 SHAP 可解释性，使阈值对用户有明确含义。

演示视频（≤3 分钟）：[待补充链接]

#AI4Climate #气候变化 #尼泊尔 #滑坡 #机器学习 #开放数据
