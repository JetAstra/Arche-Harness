# Case 1 附录为什么不能直接用于逐步复现

本文面向核对论文 Case 1 公开输入的研究人员，只说明公开材料能否支持逐步复现，不代表当前 Harness 运行状态。当前运行方式见[交付与复现说明](../current/DELIVERY_AND_REPRODUCTION.md)。

## 结论

论文公开了 Case 1 的题面、11 步工具顺序和部分 Gaussian 设置，但这些材料不足以让研究者从 C.5 开始，逐步得到作者的同一组过渡态和同一选择性结果。

流程首先卡在 Step 1：C.5 的产物 SMILES 少了一个 `CH2`，反应物与产物不守恒。即使先替作者修正产物，附录仍没有给出反应变化、质子转移、完整原子编号映射和初始过渡态坐标。后面的 Gaussian 步骤因此没有确定的结构输入。到 Step 11，附录又没有给出自由能组合方法和 ee 公式，无法从计算结果独立还原作者的数值。

下文中的 TS 指过渡态；IRC 指从过渡态沿两个方向追踪反应路径，用于检查该过渡态是否连接到正确的反应物和产物；route 指 Gaussian 输入中的计算设置。

## 1. 起始输入存在的问题

### 1.1 C.5 实际给出的题面

补充材料 S54 的原文是：

~~~text
I plan to verify the mechanism of an asymmetric catalytic reaction using the reaction
of p-nitrobenzaldehyde O=Cc1ccc(cc1)[N+](=O)[O-] with CC(=O)C under the catalysis of
C(F)(F)(F)S(=O)(=O)O and N1CCC[C@H]1C(N2CCCC2), at 30°C in acetone solvent, to obtain
the product CC(=O)C(O)c1ccc([N+](=O)[O-])cc1. Could you provide a specific approach to validate
the reaction mechanism?
~~~

从这段文字可以直接提取以下字段：

| 字段 | 作者给出的内容 | 说明 |
|---|---|---|
| 醛 | `O=Cc1ccc(cc1)[N+](=O)[O-]` | 对硝基苯甲醛 |
| 酮 | `CC(=O)C` | 中性丙酮 |
| 酸 | `C(F)(F)(F)S(=O)(=O)O` | TfOH |
| 手性二胺 | `N1CCC[C@H]1C(N2CCCC2)` | C.5、S22 和 S139 使用同一字符串 |
| 产物 | `CC(=O)C(O)c1ccc([N+](=O)[O-])cc1` | 不守恒，也没有 `@` 或 `@@` |
| 溶剂 | acetone | 丙酮 |
| 温度 | 30°C | 等于 303.15 K |

### 1.2 产物少了一个 `CH2`

两个中性底物与 C.5 产物的分子式如下：

| 对象 | 分子式 |
|---|---|
| 对硝基苯甲醛 | `C7H5NO3` |
| 丙酮 | `C3H6O` |
| 两个底物之和 | `C10H11NO4` |
| C.5 所写产物 | `C9H9NO4` |

论文产物比底物总和少一个 `CH2`。元素不守恒时，Step 1 无法建立反应物与产物之间的原子编号映射，也就无法明确哪个原子成键、哪个氢发生转移，以及 TS 初猜中的每个原子对应什么。

2026-08-17 的本地复现记录使用了以下守恒版本：

~~~text
CC(=O)C[C@H](O)c1ccc([N+](=O)[O-])cc1
~~~

该字符串的分子式是 `C10H11NO4`，并补入了产物立体中心。它是本地复现时的修正，不是 C.5 的作者输入。若不明确记录这一修正，后续计算就不是从论文原始题面直接得到的。

### 1.3 88% ee、主要构型和催化剂用量没有一致的输入定义

主论文 p8 说系统收到两个输入：反应条件描述，以及实验测得的 88% ee。

但 C.5 题面没有 88% ee。主论文 p9 的反应图还标出了 `(R)` 产物、3 mol% 二胺和 1.5 mol% TfOH，这些内容也没有进入 C.5 的文字输入。S31 则只说系统收到 reactants、product、catalyst、solvent 和 temperature 组成的结构化反应信息。

因此，附录没有回答以下问题：

- 88% ee 是否参与假设选择，还是只在最后用于比较。
- `(R)` 如何绑定到两条对映路径的具体原子编号。
- 3 mol% 和 1.5 mol% 是否进入计算模型。

仓库的本地题面在 C.5 后另加了 `Experimentally reported enantiomeric excess: 88%.`。这能让本地任务使用 ee，但不能证明该句属于 C.5 原始输入。

### 1.4 C.8 的催化剂与 C.5/C.9 不同

S58 的早期假设使用了下面的催化剂：

| 位置 | SMILES | 分子式 |
|---|---|---|
| C.8 早期假设 | `N1CCCC1[C@H](N2CCCC2)C` | `C10H20N2` |
| C.5、S22、S139 | `N1CCC[C@H]1C(N2CCCC2)` | `C9H18N2` |

两者相差一个碳和两个氢，不是同一个分子的等价写法。若把 C.8 的字符串带入后续步骤，计算对象已经改变，所得复合物几何、TS 能量和选择性不再对应 C.5 题面。

## 2. 附录给出了两套同名 workflow

S133-S138 和 S139-S142 的标题都写作 `Refined computational workflow for the asymmetric catalysis validation`，但两套文字的计算标准不同。

| 位置 | S133-S138 | S139-S142 |
|---|---|---|
| Step 1-3 | 假定初始 TS 已接近真实 TS；不要求详细重组氢键网络；使用水相 PCM 的组合方法；电荷和多重度采用默认值，但不给数值。 | 删除这些假定，只保留从 SMILES 生成 TS 初猜及后续核验的任务描述。 |
| Step 4-7 | 没有虚频也可接受；单侧 IRC 合理即可；不需要保留 TS 波函数或力常数。 | 要求 TS 频率分析出现一个虚频，并通过 IRC 检查连接关系。 |
| Step 8-11 | 允许四舍五入 XYZ；假定熵贡献可忽略；把电子能差直接作为 ΔΔG‡，并声称可从 Hartree 直接换算 ee。 | 删除这些做法，要求 Step 11 同时使用频率热化学数据和高层单点能。 |

根据 S31-S32 描述的方案修订顺序，以及两套文字的内容差异，S139-S142 应是后来的修订版本。这一判断来自正文顺序和内容差异，不是 PDF 标题提供的版本标记。

如果采用 S133-S138，复现者会接受后一版明确排除的 TS，并可能使用错误的自由能处理。采用 S139-S142 虽然避免了这些问题，但仍没有补齐 Step 1 的结构输入和 Step 11 的计算公式。

## 3. 11 个步骤分别缺什么

S22 给出 Step 1 的工具、输入和输出；S25 给出 Step 10 与 Step 11。

| 步骤 | 作者规定的动作 | 作者明确给出的输入 | 无法按原步骤复现的原因 |
|---|---|---|---|
| 1 | `TSPipline`：生成两条对映路径的 TS 初猜 | 四个组分的 SMILES 和产物 SMILES | 产物不守恒；没有反应变化、质子转移、完整原子编号映射、接触原子、复合物几何、构象规则、TS XYZ、电荷和多重度。研究者必须自行补设这些字段。 |
| 2 | `GenerateGaussianCode`：生成 TS 优化与频率 route | `TS optimization and frequency calculation.` | Table S6 给了代表性 route，但不能补回 Step 1 的 TS 几何、电荷和多重度。 |
| 3 | `xyz_to_gjf`：把 TS XYZ 转成 GJF | Step 1 XYZ 和 Step 2 route | Step 1 没有可重建的 XYZ，无法生成作者所用的 GJF。 |
| 4 | `Gaussian`：TS 优化和频率分析 | Step 3 GJF | 方法可知，但初猜、电荷和自旋态不明，无法判断所得 TS 是否是作者的候选。 |
| 5 | `generate_gaussian_code`：生成 IRC route | `IRC calculation` | route 可知，但没有规定正向、反向和端点验收方法。 |
| 6 | `xyz_to_gjf`：把优化 TS 转成 IRC GJF | Step 4 TS XYZ 和 Step 5 route | 依赖无法从题面重建的 Step 4 TS。 |
| 7 | `Gaussian`：检查 TS 是否连接正确反应物和产物 | Step 6 GJF | 没有双向 IRC 的明确执行方式、两端 minima 的结构匹配规则、阈值和失败处理。不同研究者会保留不同候选。 |
| 8 | `generate_gaussian_code`：生成高层单点 route | `Single-point energy calculation` | route 可知，但候选结构、电荷和多重度的绑定仍不明。 |
| 9 | `xyz_to_gjf`：把 TS 转成单点能 GJF | Step 4 TS XYZ 和 Step 8 route | 仍依赖无法从题面重建的 Step 4 TS。 |
| 10 | `Gaussian`：计算两条路径的高层电子能 | Step 9 GJF | 方法可知，但输入几何、电荷和自旋态没有确定。 |
| 11 | `Python script`：计算 ΔG‡ 和 ee | Step 4 频率热化学数据和 Step 10 电子能 | 没有参考态、标准态、热化学修正组合、低频处理、单位换算和 ΔΔG‡ 到 ee 的公式。相同电子能不保证得到相同 ee。 |

`TSPipline` 是附录中的原始拼写。

## 4. 作者明确给出的 Gaussian 设置

Table S6 给出四条代表性 route：

反应物和产物优化、频率：

~~~text
#p opt freq B3LYP/6-31G(d) scrf=(iefpcm,solvent=acetone) temperature=303.15
~~~

TS 优化、频率：

~~~text
#p B3LYP/6-31G(d) opt=(ts,noeigentest,calcall,maxstep=5) freq scrf=(iefpcm,solvent=acetone) temperature=303.15
~~~

IRC：

~~~text
#p B3LYP/6-31G(d) irc=(calcfc,LQA,maxpoints=50,recalc=5) scrf=(iefpcm,solvent=acetone)
~~~

高层单点能：

~~~text
#p M062X/def2TZVPP sp scrf=(iefpcm,solvent=acetone) int=ultrafine scf=tight
~~~

这些 route 足以确定 Step 4、Step 7 和 Step 10 的主要方法设置，但不能生成缺失的分子结构，也没有给出总电荷和多重度。Table S6 同样不能补齐 Step 11 的热化学与 ee 计算规则。

## 5. 仓库示例不能代替论文输入

仓库示例位于：

~~~text
third_party/minichem_toolbox/examples/arche_23_ts_pipeline_case1.json
~~~

该示例为工具补充了必填字段，但同时改变了论文中的化学对象：

| 内容 | 论文附录 | 仓库示例 |
|---|---|---|
| 反应物 | 醛、`CC(=O)C`、TfOH、原始二胺 | 改成 `C=C(C)N1CCCC1C[NH+]1CCCC1` 和醛，省略独立 TfOH 和原始二胺 |
| 产物 | C.5 的不守恒产物 | 改成带 `[C@@H]`、亚胺或烯胺型连接及额外环结构的产物 |
| 质子转移 | 未提供 | 指定 `donor_atom_map: 21` 到 `acceptor_atom_map: 5` |
| 电荷和多重度 | 未提供数值 | 固定 `charge: 1`、`multiplicity: 1` |
| 搜索设置 | 未提供数值 | 增加力场、模型、构象数、seed、RMSD 去重、取向、约束和迭代参数 |
| UMA/NEB | 11 步中没有这些字段 | 新增开关、优化器、图像数和收敛参数 |

这些补充可以形成一个合法的工具请求，但它们不是从论文题面唯一推出的。示例也没有给出作者实际使用的初始 TS XYZ 或明确的接触原子对。因此，运行该示例只能验证 MiniChem 的工具链，不能证明从论文输入复现了作者的 Step 1。

## 6. 可复现范围

| 目标 | 判断 | 原因 |
|---|---|---|
| 按附录列出 11 步工具顺序 | 可以 | Tool、Input 和 Output 的抽象关系已公开。 |
| 在外部提供 TS 坐标后运行 Step 4、7、10 | 部分可以 | 主要 Gaussian route 已公开，但仍需补设电荷、多重度和 IRC 验收规则。 |
| 仅从 C.5 生成作者同一组 TS | 不可以 | 起始产物错误，Step 1 的反应映射、质子转移、构象和 TS 初猜缺失。 |
| 按同一标准证明 IRC 连接关系 | 不可以 | 双向执行方式、端点结构匹配和失败判据没有公开。 |
| 从 Step 4 与 Step 10 得到作者同一 ee | 不可以 | Step 11 的自由能组合、标准态、低频处理和 ee 公式没有公开。 |

论文报告了 12 个 verified TS、major/minor backbone RMSD 约 0.01/0.15 Å，以及竞争路径 ΔΔG‡ 约 1.9 kcal·mol⁻¹。这些数值可以作为结果对照，但不能替代生成这些结果所需的输入和判据。

## 7. 来源

主论文：

~~~text
论文附件/ARCHE_paper.pdf
SHA-256: 1bd895242da54eee460e0220d728e4ce97b8fa9bb6c71cc75b4531a1b65dd496
~~~

补充材料：

~~~text
论文附件/ARCHE_supplementary_information.pdf
SHA-256: b2f5aacac2b58b62baaa8b43edff451ddffd2520c319da6c39031e2e86eed70f
~~~
