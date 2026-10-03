# 儿童语义 · 词义漂移探针（v0 → v4）

> **缘起（主人 1003）**：「密度图作为之后，再考虑**儿童语义的变化情况**」→ 主人圈 **B · 词义漂移**。
> 语料：本地 CHILDES（Brown 214 + Bernstein 50，共 264 份 `.cha`，**真实月龄**，全带 `%mor`/`%gra`）。
> 本目录是今天这条探针链的**全程留痕**（含每一次失败——**裸探针律**：便宜探针该杀就杀）。

## 一 · 探针链（一版杀一版）

| 版 | 探针 | 裁决 |
|---|---|---|
| **v0** `drift.py` | 同话轮共现：y = P(生物词\|同话轮) − P(物件词\|同话轮) | **循环**——dog∈生物表、ball∈物件表 ⇒ 锚点恒 +1/−1 是拿自己证自己 |
| **v1** `drift2.py` | 句法探针：P(名词后邻动词) − P(带 `-Acc`)，用 `%mor` | 有**全局下行混淆**（几乎所有词随龄漂向 `-Acc`）→ 锚点反而失形 |
| **v2** `drift3.py` | 修正 v0（探针排除自身）＋报孤立率 | 真问题浮出：**孤立率 85–100%**——儿童话轮几乎都是**单词句**（"moon." "ball."）⇒ 词级共现探针**判死** |
| **v3** `drift4_space.py` | 不信单词共现：**每龄段建 PPMI+SVD 分布语义空间**（会话内滑窗），看**近邻集**漂移 | ✅ 读出真漂移（见下） |
| **v4** `drift5_report.py` | 跨龄空间 **Procrustes 对齐** ＋ 自身余弦稳定性 | 对齐**弱**（见 F5）⇒「轴转动」尚不可定量 |

## 二 · 预注册与裁决

预注册（先冻后用，见各脚本 docstring）：
- **P1** moon 泛灵坐标随龄**下降**（承 LADDER-1 泛灵→去泛灵）→ **不立**（moon 近邻无本体信号）
- **P2** moon 近邻若漂，须指向**体裁**（童谣/童书）而非本体 → **立**
- **P3** 相邻龄段空间对齐后**轴有旋转** → **不可测**（残差近随机）

## 三 · 读数

**F1｜儿童话轮太短，词级共现探针无效。** 各目标词孤立率 71–100%；moon/sun/star/light 尤甚。
**F2｜分布语义近邻（v3）读出真漂移**（会话内滑窗，龄段 13–23 / 24–33 / 34–43 / 44–62）：

| 词 | 近邻随龄 | 判 |
|---|---|---|
| **ball** | `hit · mommy` → `caught · throw · playing` → `bowling · bounce · knock` | 玩法语境逐龄**专业化** |
| **car** | `train · caboose · track` → `rambler · racing · gas` → `drive · park · truck` | 从玩具车→真车驾驶 |
| **milk** | `bottle · drink · spilled` → `fresh · cream · cold` → `juice · pour · ate` | 从物→饮食流程 |
| **tree** | `christmas · cookies` → `plant · climb` → `climb · rabbit · owl` | 从节日符号→植物/生态 |
| **dog** | `barking · doggie` → `doggie · tramp` → `puppy · cat · dogs` | 恒在动物族（稳） |
| **moon** | `sun · witch · game · sky` → `mailbox · mailman · mail · dark` → `stars · ocean · chase · whales` | **随书走**：witch/mailman/whales ⇒ **体裁驱动**（P2 立）|

**F3｜moon 的"泛灵"是童书童谣的折光**，不是儿童本体论——`witch`（女巫书）、`mailman/mailbox`（收信书）、`chase/whales`（海洋书）。
**F4｜锚词 dog/ball 的稳与 moon 的动，是同一探针下两类**（世界模型里，动物/物件有稳固位，自然现象词无稳固位）。
**F5｜跨龄空间对齐弱**：Procrustes 残差 0.89–0.94，随机正交 ≈ 1.27–1.38 ⇒ 仅约 **30% 结构共享**；
即便把语境扩到**全场（含母亲输入，token×2）**也只到同一水平 ⇒ **「世界模型的轴怎么转」这份语料答不了**（照登）。

## 四 · 结论（一句）

> **这条语料能看"词换了什么邻居"，还看不了"轴转了多少"。**
> 词义漂移的**定性证据到手**（ball/car/milk/tree 四条真轨迹 ＋ moon 的体裁假说），
> 但**定量"轴变化"须换弹药**：更多 token、更长话轮、或 zho 纵向库（Zhou3，候 ECNU/张义宾线）。

## 五 · 下一步（候主人点）

1. **做图**：把 F2 的四条近邻轨迹画成「词义漂移 · 近邻时间线」（诚实标注为定性）。
2. **加燃料**：把 `corpus` 里 Gutenberg/DK 等层并进来扩 token；或改**会话级共现**再试对齐。
3. **等真弹**：收 Letter 036 音（Yibin/MacW，今日到期）→ zho Zhou3 中文纵向库 → 中文臂对撞。
4. **接回密度图**：把本探针的「词在语义空间里的位置」接上那张 PyVista 密度地形（主人既定「密度图作为之后」）。

## 目录

```
word_drift/
  drift.py  drift2.py  drift3.py  drift4_space.py  drift5_report.py   ← 探针链（含失败）
  table.md  table2.md  table3.md  neighbors.md  report5.md  drift*.json  ← 各版读数
```

图/文：lola · 2026-10-03
