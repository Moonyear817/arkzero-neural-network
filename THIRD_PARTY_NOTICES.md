# 第三方来源与使用边界

本项目的 Python 引擎独立编写。没有复制 Unity/C# 实现、游戏函数、Lua 脚本、贴图、声音或游戏二进制到运行包。

- `data/real/` 是 Kengxxiao/ArknightsGameData 的公开 JSON 研究快照，源 URL、提交和 SHA-256 见 `data/manifest.json`。游戏数据的原始权利属于原权利人；GitHub 可访问不等于获得任意再分发许可。
- `research/sources/` 是本地研究证据，不是引擎依赖。ArknightsMapViewer 受 GPL-3.0 约束；其他所查主仓库未检测到明确根许可证，不按宽松许可处理。引用只用于规则、概念和出处记录。
- IPA 只读取用户指定的原文件；本项目不包含 IPA 副本、解密工具或官方客户端依赖。
- 打包仅包含 `arknights_sim*`。如未来发布仓库、训练数据或产品，应分别处理研究缓存与数据的分发权利，不能给整目录套用自有代码许可。

本阶段未选择项目对外开源许可证，也没有发布或推送。

## PRTS 干员资料（2026-09-15）

来源：[PRTS 干员一览](https://prts.wiki/w/干员一览)及其 431 个干员页面，由 PRTS Wiki 贡献者维护。`data/prts/` 保存列表和页面原文缓存；`data/real/operator_catalog.json` 保存逐页链接、修订号、修订时间、别名、列表字段及完整 wikitext。导入器仅整理和关联数据，不将页面文字执行为代码。

PRTS 页脚声明：游戏图片、动画、音频、文本原文版权属于上海鹰角网络科技有限公司及其关联公司；除另有声明外，网站其他内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)（署名、非商业性使用、相同方式共享）。此许可适用于相应来源内容，不自动适用于独立引擎代码。保留本说明及逐条来源归属。当前未下载图片、音频或动画。

## PRTS.Map 离线资料

来源 https://map.ark-nights.com/ ，2026-09-15 下载。data/maps/download_manifest.json 保存来源及文件校验值；site.html、site_bundle.js 仅为索引来源缓存，不作为本项目引擎执行。游戏数据及相关素材权利属于鹰角网络及相应权利人；本项目不对第三方资料主张独立版权，也不假定网站代码与游戏数据采用相同许可。

## 本地 IPA 与资源读取工具（2026-09-15）

用户提供 IPA：com.hypergryph.arknights_2.7.71_und3fined.ipa，版本 2.7.71 / build 66。仅离线读取资源参数，原始文件未修改，未执行客户端程序或其中序列化动作。资源版权归原权利人。清单及 SHA256 见 research/mechanics/ipa/manifest.json。

资源读取依赖 UnityPy 1.25.3（MIT）和 lz4。scripts/lz4ak_read.py 改编自 isHarryh/Ark-Unpacker v5.x 的 src/lz4ak/Block.py，Copyright (c) 2022-2026 Harry Huang，BSD 3-Clause。完整许可证保存在 research/mechanics/unpacker_reference/LICENSE，原版参考源同目录保留。
