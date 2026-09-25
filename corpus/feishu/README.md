# 园账·飞书阵地

- **晨圈台账（应用自持，写权全通）**：base `GARDEN_BASE` / 表 `GARDEN_TABLE`（见 ~/.config/pocket/feishu.env）
  链接: https://my.feishu.cn/base/OTPFbjn7MaRWSis9KOYclUNTnte （base 现名"园笔探针台-可删"，主人 UI 一键可改为"园账·晨圈台账"；drive 改名 API 无权限，不强求）
  另有空壳"园笔试验田-诊断用"一个（API 无删权，UI 顺手可清，留着无害）
- **存储周报 base（只读历书）**：42 环周报索引，读路常驻；写不申请——各安其分
- 桥器: bin/feishu-bitable.sh（tables/fields/recs/add/upd/del/garden/gadd/gupd；token 自缓存）
- 备份: bitable-backup-2026-09-25.json（sha 25c87f25c0a8）
- 诊断定案: 91403 病在共享层非应用层——二分法（自持 base 能写 = 权限包完好）
- **2026-09-25 方案一全装**: @larksuite/cli(官方CLI v1.0.96) + 用户身份登录(Madas, base 全读写 scope) + 28 项 lark-* 技能入 ~/.pi/agent/skills
  共享表写路打通(用户越闸)，台账 12 行迁至本 base「✅任务管理」；自持 base 降为备份(指向见 env GARDEN_BASE/_BACKUP)
  日常读写首选: lark-cli base +record-list/+record-upsert --as user ; 桥器 feishu-bitable.sh 保留为租户通道
