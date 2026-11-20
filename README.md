# idea-verify — 验证集合仓库

一个子目录一个独立验证项目:每个子目录自包含(README / 依赖清单 / 脚本 / 数据 / 结果产物),互不依赖;通用忽略规则在仓库根 [.gitignore](.gitignore)。新增验证项目时直接新建兄弟目录。

## 目录

| 子项目 | 主题 | 入口 |
|---|---|---|
| [`dlpack-verify/`](dlpack-verify/README.md) | MindSpore → PyTorch 渐进迁移:DLPack 混合过渡态验证(零拷贝交换 + sens 梯度桥接,昇腾 910B Atlas A2) | [README](dlpack-verify/README.md) |

## 通用约定

- **依赖清单放各自子目录内**(`requirements-*.txt`),不共享——不同验证项目的框架与版本互不牵制;
- 实验产物(`results/`、`charts/`、`docs/`)随子项目入库,作为可审计证据;
- 仓库根 `.gitignore` 只放全仓通用规则(`__pycache__/`、`*.pyc`、`results_*.xlsx` 等),
  子项目内不另建 `.gitignore`;
- 一键执行入口在各子项目的 `scripts/run_all.py`(如有),最终交付报告由各子项目生成。
