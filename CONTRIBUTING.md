# 维护课程与实验

章节名称及主要内容在 `web/module-spec.mjs`，专题讲解、实现细节、实操案例和图解定义分别在 `web/module-lessons.mjs`、`web/implementation-notes.mjs`、`web/module-cases.mjs`、`web/module-figures.mjs`；课程基础与其他扩展位于 `web/course.mjs`、`web/deep-dive.mjs` 和 `web/applied-practice.mjs`。为避免重复，`web/module-selection.mjs` 决定旧扩展中哪些小节进入课程；新增小节时应核对选择结果。修改章节正文、题目、答案或术语后，运行 `node scripts/build.mjs`，同步生成网页 JSON、Markdown 课程和下载材料。不要只修改生成文件。

数学实现位于 `web/dist/math.mjs` 与 `labs/core_math.py`。新增公式实验时应说明符号、假设、适用边界，并使用可手算的案例验证。交互结果必须标为教学计算，不能伪装成真实模型实验。

提交前运行 README 中的检查。涉及页面交互时，在浏览器验证实际用户流程；涉及响应式布局时，检查窄屏阅读和键盘可操作性。自动测试只覆盖其明确的断言，不能代替内容审阅。

学习记录是用户本地数据。导入应校验格式、保留已有笔记、清楚报告冲突，不能把不可信内容直接插入 HTML。

引用优先使用原始论文和官方文档。描述方法时区分原始算法、实现变体和当前库版本，新增实际训练结果时附上配置、数据版本、成本与独立评测。

私人站点的部署快照位于本地忽略的 `.site-deploy/`，其 Git 历史与项目主仓库独立。发布前把 `web/` 内容同步到该目录（保留该目录的 `.git`），使用既有 `.openai/hosting.json` 中的 Site ID 完成来源提交与发布。不要提交或保存短期凭据。
