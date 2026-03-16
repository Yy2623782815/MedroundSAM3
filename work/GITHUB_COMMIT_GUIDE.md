# 如何将最新代码提交到 GitHub

以下是在本地仓库中最常用、最稳妥的提交流程：

1. 查看当前状态

```bash
git status
```

2. 拉取远端最新代码（避免直接推送冲突）

```bash
git pull --rebase origin <你的分支名>
```

3. 添加要提交的文件

```bash
git add .
```

> 如果只想提交部分文件，请改为：
>
> ```bash
> git add 路径/文件名
> ```

4. 提交代码并写清楚提交信息

```bash
git commit -m "feat: 简要描述本次改动"
```

5. 推送到 GitHub

```bash
git push origin <你的分支名>
```

---

## 常见问题排查

### 1) `nothing to commit, working tree clean`
说明你没有新改动，或者改动还没保存。

### 2) 推送被拒绝（`non-fast-forward`）
先同步远端再推送：

```bash
git pull --rebase origin <你的分支名>
git push origin <你的分支名>
```

### 3) 提交信息写错了
如果是最近一次提交且还没推送：

```bash
git commit --amend -m "新的提交信息"
```

---

## 一个可直接复制的最小流程

```bash
git status
git add .
git commit -m "chore: update latest changes"
git push origin $(git branch --show-current)
```
