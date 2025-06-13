# Git常用命令
![alt text](img/gitcommand.png)

## Git的原理
存储项目随时间变化的快照

## Git常用场景
### 1.将本地文件夹中的内容上传到新建的github空仓库中
编辑好.gitignore文件之后，执行以下命令
```
# 初始化git
git init 
# 添加文件
git add
#如果有添加上了不想添加的，删除
git rm --cached [文件名]
#查看文件的状态
git status
# 提交更改
git commit -m 'message'
# 关联远程仓库
git remote add origin [ssh地址]
# 一般新建的仓库的主分支默认为main
git branch -M main
# 推送
git push -u origin main
```
### 2.新建仓库后想要在本地电脑编辑项目
在github上面新建一个repository成功，并建立了readme.md之后，想要在本地文件夹里对repository进行编辑，并与github保持同步。
```
git clone [ssh地址/https地址]
```
### 3.本地和远程均有修改
```
# 先同步拉取远程的内容
git pull origin main
# 处理可能出现的冲突
如果本地改动和远程改动涉及同一个文件、同一行，Git 会提示“冲突”（conflict）。
打开冲突文件，按照 <<<<<<<, =======, >>>>>>> 标记手动合并内容。
合并后，记得 git add 冲突文件，再 git commit。
# 冲突解决后，push
git push origin main

```
## Git常用命令


### git init 初始化
开始跟踪当前文件夹的diff（变化）

### git add 添加要提交的东西
```
git add .
git add README.md
```

### git commit 提交
给你的代码照一个快照，以便可以穿越回任何一个快照当时的状态
```
git commit -m "first commit"
```
####  HEAD 最近的一次提交

### git push 推送

### git branch 分支
创建代码宇宙的一个新的支线
```
git branch [newBranch]
```

### git merge 合并
```
git merge newBranch
```
如果发生conflict（冲突），则需要解决

## 一般常用命令组合
### 和remote远端同步
```
git pull

```
### 修改或增加后添加-提交-推送
```
git add .
git commit -m "[提交说明]"
git push
```
