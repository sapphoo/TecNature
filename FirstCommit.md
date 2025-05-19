# 第一次提交代码到github相关配置 
新建一个repository（我命名为test）之后，一切都是默认配置，会看到下列提示。

Quick setup — if you’ve done this kind of thing before
在github桌面应用设置  or http/ssh	git@github.com:sapphoo/test.git（ssh地址）   
Get started by creating a new file or uploading an existing file. We recommend every repository include a README, LICENSE, and .gitignore.

…or create a new repository on the command line


我们作为一个有尊严的程序员，当然是选择使用git的命令行来提交代码，值得一提的是，建议使用ssh比较方便，用http验证比较麻烦。

所以，首先来配置ssh密钥。

## ssh密钥配置
git bash中运行下面代码生成公钥，这里使用的是ed25519加密算法
```
ssh-keygen -t ed25519 -C "youremail@example.com"
```
回车，会在你的 _c/Users/用户名/.ssh/_ 文件夹下生成一对私钥和公钥（.pub文件是公钥），把公钥的内容拷贝，到 https://github.com/settings/keys 里选择sign key和authentication key新建。

运行代码段
```
echo "# test" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:sapphoo/test.git
git push -u origin main
```
会返回
```
ssh: connect to host github.com port 22: Connection refused
fatal: Could not read from remote repository.
```
这是因为ssh默认使用了22端口，22端口是默认SSH端口，通常会被防火墙封锁，无法科学上网，因此我们应该使用HTTPS（超文本传输安全协议）​​服务的默认端口443，配置方法如下：
_c/Users/用户名/.ssh/_ 文件夹下新建"config"文件，填入以下内容注意，文件 __无后缀名__
```
Host github.com
  HostName ssh.github.com
  User git
  Port 443
```

设置成功后，在 Git Bash 或 PowerShell 输入：
```
ssh -T git@github.com
```
如果出现
```
Hi yourname! You've successfully authenticated, but GitHub does not provide shell access.
```
说明配置成功！就可以愉快的重新运行push命令了。


## ssh两个密钥配置
聪明的你可能发现了，我的auth和sigh使用的ssh key是同一个，这样没有问题，但显然不够好，因为这两个key的功能是不一样的。
__认证Authentication Key__ 用于 SSH 连接 GitHub（git push、git fetch等）。
__签名Sign Key__ 用于 commit/release/tag 等的数字签名。

### 1.生成两对密钥
git bash中运行下面代码生成公钥，这里使用的是ed25519加密算法
```
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_auth -C "your_email@example.com"
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_sign -C "your_email@example.com"
```
回车，会在你的c/Users/用户名/.ssh/文件夹下分别生成两对私钥和公钥（.pub文件是公钥），把公钥的内容拷贝，到https://github.com/settings/keys里分别选择sign key和authentication key新建。

### 2.配置认证key
编辑 ~/.ssh/config 文件（Windows 路径：C:\Users\你的用户名\.ssh\config）,空格必须：

```
Host github.com
    HostName ssh.github.com
    User git
    Port 443
    IdentityFile ~/.ssh/id_ed25519_auth
```

### 3.配置签名key
设置 Git 采用 SSH 签名方式
```
git config --global gpg.format ssh
```
设置签名 key 路径
```
git config --global user.signingkey ~/.ssh/id_ed25519_sign.pub
```
开启自动签名
```
git config --global commit.gpgsign true
```

SSH config 里 IdentityFile 决定认证 key（推拉代码用）。
Git 配置里 user.signingkey 决定签名 key（签 commit/tag 用）。
两者可以完全不同，互不影响。


